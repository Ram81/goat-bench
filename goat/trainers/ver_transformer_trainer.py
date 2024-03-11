import copy
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from habitat import logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat_baselines import VERTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    get_free_port_distributed,
    get_main_addr,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ver.environment_worker import (
    build_action_plugin_from_policy_action_space,
    construct_environment_workers,
)
from habitat_baselines.rl.ver.preemption_decider import PreemptionDeciderWorker
from habitat_baselines.rl.ver.report_worker import ReportWorker
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.timing import Timing
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.rl.ver.worker_common import (
    InferenceWorkerSync,
    WorkerBase,
    WorkerQueues,
)
from habitat_baselines.utils.common import (
    cosine_decay,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
)
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR

from goat.trainers.inference_worker_with_kv import (
    InferenceWorkerWithKV,
    InferenceWorkerWithKVProcess,
)
from goat.trainers.ver_rollout_storage_with_kv import (
    VERRolloutStorageWithKVCache,
)

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass


@baseline_registry.register_trainer(name="ver_transformer")
class VERTransformerTrainer(VERTrainer):
    """
    If the actor critic is NOT a transformer, the trainer will be the same as the
    VERTrainer.
    """

    def _init_train(self, resume_state):
        r"""Copy of VERTrainer._init_train, but the rollout storage will use the
        VERRolloutStorageWithKV rollout class if the policy is transformer-based. This
        method will also change the inference workers to a different variant if the
        policy is transformer-based. These different workers will provide a KV cache
        during onling inference instead of the usual RNN hidden state."""
        if self._is_distributed:
            local_rank, world_rank, _ = get_distrib_size()

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique
                # seeds
                self.config.habitat.seed += (
                    world_rank * self.config.habitat_baselines.num_environments
                )

        random.seed(self.config.habitat.seed)
        np.random.seed(self.config.habitat.seed)
        torch.manual_seed(self.config.habitat.seed)

        self.mp_ctx = torch.multiprocessing.get_context("forkserver")
        self.queues = WorkerQueues(
            self.config.habitat_baselines.num_environments
        )
        self.environment_workers = construct_environment_workers(
            self.config, self.mp_ctx, self.queues
        )
        [ew.start() for ew in self.environment_workers]
        [ew.reset() for ew in self.environment_workers]

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized VER+DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )
        else:
            logger.info("Initialized VER")
            tcp_store = None

        self._last_should_end_val = None
        self._last_should_end_calc_time = 0

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        self._all_workers: List[WorkerBase] = []

        if self._is_distributed:
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            world_rank = 0
            world_size = 1

        self._my_t_zero = time.perf_counter()
        self.preemption_decider = PreemptionDeciderWorker(
            self.mp_ctx,
            get_main_addr(),
            get_free_port_distributed("preemption", tcp_store),
            world_rank,
            world_size,
            self.config,
            self.queues,
            self._my_t_zero,
        )

        has_report_resume_state = (
            resume_state is not None
            and "report_worker_state" in resume_state["requeue_stats"]
        )
        run_id = None
        if (
            has_report_resume_state
            and resume_state["requeue_stats"]["report_worker_state"] is not None
        ):
            run_id = resume_state["requeue_stats"]["report_worker_state"][
                "run_id"
            ]

        self.report_worker = ReportWorker(
            self.mp_ctx,
            get_free_port_distributed("report", tcp_store),
            self.config,
            self.queues.report,
            self._my_t_zero,
            self.num_steps_done,
            run_id=run_id,
        )

        if has_report_resume_state:
            self.report_worker.load_state_dict(
                resume_state["requeue_stats"]["report_worker_state"]
            )

        init_reports = [self.environment_workers[0].get_init_report()]

        action_space = init_reports[0]["act_space"]

        self.policy_action_space = action_space
        self.orig_policy_action_space = None

        [
            ew.set_action_plugin(
                build_action_plugin_from_policy_action_space(
                    self.policy_action_space
                )
            )
            for ew in self.environment_workers
        ]
        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda", self.config.habitat_baselines.torch_gpu_id
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.exists(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        actor_obs_space = init_reports[0]["obs_space"]
        self.obs_space = copy.deepcopy(actor_obs_space)
        self.ver_config = self.config.habitat_baselines.rl.ver

        self._setup_actor_critic_agent(ppo_cfg)
        """BEGINNING OF CHANGES RELATIVE TO VERTrainer._init_train"""
        self._is_transformer = getattr(
            self.actor_critic, "is_transformer", False
        )
        """END OF CHANGES RELATIVE TO VERTrainer._init_train"""
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])

        rollouts_obs_space = copy.deepcopy(self.obs_space)
        if self._static_encoder and hasattr(self.actor_critic, "net"):
            self._encoder = self.actor_critic.net.visual_encoder
            rollouts_obs_space = spaces.Dict(
                {
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **rollouts_obs_space.spaces,
                }
            )

        n_inference_workers = self.ver_config.num_inference_workers
        main_is_iw = not self.ver_config.overlap_rollouts_and_learn
        with inference_mode():
            storage_kwargs = dict(
                variable_experience=self.ver_config.variable_experience,
                numsteps=ppo_cfg.num_steps,
                num_envs=len(self.environment_workers),
                action_space=self.policy_action_space,
                recurrent_hidden_state_size=ppo_cfg.hidden_size,
                num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
                action_shape=action_shape,
                discrete_actions=discrete_actions,
                observation_space=rollouts_obs_space,
            )
            """BEGINNING OF CHANGES RELATIVE TO VERTrainer._init_train"""
            if self._is_transformer:
                tf_cfg = (
                    self.config.habitat_baselines.rl.policy.transformer_config
                )
                storage_kwargs["num_layers"] = tf_cfg.n_layers
                storage_kwargs["num_heads"] = tf_cfg.n_heads
                storage_kwargs["max_context_length"] = tf_cfg.max_context_length
                storage_kwargs["head_dim"] = tf_cfg.n_hidden // tf_cfg.n_heads
                storage_kwargs["recurrent_hidden_state_size"] = 1
                storage_kwargs["num_recurrent_layers"] = 1

                rollout_storage_class = VERRolloutStorageWithKVCache
            else:
                rollout_storage_class = VERRolloutStorage
            """END OF CHANGES RELATIVE TO VERTrainer._init_train"""

            self.rollouts = rollout_storage_class(**storage_kwargs)
            self.rollouts.to(self.device)
            self.rollouts.share_memory_()
            if self.ver_config.overlap_rollouts_and_learn:
                self.learning_rollouts = rollout_storage_class(**storage_kwargs)
                self.learning_rollouts.to(self.device)
            else:
                self.learning_rollouts = self.rollouts

            storage_kwargs["observation_space"] = actor_obs_space
            storage_kwargs["numsteps"] = 1

            self._transfer_buffers = (
                rollout_storage_class(**storage_kwargs)
                .buffers.slice_keys(
                    "rewards",
                    "masks",
                    "observations",
                    "episode_ids",
                    "environment_ids",
                    "actions",
                    "step_ids",
                )
                .map_in_place(lambda t: t.share_memory_())
            )[
                (
                    slice(0, len(self.environment_workers))
                    if self.ver_config.variable_experience
                    else 0
                )
            ]

        self.actor_critic.share_memory()

        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.agent.actor_critic.parameters()
                )
            )
        )
        self._iw_sync = InferenceWorkerSync(
            self.mp_ctx,
            n_inference_workers,
        )

        inference_worker_args = (
            n_inference_workers,
            self.config,
            self.queues,
            self._iw_sync,
            self._transfer_buffers,
            self.config.habitat_baselines.rl.policy.name,
            (self.config, self.obs_space, self.policy_action_space),
            self.device,
            self.preemption_decider.rollout_ends,
        )

        self._transfer_policy_tensors = list(
            self.actor_critic.all_policy_tensors()
        )

        self.inference_workers = [
            InferenceWorkerWithKV(
                self.mp_ctx, self._is_transformer, i, *inference_worker_args
            )
            for i in range(1 if main_is_iw else 0, n_inference_workers)
        ]
        if main_is_iw:
            self._inference_worker_impl = InferenceWorkerWithKVProcess(
                None,
                None,
                None,
                0,
                *inference_worker_args,
            )
            self._inference_worker_impl.set_actor_critic_tensors(
                self._transfer_policy_tensors
            )
            self._inference_worker_impl.set_rollouts(self.rollouts)
        else:
            self._inference_worker_impl = None

        for iw in self.inference_workers:
            # We send the policy weights and the rollouts
            # via a torch.multiprocessing.SimpleQueue instead
            # of in the constructor as otherwise the shared
            # cuda tensors don't get properly freed on
            # destruction which causes an error.
            iw.set_actor_critic_tensors(self._transfer_policy_tensors)
            iw.set_rollouts(self.rollouts)
            iw.start()

        ews_to_wait = []
        for i, ew in enumerate(self.environment_workers):
            ew.set_transfer_buffers(self._transfer_buffers)
            if i > 0:
                init_reports.append(ew.get_init_report())

            ew.wait_start()
            ews_to_wait.append(ew)
            if len(ews_to_wait) >= 4:
                [a.wait_sync() for a in ews_to_wait]
                ews_to_wait = []

        [a.wait_sync() for a in ews_to_wait]
        ews_to_wait = []

        if self._is_distributed:
            torch.distributed.barrier()
        [aw.start_experience_collection() for aw in self.environment_workers]
        self.report_worker.start_collection()

        self.timer = Timing()

        self._all_workers.extend(self.environment_workers)
        self._all_workers.extend(self.inference_workers)
        self._all_workers.append(self.report_worker)
        self._all_workers.append(self.preemption_decider)
