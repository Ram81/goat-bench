from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn
from torchvision import transforms as T

from goat.task.sensors import (
    CacheImageGoalSensor,
    ClipGoalSelectorSensor,
    ClipImageGoalSensor,
    ClipObjectGoalSensor,
    LanguageGoalSensor,
)


@baseline_registry.register_policy
class GoatHighLevelPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        config: "DictConfig",
        **kwargs,
    ):
        super().__init__(
            GoatHighLevelPolicyNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                config=config,
            ),
            action_space=action_space,
            policy_config=None,
            aux_loss_config=None,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names: List[str] = []
        for agent_config in config.habitat.simulator.agents.values():
            ignore_names.extend(
                agent_config.sim_sensors[k].uuid
                for k in config.habitat_baselines.video_render_views
                if k in agent_config.sim_sensors
            )
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            config=config,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        value, action, action_log_probs, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return value, action, action_log_probs, rnn_hidden_states


class GoatHighLevelPolicyNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        config: "DictConfig",
    ):
        super().__init__()

        print("Start initialization of GOAT high level policy.......")

        ovon_policy_cls = baseline_registry.get_policy(
            "PointNavResnetCLIPPolicy"
        )
        print(observation_space)
        print(dir(observation_space))
        low_level_policy_action_space = spaces.Discrete(6)
        ovon_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if k
                not in [
                    LanguageGoalSensor.cls_uuid,
                    CacheImageGoalSensor.cls_uuid,
                ]
            }
        )
        # TODO: Find a better way to do this
        # Set the config croco parameters to False explicitly!
        original_params = {}
        with read_write(config):
            original_params['use_croco'] = config.habitat_baselines.rl.policy.use_croco
            config.habitat_baselines.rl.policy.use_croco=False
            original_params['croco_adapter'] = config.habitat_baselines.rl.policy.croco_adapter
            config.habitat_baselines.rl.policy.croco_adapter=False
            original_params['add_instance_linear_projection'] = config.habitat_baselines.rl.policy.add_instance_linear_projection
            config.habitat_baselines.rl.policy.add_instance_linear_projection=False
            
        self.ovon_policy = ovon_policy_cls.from_config(
            config,
            ovon_obs_space,
            low_level_policy_action_space,
        )
        missing_keys = self.ovon_policy.load_state_dict(
            self.load_ckpt(
                "/srv/flash1/rramrakhya3/spring_2023/ovon/data/new_checkpoints/ovon/ver/resnetclip_rgb_text/seed_1/ckpt.121.pth"
            )
        )
        print("OVON missing keys: {}\n\n".format(missing_keys))

        lnav_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if k
                not in [
                    ClipObjectGoalSensor.cls_uuid,
                    CacheImageGoalSensor.cls_uuid,
                ]
            }
        )
        language_policy_cls = baseline_registry.get_policy(
            "PointNavResnetCLIPPolicy"
        )
        print("Pol ", language_policy_cls)
        self.language_policy = language_policy_cls.from_config(
            config,
            lnav_obs_space,
            low_level_policy_action_space,
        )
        missing_keys = self.language_policy.load_state_dict(
            self.load_ckpt(
                "/srv/flash1/rramrakhya3/fall_2023/goat/data/new_checkpoints/languagenav/ver/resnetclip_rgb_bert_text/seed_3/ckpt.18.pth"
            )
        )
        print("Language nav missing keys: {}\n\n".format(missing_keys))

        iinav_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if k
                not in [
                    ClipObjectGoalSensor.cls_uuid,
                    LanguageGoalSensor.cls_uuid,
                ]
            }
        )
        image_policy_cls = baseline_registry.get_policy(
            "PointNavResnetCLIPPolicy"
        )
        
        with read_write(config):
            config.habitat_baselines.rl.policy.use_croco = original_params['use_croco']
            config.habitat_baselines.rl.policy.croco_adapter = original_params['croco_adapter']
            config.habitat_baselines.rl.policy.add_instance_linear_projection=original_params['add_instance_linear_projection']
        self.image_policy = image_policy_cls.from_config(
            config,
            iinav_obs_space,
            low_level_policy_action_space,
        )
        missing_keys = self.image_policy.load_state_dict(
            self.load_ckpt(
                # "data/new_checkpoints/iin/ver/resnetclip_rgb_vc1_image/seed_1/ckpt.70.pth"
                "/srv/flash1/gchhablani3/goat/data/new_checkpoints/iin/ver/resnetclip_rgb_croco_image/4_gpus/ckpt.28.pth"
                
            )
        )
        print("IIN missing keys: {}".format(missing_keys))
        print("Initialization of GOAT high level policy done.......")

        self.train()

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        return {
            k.replace("actor_critic.", ""): v
            for k, v in ckpt["state_dict"].items()
        }

    @property
    def output_size(self):
        return self.ovon_policy.net._hidden_size

    @property
    def is_blind(self):
        return self.ovon_policy.net.is_blind

    @property
    def num_recurrent_layers(self):
        return self.ovon_policy.net.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self.ovon_policy.net._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations["current_subtask"][0].item() == 0:
            if prev_actions[0].item() == 6:
                prev_actions[0] = HabitatSimActions.stop
                # print(
                #     "ovon relabel subtaskstop to stop: {}".format(prev_actions)
                # )
            (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            ) = self.ovon_policy.act(
                {
                    k: v
                    for k, v in observations.items()
                    if k
                    not in [
                        CacheImageGoalSensor.cls_uuid,
                        LanguageGoalSensor.cls_uuid,
                    ]
                },
                rnn_hidden_states,
                prev_actions,
                masks,
            )
            if action[0].item() == HabitatSimActions.stop:
                action[0] = 6
                # print("ovon relabel stop to subtask stop: {}".format(action))
        elif observations["current_subtask"][0].item() == 1:
            if prev_actions[0].item() == 6:
                prev_actions[0] = HabitatSimActions.stop
                print(
                    "lmav relabel subtaskstop to stop: {}".format(prev_actions)
                )
            (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            ) = self.language_policy.act(
                {
                    k: v
                    for k, v in observations.items()
                    if k
                    not in [
                        CacheImageGoalSensor.cls_uuid,
                        ClipObjectGoalSensor.cls_uuid,
                    ]
                },
                rnn_hidden_states,
                prev_actions,
                masks,
            )
            if action[0].item() == HabitatSimActions.stop:
                action[0] = 6
                # print("lnav relabel stop to subtask stop: {}".format(action))
        elif observations["current_subtask"][0].item() == 2:
            if prev_actions[0].item() == 6:
                prev_actions[0] = HabitatSimActions.stop
                print(
                    "image relabel subtaskstop to stop: {}".format(prev_actions)
                )
            (
                value,
                action,
                action_log_probs,
                rnn_hidden_states,
            ) = self.image_policy.act(
                {
                    k: v
                    for k, v in observations.items()
                    if k
                    not in [
                        LanguageGoalSensor.cls_uuid,
                        ClipObjectGoalSensor.cls_uuid,
                    ]
                },
                rnn_hidden_states,
                prev_actions,
                masks,
            )
            if action[0].item() == HabitatSimActions.stop:
                action[0] = 6
                # print("image relabel stop to subtask stop: {}".format(action))
        else:
            action = torch.tensor([0])
            value = torch.tensor([0])
            action_log_probs = torch.tensor([0])
            rnn_hidden_states = torch.zeros_like(rnn_hidden_states)

        return value, action, action_log_probs, rnn_hidden_states
