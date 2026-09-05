import warnings

# These imports exist for their side effects: each module registers a task,
# dataset, simulator, sensor, measurement, policy or trainer with habitat's
# registry, so they must run before any GOAT config is instantiated.
#
# They are split into two guarded groups so that `goat_bench.exploration`
# (frontier exploration for coverage tours) can run without the full GOAT
# stack. Exploration needs the simulator and the episode datasets -- to know
# which scenes and start poses make up a split -- but none of the
# goal-conditioned machinery: no CLIP/LAVIS/BLIP-2 goal encoders, no RL
# policies, and none of the habitat-lab 0.2.3-only action and gym adapters.
#
# Both guards catch only ImportError and warn loudly rather than passing
# silently, so a genuinely broken install stays visible. If a group fails, what
# it registers surfaces later as a habitat registry miss ("no registered task
# Goat-v1"); these warnings are the pointer to look here first.

# Group 1 -- core registration: datasets, the GOAT episode/task types and the
# simulator. Depends only on habitat itself, and imports cleanly on both
# habitat-lab 0.2.3 and 0.2.5.
try:
    from goat_bench import config
    from goat_bench.dataset import goat_dataset, languagenav_dataset, ovon_dataset
    from goat_bench.task import goat_task, rewards, simulator

    _GOAT_CORE_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - depends on the installed extras
    _GOAT_CORE_IMPORT_ERROR = e
    warnings.warn(
        f"goat_bench: core task/dataset registration skipped "
        f"({e.__class__.__name__}: {e}). The Goat-v1 episode datasets will not "
        f"be available.",
        ImportWarning,
        stacklevel=2,
    )

# Group 2 -- the goal-conditioned stack: goal sensors, observation transforms,
# measurements, policies and trainers, plus the action-space and gym adapters.
# Needs the CLIP/LAVIS/VC-1 model dependencies, and `task.actions` /
# `task.environments` additionally require habitat-lab 0.2.3 APIs
# (`HabitatSimV1ActionSpaceConfiguration`, `habitat.utils.gym_adapter`) that
# were removed in 0.2.5.
try:
    from goat_bench.measurements import collision_penalty, nav, sum_reward
    from goat_bench.models import (
        clip_policy,
        high_level_policy,
        objaverse_clip_policy,
        ovrl_policy,
    )
    from goat_bench.obs_transformer import resize
    from goat_bench.task import actions, environments, sensors
    from goat_bench.trainers import ppo_trainer_no_2d

    _GOAT_RL_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - depends on the installed extras
    _GOAT_RL_IMPORT_ERROR = e
    warnings.warn(
        f"goat_bench: goal-sensor/policy registration skipped "
        f"({e.__class__.__name__}: {e}). GOAT training and evaluation will not "
        f"work; goat_bench.exploration is unaffected. Install the full "
        f"requirements on habitat-lab 0.2.3 to restore it.",
        ImportWarning,
        stacklevel=2,
    )
