import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gym import spaces
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import NavigationEpisode

from goat_bench.task.goat_task import GoatEpisode

cv2 = try_cv2_import()


from goat_bench.utils.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        dummy_embedding = np.zeros((1024,), dtype=np.float32)
        try:
            if isinstance(episode, GoatEpisode):
                # print(
                #     "GoatEpisode: {} - {}".format(
                #         episode.tasks[task.active_subtask_idx],
                #         isinstance(episode, GoatEpisode),
                #     )
                # )
                if task.active_subtask_idx < len(episode.tasks):
                    if episode.tasks[task.active_subtask_idx][1] == "object":
                        category = episode.tasks[task.active_subtask_idx][0]
                    else:
                        return dummy_embedding
                else:
                    return dummy_embedding
            else:
                category = (
                    episode.object_category
                    if hasattr(episode, "object_category")
                    else ""
                )
            if category not in self.cache:
                print("ObjectGoal Missing category: {}".format(category))
            # print("ObjectGoal Found category: {}".format(category))
        except Exception as e:
            print("Object goal exception ", e)
        return self.cache[category]


@registry.register_sensor
class ClipImageGoalSensor(Sensor):
    cls_uuid: str = "clip_imagegoal"

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )
        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        super().__init__(config=config)
        self._curr_ep_id = None
        self.image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        sampled_goal = random.choice(episode.goals)
        sampled_viewpoint = random.choice(sampled_goal.view_points)
        observations = self._sim.get_observations_at(
            position=sampled_viewpoint.agent_state.position,
            rotation=sampled_viewpoint.agent_state.rotation,
            keep_agent_at_new_pose=False,
        )
        assert observations is not None
        self.image_goal = observations["rgb"]
        # Mutate the episode
        episode.goals = [sampled_goal]

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.image_goal is None or self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        assert self.image_goal is not None
        return self.image_goal


@registry.register_sensor
class ClipGoalSelectorSensor(Sensor):
    cls_uuid: str = "clip_goal_selector"

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        self._image_sampling_prob = config.image_sampling_probability
        self._curr_ep_id = None
        self._use_image_goal = True

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.bool,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        self._use_image_goal = random.random() < self._image_sampling_prob

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        return np.array([self._use_image_goal], dtype=np.bool)


@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "image_goal_rotation"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode
        if self.config.sample_angle:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class CurrentEpisodeUUIDSensor(Sensor):
    r"""Sensor for current episode uuid observations.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "current_episode_uuid"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id: Optional[str] = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        episode_uuid = (
            int(hashlib.sha1(episode_uniq_id.encode("utf-8")).hexdigest(), 16)
            % 10**8
        )
        return episode_uuid


@registry.register_sensor
class LanguageGoalSensor(Sensor):
    r"""A sensor for language goal specification as observations which is used in
    Language Navigation.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "language_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        self.embedding_dim = config.embedding_dim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedding_dim,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        uuid = ""

        try:
            dummy_embedding = np.zeros((self.embedding_dim,), dtype=np.float32)
            if isinstance(episode, GoatEpisode):
                # print(
                #     "Lang GoatEpisode: {} - {}".format(
                #         episode.tasks[task.active_subtask_idx],
                #         isinstance(episode, GoatEpisode),
                #     )
                # )
                if task.active_subtask_idx < len(episode.tasks):
                    if (
                        episode.tasks[task.active_subtask_idx][1]
                        == "description"
                    ):
                        # print("not retur lang")
                        instance_id = episode.tasks[task.active_subtask_idx][2]
                        # print("instance id", instance_id)
                        # print(
                        #     "episode goals",
                        #     [
                        #         list(g.keys())
                        #         for g in episode.goals[task.active_subtask_idx]
                        #     ],
                        # )
                        goal = [
                            g
                            for g in episode.goals[task.active_subtask_idx]
                            if g["object_id"] == instance_id
                        ]
                        uuid = goal[0]["lang_desc"].lower()
                    else:
                        return dummy_embedding
                else:
                    return dummy_embedding
            else:
                uuid = episode.instructions[0].lower()
                first_3_words = [
                    "prefix: instruction: go",
                    "instruction: find the",
                    "instruction: go to",
                    "api_failure",
                    "instruction: locate the",
                ]
                for prefix in first_3_words:
                    uuid = uuid.replace(prefix, "")
                    uuid = uuid.replace("\n", " ")
                uuid = uuid.strip()

            if self.cache.get(uuid) is None:
                print("Lang Missing category: {}".format(uuid))
        except Exception as e:
            print("Language goal exception ", e)
        return self.cache[uuid]


@registry.register_sensor
class CacheImageGoalSensor(Sensor):
    r"""A sensor for Image goal specification as observations which is used in IIN.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "cache_instance_imagegoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache_base_dir = config.cache
        self.image_encoder = config.image_cache_encoder
        self.cache = None
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = np.zeros((1024,), dtype=np.float32)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"
        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]

            suffix = "embedding.pkl"
            if self.image_encoder != "":
                suffix = "{}_iin_{}".format(self.image_encoder, suffix)
            if isinstance(episode, GoatEpisode):
                suffix = suffix.replace("iin", "goat")

            print(
                "Cache dir: {}".format(
                    os.path.join(self.cache_base_dir, f"{scene_id}_{suffix}")
                )
            )
            self.cache = load_pickle(
                os.path.join(self.cache_base_dir, f"{scene_id}_{suffix}")
            )

        try:
            if self._current_episode_id != episode_id:
                self._current_episode_id = episode_id

                dummy_embedding = np.zeros((1024,), dtype=np.float32)
                if isinstance(episode, GoatEpisode):
                    if task.active_subtask_idx < len(episode.tasks):
                        if episode.tasks[task.active_subtask_idx][1] == "image":
                            instance_id = episode.tasks[
                                task.active_subtask_idx
                            ][2]
                            curent_task = episode.tasks[task.active_subtask_idx]
                            scene_id = episode.scene_id.split("/")[-1].split(
                                "."
                            )[0]

                            uuid = "{}_{}".format(scene_id, instance_id)

                            self._current_episode_image_goal = self.cache[
                                "{}_{}".format(scene_id, instance_id)
                            ][curent_task[-1]]["embedding"]
                        else:
                            self._current_episode_image_goal = dummy_embedding
                    else:
                        self._current_episode_image_goal = dummy_embedding
                else:
                    self._current_episode_image_goal = self.cache[
                        episode.goal_key
                    ][episode.goal_image_id]["embedding"]
        except Exception as e:
            print("Image goal exception ", e)
            raise e

        return self._current_episode_image_goal


@registry.register_sensor
class GoatCurrentSubtaskSensor(Sensor):
    r"""A sensor for Image goal specification as observations which is used in IIN.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "current_subtask"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.sub_task_type = config.sub_task_type
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0, high=len(self.sub_task_type) + 1, shape=(1,), dtype=np.int32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        current_subtask = task.active_subtask_idx
        current_subtask_id = len(self.sub_task_type)
        if current_subtask < len(episode.tasks):
            current_subtask_id = self.sub_task_type.index(
                episode.tasks[current_subtask][1]
            )

        return current_subtask_id


@registry.register_sensor
class GoatGoalSensor(Sensor):
    r"""A sensor for Goat goals"""
    cls_uuid: str = "goat_subtask_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        self.language_cache = load_pickle(config.language_cache)
        self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"

        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            self.image_cache = load_pickle(
                os.path.join(
                    self.image_cache_base_dir,
                    f"{scene_id}_{self.image_encoder}_embedding.pkl",
                )
            )

        output_embedding = np.zeros((1024,), dtype=np.float32)

        task_type = "none"
        if task.active_subtask_idx < len(episode.tasks):
            if episode.tasks[task.active_subtask_idx][1] == "object":
                category = episode.tasks[task.active_subtask_idx][0]
                output_embedding = self.object_cache[category]
                task_type = "object"
            elif episode.tasks[task.active_subtask_idx][1] == "description":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"].lower()
                output_embedding = self.language_cache[uuid]
                task_type = "lang"
            elif episode.tasks[task.active_subtask_idx][1] == "image":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                curent_task = episode.tasks[task.active_subtask_idx]
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]

                uuid = "{}_{}".format(scene_id, instance_id)

                output_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][curent_task[-1]]["embedding"]
                task_type = "image"
            else:
                raise NotImplementedError
        return output_embedding


@registry.register_sensor
class GoatMultiGoalSensor(Sensor):
    r"""A sensor for Goat goals"""
    cls_uuid: str = "goat_subtask_multi_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        self.language_cache = load_pickle(config.language_cache)
        self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024 * 3,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"

        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            self.image_cache = load_pickle(
                os.path.join(
                    self.image_cache_base_dir,
                    f"{scene_id}_{self.image_encoder}_embedding.pkl",
                )
            )

        output_embedding = np.zeros((1024 * 3,), dtype=np.float32)
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]

        task_type = "none"
        if task.active_subtask_idx < len(episode.tasks):
            if episode.tasks[task.active_subtask_idx][1] == "object":
                category = episode.tasks[task.active_subtask_idx][0]
                obj_embedding = self.object_cache[category]
                output_embedding = np.concatenate(
                    (obj_embedding, obj_embedding, obj_embedding)
                )
                task_type = "object"
            elif episode.tasks[task.active_subtask_idx][1] == "description":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"].lower()
                lang_embedding = self.language_cache[uuid]

                uuid = "{}_{}".format(scene_id, instance_id)
                random_idx = random.choice(
                    range(
                        len(
                            self.image_cache[
                                "{}_{}".format(scene_id, instance_id)
                            ]
                        )
                    ),
                )

                img_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][random_idx]["embedding"]

                category = episode.tasks[task.active_subtask_idx][0]
                cat_embedding = self.object_cache[category]

                output_embedding = np.concatenate(
                    (lang_embedding, img_embedding, cat_embedding)
                )
                task_type = "lang"
            elif episode.tasks[task.active_subtask_idx][1] == "image":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                curent_task = episode.tasks[task.active_subtask_idx]
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]

                uuid = "{}_{}".format(scene_id, instance_id)

                img_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][curent_task[-1]]["embedding"]

                category = episode.tasks[task.active_subtask_idx][0]
                cat_embedding = self.object_cache[category]

                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"]
                if uuid is not None:
                    uuid = uuid.lower()
                    lang_embedding = self.language_cache[uuid]
                else:
                    lang_embedding = cat_embedding

                output_embedding = np.concatenate(
                    (lang_embedding, img_embedding, cat_embedding)
                )

                task_type = "image"
            else:
                raise NotImplementedError
        return output_embedding
