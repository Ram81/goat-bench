import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)


@registry.register_action_space_configuration(name="v2-goat")
class HabitatSimV2ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def get(self):
        config = super().get()
        HabitatSimActions._known_actions["subtask_stop"] = 6
        new_config = {
            HabitatSimActions.subtask_stop: habitat_sim.ActionSpec(
                "subtask_stop"
            ),
        }

        config.update(new_config)

        return config
