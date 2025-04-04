from agentlab.agents.dynamic_prompting import ObsFlags, Observation

from .prompt_addition_graph import Graph

class ObservationWithGraph(Observation):
    def __init__(self, obs, flags: ObsFlags) -> None:
        super().__init__(obs, flags)
        self.graph_grounding = obs.get("graph_grounding", None)

        self.graph = Graph(
            obs,
            visible=lambda: flags.use_graph,
            prefix="## ",
        )

    @property
    def _prompt(self) -> str:
        super_observation = super()._prompt
        return super_observation + self.graph._prompt + "\n\n"
