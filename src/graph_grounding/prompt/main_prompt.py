
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from browsergym.core.action.base import AbstractActionSet

from graph_grounding.prompt.observation import ObservationWithGraph

class MainPromptWithGraph(MainPrompt):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: GenericPromptFlags,
    ) -> None:
        super().__init__(
            action_set,
            obs_history,
            actions,
            memories,
            thoughts,
            previous_plan,
            step,
            flags
        )
        self.obs = ObservationWithGraph(
            obs_history[-1],
            self.flags.obs,
        )
