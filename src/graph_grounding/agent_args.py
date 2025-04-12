from dataclasses import dataclass
from agentlab.agents.dynamic_prompting import ObsFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from graph_grounding.agent import GraphGroundingAgent


@dataclass
class GraphGroundingObsFlags(ObsFlags):
    """
    Extends ObsFlags to include graph grounding.

    Attributes:
        use_graph (bool): Add the navigation graph to the prompt.
    """
    use_graph: bool = False

    def __init__(self, use_graph: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_graph = use_graph

@dataclass
class GraphGroundingPromptFlags(GenericPromptFlags):
    """
    Extends GenericPromptFlags to include graph grounding.

    Attributes:
        obs (GraphGroundingObsFlags): Observation flags.
    """
    obs: GraphGroundingObsFlags

class GraphGroundingAgentArgs(GenericAgentArgs):
    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GraphGroundingAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return GraphGroundingAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )
