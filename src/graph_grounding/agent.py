from agentlab.agents.generic_agent.generic_agent_prompt import MainPrompt
from agentlab_ext.genericagent_ext import GenericAgentWithSleepAndExtractedMainPrompt
from graph_grounding.embeddings import Vectorizer
from graph_grounding.prompt.main_prompt import MainPromptWithGraph

from .constant import POCIterations
from .retriever import NavigationGraphGroundingRetrieverFactory
from .llm import LLM
from .navigation_graph import create_navigation_graph_client
from .state_abstraction import StateAbstractorFactory

from time import sleep

POC_ITERATION = POCIterations.FOUR

class GraphGroundingAgent(GenericAgentWithSleepAndExtractedMainPrompt):
    """
    Graph Grounding Agent
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.navigation_graph = create_navigation_graph_client()
        self.llm = LLM("gpt-4o")

        self.embeddings = Vectorizer(model_name="text-embedding-3-small")
        self.state_abstractor = StateAbstractorFactory(POC_ITERATION, self.llm).create()
        self.retriever = NavigationGraphGroundingRetrieverFactory(
            POC_ITERATION,
            self.navigation_graph,
            self.embeddings
        ).create()

    def _get_graph_grounding(self, current_observation: dict):
        """
        Get the graph grounding for the current observation
        """
        try:
            current_state_repr = self.state_abstractor.abstract_state(current_observation)
            return self.retriever.retrieve(current_state_repr, current_observation["goal"])
        except Exception as e:
            print(f"Error retrieving graph grounding: {e}")
            return None

    def get_main_prompt(self) -> MainPrompt:
        return MainPromptWithGraph(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

    def obs_preprocessor(self, obs):
        # sleep to avoid AI core rate limit
        print(f"Sleeping for 5 seconds to avoid AI core rate limit")
        sleep(5)

        obs = super().obs_preprocessor(obs)

        if not self.flags.obs.use_graph:
            return obs

        #######################################################
        # Inference Time: Query the navigation graph to get   #
        # possible next actions to take.                      #
        #######################################################
        obs["graph_grounding"] = self._get_graph_grounding(obs)
        return obs
