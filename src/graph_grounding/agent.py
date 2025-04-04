from agentlab.agents.generic_agent.generic_agent import GenericAgent
from graph_grounding.prompt.main_prompt import MainPromptWithGraph

from .navigation_graph import create_navigation_graph_client
from .urls import build_abstract_url, replace_urls


class GraphGroundingAgent(GenericAgent):
    """
    Graph Grounding Agent
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigation_graph = create_navigation_graph_client()

    def obs_preprocessor(self, obs):
        obs = super().obs_preprocessor(obs)

        if not self.flags.obs.use_graph:
            return obs

        #######################################################
        # Inference Time: Query the navigation graph to get   #
        # possible next actions to take.                      #
        #######################################################
        url = build_abstract_url(obs["url"])
        url = replace_urls(url)

        try:
            current_node_id = self.navigation_graph.get_page_node_id(url)
            obs["graph_grounding"] = self.navigation_graph.infer_actions_at_position(current_node_id, n_hops=2) if current_node_id else None
        except ValueError:
            print(f"Could not find page node for url: {url}")
            obs["graph_grounding"] = None
        return obs

    def get_main_prompt(self):
        return MainPromptWithGraph(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags
        )
