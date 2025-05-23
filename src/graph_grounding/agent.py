from enum import Enum
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from graph_grounding.prompt.main_prompt import MainPromptWithGraph

from .llm import LLM, system_message, user_message, assistant_message, escape_url_system_prompt
from .navigation_graph import create_navigation_graph_client
from .urls import build_abstract_url, replace_urls

from time import sleep

class POCIterations(Enum):
    """
    Number of iterations for the POC
    """
    ONE = "standard"
    TWO = "advanced"
    THREE = "llm-generated-graph"

POC_ITERATION = POCIterations.THREE

class GraphGroundingAgent(GenericAgent):
    """
    Graph Grounding Agent
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigation_graph = create_navigation_graph_client()
        self.llm = LLM("gpt-4o")

    def _get_graph_grounding(self, current_observation: dict, poc_iteration: POCIterations):
        """
        Get the graph grounding for the current observation
        """
        if poc_iteration == POCIterations.ONE:
            url = build_abstract_url(current_observation["url"])
            url = replace_urls(url)
            current_node_id = self.navigation_graph.get_page_node_id(url)
            return self.navigation_graph.infer_actions_at_position(current_node_id, n_hops=2) if current_node_id else None

        elif poc_iteration == POCIterations.TWO:
            url = build_abstract_url(current_observation["url"])
            url = replace_urls(url)
            
            escaped_url = self.llm.complete([
                system_message(escape_url_system_prompt),
                user_message("https://www.example.com/products/12345"),
                assistant_message("https://www.example.com/products/<product_id>"),

                user_message("https://www.example.com/products/5675"),
                assistant_message("https://www.example.com/products/<product_id>"),

                user_message("https://www.example.com/products/67890"),
                assistant_message("https://www.example.com/products/<product_id>"),

                user_message("https://www.example.com/users/12345/reviews/pages/1"),
                assistant_message("https://www.example.com/users/<user_id>/reviews/pages/<page_number>"),

                user_message("https://shopping.com/health-household.html"),
                assistant_message("https://shopping.com/<category>.html"),

                user_message("https://shopping.com/office-supplies.html"),
                assistant_message("https://shopping.com/<category>.html"),

                user_message("https://shopping.com/3-pack-samsung-galaxy-s6-screen-protector-scratch-resist.html"),
                assistant_message("https://shopping.com/<product_name>.html"),   

                user_message("https://social-forum.com/user/WaffleCactus42/submissions"),
                assistant_message("https://social-forum.com/user/<username>/submissions"),


                user_message(url),
            ])
            print(f"URL {url} escaped to {escaped_url}")
            url = escaped_url
            
            grounding = self.navigation_graph.find_by_url_and_task_iteration_2(
                url, current_observation["goal"]
            )
            
            stringify_path = lambda path: " -> ".join([f"({action['action']}, {action['input']}, {action['target_line']})" for (_, action) in path])

            paths = [zip(g['nodes'], g['rels']) for g in grounding]
            stringified_paths = [stringify_path(path) for path in paths]
            return "\n".join(stringified_paths)
  
        elif poc_iteration == POCIterations.THREE:
            task_description = current_observation["goal"]
            paths = self.navigation_graph.find_by_task_iteration3(task_description)
            return "\n\n".join(paths) if paths and len(paths) > 0 else None
        else:
            raise ValueError(f"Invalid POC iteration: {poc_iteration}")

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

        try:
            obs["graph_grounding"] = self._get_graph_grounding(obs, POC_ITERATION)
        except ValueError:
            print(f"Could not retrieve graph grounding for {obs['url']}")
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
