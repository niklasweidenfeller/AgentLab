
from agentlab.agents.dynamic_prompting import PromptElement


class Graph(PromptElement):
    def __init__(self, obs, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible)
        self.obs = obs
        self.prefix = prefix

    @property
    def _prompt(self) -> str:
        grounding = self.obs.get("graph_grounding", None)
        
        if not grounding:
            return ""

        return f"""
From past interactions, we are aware of the following actions and targets, that are
reachable from your current location. You can use this information to guide your actions,
meaning that you can use these known paths from our NAVIGATION GRAPH to help you
decide what to do next.

Please explicitly state what you have learned from the graph, how it relates to the known
paths and how it influenced your next action suggestion.

The graph is presented you as a list of paths, where each path is described as

(state, action) -> (state, action) -> ... -> (state, action)

Try to understand which process the paths represent and if any of them can be used
to solve the current task.

Following paths are known to us:

    {grounding}
        """
