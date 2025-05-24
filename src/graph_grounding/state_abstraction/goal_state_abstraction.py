from . import StateAbstractor

class GoalBasedStateAbstractor(StateAbstractor):
    """
    Abstracts the state of the navigation graph to a more general form by using the goal of the task.
    This is used to create a more general representation of the state of the navigation graph.
    """

    def abstract_state(self, state: dict) -> str:
        """
        Abstracts the state of the navigation graph to a more general form by using the goal of the task.
        """
        return state["goal"] if "goal" in state else None
