from __future__ import annotations

from graph_grounding.state_abstraction.base import StateAbstractor

from .abstract_url_state_abstraction import AbstractUrlStateAbstractor
from .llm_aided_abstract_url_state_abstraction import LLMAidedAbstractUrlStateAbstractor
from .goal_state_abstraction import GoalBasedStateAbstractor
from ..constant import POCIterations

class StateAbstractorFactory:
    """
    Factory class to create state abstractors based on the POC iteration.
    This class is used to create the appropriate state abstractor based on the POC iteration.
    """

    def __init__(self, iteration: POCIterations, llm=None):
        """
        Initialize the factory with the POC iteration and an optional LLM.
        :param iteration: The POC iteration to determine which state abstractor to create.
        :param llm: An optional LLM instance for iterations that require it.
        """
        self.iteration = iteration
        self.llm = llm

    def create(self) -> StateAbstractor:
        """
        Create the state abstractor instance based on the POC iteration.
        
        :return: An instance of the appropriate StateAbstractor subclass.
        """
        
        if self.iteration == POCIterations.ONE:
            return AbstractUrlStateAbstractor()
        elif self.iteration == POCIterations.TWO:
            if self.llm is None:
                raise ValueError("LLM must be provided for iteration 2")
            return LLMAidedAbstractUrlStateAbstractor(self.llm)
        elif self.iteration == POCIterations.THREE:
            return GoalBasedStateAbstractor()
        elif self.iteration == POCIterations.FOUR:
            return GoalBasedStateAbstractor()
        raise ValueError(f"Unknown POC iteration: {self.iteration}")
