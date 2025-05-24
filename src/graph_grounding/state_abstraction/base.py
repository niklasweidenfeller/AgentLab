
from abc import ABC, abstractmethod


class StateAbstractor(ABC):
    """
    Abstracts the state of the navigation graph to a more general form.
    This is used to create a more general representation of the state of the navigation graph.
    """

    @abstractmethod
    def abstract_state(self, state: dict) -> str:
        """
        Abstracts the state of the navigation graph to a more general form.
        """
        pass
