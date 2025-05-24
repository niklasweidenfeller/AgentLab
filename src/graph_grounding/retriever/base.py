
from abc import ABC, abstractmethod

from ..navigation_graph import NavigationGraph
from ..embeddings import Vectorizer


class NavigationGraphGroundingRetriever(ABC):
    """
    Abstract class for navigation graph grounding retrievers.
    This class defines the interface for retrieving navigation graphs based on a query.
    """

    def __init__(self, navigation_graph_client: NavigationGraph, embedding_client: Vectorizer):
        """
        Initialize the retriever with a navigation graph client.

        :param navigation_graph_client: An instance of NavigationGraph to interact with the navigation graphs.
        """
        self.navigation_graph_client = navigation_graph_client
        self.embedding_client = embedding_client

    def retrieve(self, state, goal) -> str:
        """
        Retrieve navigation graphs based on the current state and return them as a string.

        :param state: The current state or query string to search for in the navigation graphs.
        :return: A string representation of the navigation graphs that match the query.
        """
        graphs = self._retrieve(state, goal)
        return self._verbalize(graphs)

    @abstractmethod
    def _retrieve(self, state: str, goal: str) -> list:
        """
        Retrieve a list of navigation graphs based on the current state.

        :param state: The current state or query string to search for in the navigation graphs.
        :return: A list of navigation graphs that match the query.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def _verbalize(self, graphs: list) -> str:
        """
        Convert the retrieved navigation graphs into a human-readable string format.

        :param graphs: A list of navigation graphs to be verbalized.
        :return: A string representation of the navigation graphs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
