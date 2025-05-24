from graph_grounding.retriever.url_task_based_retriever import UrlTaskBasedRetriever
from ..embeddings import Vectorizer
from ..constant import POCIterations
from ..navigation_graph import NavigationGraph
from .base import NavigationGraphGroundingRetriever
from .longest_path_based_retriever import LongestPathBasedRetriever
from .hamming_distance_based_retriever import HammingDistanceBasedRetriever
from .tak_embedding_based_retriever import TaskEmbeddingBasedRetriever

class NavigationGraphGroundingRetrieverFactory:
    def __init__(self, iteration: POCIterations, navigation_graph_client: NavigationGraph, embedding_client: Vectorizer):
        """
        Initialize the factory with the POC iteration and a navigation graph client.

        :param iteration: The POC iteration to determine which retriever to create.
        :param navigation_graph_client: An instance of NavigationGraph to interact with the navigation graphs.
        """
        self.iteration = iteration
        self.navigation_graph_client = navigation_graph_client
        self.embedding_client = embedding_client

    def create(self) -> NavigationGraphGroundingRetriever:
        """
        Create the navigation graph grounding retriever instance based on the POC iteration.

        :return: An instance of the appropriate NavigationGraphGroundingRetriever subclass.
        """
        if self.iteration == POCIterations.ONE:
            return LongestPathBasedRetriever(self.navigation_graph_client)
        elif self.iteration == POCIterations.TWO:
            return UrlTaskBasedRetriever(self.navigation_graph_client, self.navigation_graph_client.embeddings)
        elif self.iteration == POCIterations.THREE:
            return HammingDistanceBasedRetriever(self.navigation_graph_client)
        elif self.iteration == POCIterations.FOUR:
            return TaskEmbeddingBasedRetriever(self.navigation_graph_client, self.embedding_client)

        raise ValueError(f"Unsupported POC iteration: {self.iteration}. Supported iterations are: {list(POCIterations)}")
