from neo4j import GraphDatabase


from .database import AUTH
from .embeddings import Vectorizer
from .webpage_repository import WebpageRepository

def create_navigation_graph_client():
    return NavigationGraph("bolt://localhost:7687", AUTH)

class NavigationGraph:
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.client = GraphDatabase.driver(uri, auth=auth)
        self.client.verify_connectivity()
        self.webpage_repository = WebpageRepository(self.client)
        self.embeddings = Vectorizer(model_name="text-embedding-3-small")

    def __del__(self):
        self.client.close()

    def execute_query(self, query: str, **params):
        """
        Execute a Cypher query with the given parameters.
        """
        records, _, _ = self.client.execute_query(
            query,
            **params
        )
        return records

    def run_statement(self, statement: str, **params):
        """
        Run a Cypher statement with the given parameters.
        """
        with self.client.session() as session:
            result = session.run(statement, **params)
            return result.data()

    def get_page_node_id(self, url: str) -> int:
        return self.webpage_repository.find_by_url(url).id
