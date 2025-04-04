from urllib.parse import urlparse
from neo4j import GraphDatabase

from .webpage_repository import WebpageRepository
from .database import AUTH

def create_navigation_graph_client():
    return NavigationGraph("bolt://localhost:7687", AUTH)

class NavigationGraph:
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.client = GraphDatabase.driver(uri, auth=auth)
        self.client.verify_connectivity()
        self.webpage_repository = WebpageRepository(self.client)
    
    def __del__(self):
        self.client.close()

    """
    RAG-based inference at runtime
    """
    def get_page_node_id(self, url: str) -> int:
        return self.webpage_repository.find_by_url(url).id

    def infer_actions_at_position(self, node_id: int, *, n_hops: int = 1):
        return self.query_longest_paths(node_id)

    def query_longest_paths(self, source_id: int):

        MAX_PATH_LENGTH = 10

        query = f'''
        MATCH (a:URL), (b:URL)
        WHERE elementID(a) = $source_id AND a <> b
        CALL apoc.algo.allSimplePaths(a, b, 'ACTION', {MAX_PATH_LENGTH})
        YIELD path

        LIMIT 200

        WITH COLLECT(path) as paths

        CALL navgraph.getPrimePaths(paths)
        YIELD primePath as path
        WITH COLLECT(path) as paths
        CALL navgraph.allNodesInPathHaveSameValueInNodePropertyArray(paths, 'flows')
        YIELD path

        WITH
            [node in nodes(path) | node.url] as nodesInPath,
            [rel in relationships(path) | rel.action + ": " + rel.target_line] as actionDescriptions, 
            length(path) as pathLength
        ORDER BY pathLength DESC
        WITH
            REDUCE(acc = [], i IN RANGE(0, pathLength - 2) |
                acc + [nodesInPath[i], actionDescriptions[i]]
            ) + [LAST(nodesInPath)] as nodeEdgeArray
        RETURN nodeEdgeArray
        '''

        print("******Querying known paths ...*")
        print(f"******Source ID: {source_id} ...*")
              
        records, _, _ = self.client.execute_query(
            query,
            source_id=source_id
        )

        paths = [r['nodeEdgeArray'] for r in records]
        merged = [ self.merge_path(path) for path in paths ]
        merged = [ path for path in merged if path ]
        print("******Number of paths found: ", len(merged))

        text = "\n\n".join([f"\t{i+1}. {path}" for i, path in enumerate(merged)]).strip()
        return text if text else None

    def merge_path(self, path):
        if len(path) == 1:
            print("Path length is 1")
            return None
        
        parts = path
        cleaned_parts = []

        current_url = None

        for part in parts:
            if not current_url:
                current_url = self.remove_host(part)
            else:
                cleaned_parts.append((current_url, part))
                current_url = None

        joined_tuples = [ f"({tup[0]}, {tup[1]})" for tup in cleaned_parts ]

        # append the end state
        if len(parts) % 2 != 0:
            joined_tuples.append(f"({self.remove_host(parts[-1])})")

        return " -> ".join(joined_tuples)

    def remove_host(self, url: str) -> str:
        parsed_url = urlparse(url)

        cleaned = parsed_url.path
        if parsed_url.query:
            cleaned += "?" + parsed_url.query
        if parsed_url.fragment:
            cleaned += "#" + parsed_url.fragment    
        return cleaned
