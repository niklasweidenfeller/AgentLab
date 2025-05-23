from urllib.parse import urlparse
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

    def find_by_url_and_task_iteration_2(self, abstract_url: str, task: str):
        """
        Find the graph grounding for a given URL and task
        """

        query = '''
        MATCH (a:URL { url: $url })-[]-(s:STEP)-[]-(t:GOAL)
        WITH a, t, vector.similarity.cosine(t.embedding, $embedding) AS score, MIN(s.step_id) AS start_step_id
        WHERE score > 0.6
        MATCH p=(t)-[]-(s:STEP { step_id: start_step_id })
        WITH s
        MATCH path=(s)-[:ACTION*1..]->(t:STEP)

        WITH COLLECT(path) as paths

        CALL navgraph.getPrimePaths(paths)
        YIELD primePath as path
        RETURN nodes(path) as nodes, relationships(path) as rels
        '''
        # MATCH p=(s)-[:ACTION*1..]-(t:STEP)
        # RETURN p

        records, _, _ = self.client.execute_query(
            query,
            url=abstract_url,
            embedding=self.embeddings(task),
        )

        if len(records) == 0:
            
            print("******Number of result records: ", len(records))
            print("Finding Iteration 2 graph grounding for URL: ", abstract_url, " and task: ", task)

            
            raise ValueError(f'No results found for URL: {abstract_url} and task: {task}')
        return records

    def find_by_task_iteration3(self, task_description: str):
        stmt = """
        MATCH (t:Task) WITH t.goal as goal
        WITH goal, apoc.text.distance(goal, $task) as hammingDistance
        ORDER BY hammingDistance ASC
        LIMIT 3

        WITH goal, hammingDistance

        MATCH path=(t:Task { goal: goal })-[*..25]->(n)

        WITH COLLECT(path) as paths, goal, hammingDistance

        UNWIND paths as p1
        WITH goal, hammingDistance, p1, [x IN paths WHERE x <> p1 AND all(r IN relationships(p1) WHERE r IN relationships(x))] AS supersets
        WHERE size(supersets) = 0

        RETURN goal, p1 as path, hammingDistance
        ORDER BY hammingDistance ASC
        """
        with self.client.session() as session:
            result = session.run(stmt, task=task_description)
            result = result.data()


        def stringify_path(path):
            result = []
            ctr = 1
            
            for i, item in enumerate(path):
                if isinstance(item, str):
                    result.append(item)
                    continue

                if i == 0:
                    goal_text = item.get("description", None)
                    result.append(f"Goal: {goal_text}")
                else:
                    node_text = item.get("description", None)
                    result.append(f"{ctr}. {node_text}")
                    ctr += 1

            return "\n".join(result)


        result = [ r['path'] for r in result ]
        return [stringify_path(path) for path in result]

