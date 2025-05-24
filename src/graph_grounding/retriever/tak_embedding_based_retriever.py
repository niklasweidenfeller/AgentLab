from abc import abstractmethod
from .base import NavigationGraphGroundingRetriever

class TaskEmbeddingBasedRetriever(NavigationGraphGroundingRetriever):

    @abstractmethod
    def _retrieve(self, state: str, goal: str) -> list:
        stmt = """
        MATCH (n)
        WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
        WHERE score > 0.75

        MATCH path=(n)-[*..25]->(m)

        WITH COLLECT(path) as paths, n.description as goal, score

        UNWIND paths as p1
        WITH goal, score, p1, [x IN paths WHERE x <> p1 AND all(r IN relationships(p1) WHERE r IN relationships(x))] AS supersets
        WHERE size(supersets) = 0

        WITH goal, p1 as path, relationships(p1) as rels, score
        UNWIND rels as rel
        WITH goal, path, collect(rel) as rels, collect(properties(rel)) as properties, score
        RETURN goal, path, rels, properties, score
        ORDER BY score ASC
        LIMIT 3
        """

        print(f"******Task description: {goal} ******")
        embedding = self.embedding_client(goal)
        result = self.navigation_graph_client.run_statement(
            stmt,
            embedding=embedding
        )

        print(f"******Number of result records: {len(result)} ******")
        return [(r['path'], r['rels'], r['properties'], r['score']) for r in result]

    @abstractmethod
    def _verbalize(self, graphs: list) -> str:
        """
        Convert the retrieved navigation graphs into a human-readable string format.

        :param graphs: A list of navigation graphs to be verbalized.
        :return: A string representation of the navigation graphs.
        """
        stringified_items = [
            self.stringify_path_i4(path, score) for path, _rels, _properties, score in graphs
        ]
        return "\n\n".join(stringified_items) if stringified_items else None

    def stringify_path_i4(self, path, similarity_score):
        result = []
        ctr = 1

        # rels_lookup = {}
        # for rel in rels:
        #     rels_lookup[rel['start']] = rel['properties']
        
        for i, item in enumerate(path):
            if isinstance(item, str):
                result.append(item)
                continue

            if i == 0:
                goal_text = item.get("description", None)
                result.append(f"Goal: {goal_text} (Similarity: {similarity_score})")
            else:
                node_text = item.get("description", None)
                result.append(f"{ctr}. {node_text}")
                ctr += 1

        return "\n".join(result)

