from abc import abstractmethod
from .base import NavigationGraphGroundingRetriever

class HammingDistanceBasedRetriever(NavigationGraphGroundingRetriever):

    @abstractmethod
    def _retrieve(self, state: str, goal: str) -> list:
        """
        Retrieve a list of navigation graphs based on the current state.

        :param state: The current state or query string to search for in the navigation graphs.
        :return: A list of navigation graphs that match the query.
        """
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

        result = self.navigation_graph_client.run_statement(stmt, task=goal)
        return [ r['path'] for r in result ]

    @abstractmethod
    def _verbalize(self, graphs: list) -> str:
        """
        Convert the retrieved navigation graphs into a human-readable string format.

        :param graphs: A list of navigation graphs to be verbalized.
        :return: A string representation of the navigation graphs.
        """
        verbalized_paths = [stringify_path(path) for path in graphs]
        return "\n".join(verbalized_paths)


def stringify_path(path):
    result = []
    ctr = 1
    
    for i, item in enumerate(path):
        if isinstance(item, str):
            result.append(item)
            continue

        if not isinstance(item, dict):
            continue

        if i == 0:
            goal_text = item.get("description", None)
            result.append(f"Goal: {goal_text}")
        else:
            node_text = item.get("description", None)
            result.append(f"{ctr}. {node_text}")
            ctr += 1

    return "\n".join(result)


