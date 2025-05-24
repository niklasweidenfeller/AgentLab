from urllib.parse import urlparse

from . import NavigationGraphGroundingRetriever

class LongestPathBasedRetriever(NavigationGraphGroundingRetriever):
    """
    A retriever that finds the longest path in a navigation graph based on a given query.
    This class extends the AbstractNavigationGraphGroundingRetriever to implement the retrieval logic.
    """

    def _retrieve(self, state: str, goal: str) -> list:
        """
        Retrieve the longest path in the navigation graph based on the current state.

        :param state: The current state or query string to search for in the navigation graphs.
        :return: A list containing the longest path found in the navigation graph.
        """
        node_id = self.navigation_graph_client.get_page_node_id(state)
        paths = self._infer_actions_at_position(node_id)
        return paths

    def _verbalize(self, graphs: list) -> str:
        """
        Convert the retrieved longest path into a human-readable string format.

        :param graphs: A list containing the longest path to be verbalized.
        :return: A string representation of the longest path.
        """

        merged = [ self._merge_path(path) for path in graphs ]
        merged = [ path for path in merged if path ]
        print("******Number of paths found: ", len(merged))

        text = "\n\n".join([f"\t{i+1}. {path}" for i, path in enumerate(merged)]).strip()
        return text if text else None

    def _infer_actions_at_position(self, node_id: int) -> list:
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
        print(f"******Source ID: {node_id} ...*")
              
        records = self.navigation_graph_client.execute_query(
            query,
            source_id=node_id
        )

        paths = [r['nodeEdgeArray'] for r in records]
        return paths

    def _merge_path(self, path):
        if len(path) == 1:
            print("Path length is 1")
            return None
        
        parts = path
        cleaned_parts = []

        current_url = None

        for part in parts:
            if not current_url:
                current_url = self._remove_host(part)
            else:
                cleaned_parts.append((current_url, part))
                current_url = None

        joined_tuples = [ f"({tup[0]}, {tup[1]})" for tup in cleaned_parts ]

        # append the end state
        if len(parts) % 2 != 0:
            joined_tuples.append(f"({self._remove_host(parts[-1])})")

        return " -> ".join(joined_tuples)

    def _remove_host(self, url: str) -> str:
        parsed_url = urlparse(url)

        cleaned = parsed_url.path
        if parsed_url.query:
            cleaned += "?" + parsed_url.query
        if parsed_url.fragment:
            cleaned += "#" + parsed_url.fragment    
        return cleaned
