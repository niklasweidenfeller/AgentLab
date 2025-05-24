from .base import NavigationGraphGroundingRetriever


class UrlTaskBasedRetriever(NavigationGraphGroundingRetriever):
    def _retrieve(self, state, goal) -> list:
        url = state
        grounding = self._query_graph(url, goal)
        return grounding
    
    def _verbalize(self, graphs):
        """
        Convert the retrieved navigation graphs into a human-readable string format.
        
        :param graphs: A list of navigation graphs to be verbalized.
        :return: A string representation of the navigation graphs.
        """

        stringify_path = lambda path: " -> ".join([f"({action['action']}, {action['input']}, {action['target_line']})" for (_, action) in path])

        paths = [zip(g['nodes'], g['rels']) for g in graphs]
        stringified_paths = [stringify_path(path) for path in paths]
        return "\n".join(stringified_paths)

    def _query_graph(self, abstract_url: str, task: str):
        """
        Query the navigation graph for a specific URL and task.
        
        :param abstract_url: The URL to query in the navigation graph.
        :param task: The task description to filter the results.
        :return: A list of records containing the nodes and relationships of the path.
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

        records = self.navigation_graph_client.execute_query(
            query,
            url=abstract_url,
            embedding=self.embedding_client(task),
        )

        if len(records) == 0:
            raise ValueError(f'No results found for URL: {abstract_url} and task: {task}')

        return records
