
from dataclasses import dataclass
from neo4j import Driver


@dataclass
class Webpage:
    """ A class to represent a webpage. """
    id: int | None
    abstract_url: str

class WebpageRepository:
    def __init__(self, client: Driver):
        self.client = client

    def find_by_url(self, abstract_url: str) -> Webpage:
        records, _, _ = self.client.execute_query(
            "MATCH (a:URL) WHERE a.url = $url RETURN elementID(a), a.url",
            url=abstract_url
        )
        if len(records) == 0:
            raise ValueError(f'No webpage found with abstract URL: {abstract_url}')
        return Webpage(id=records[0][0], abstract_url=records[0][1])
