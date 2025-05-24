from . import StateAbstractor
from ..urls import build_abstract_url, replace_urls

class AbstractUrlStateAbstractor(StateAbstractor):
    """
    Abstracts the state of the navigation graph to a more general form by removing the host from the URL.
    """

    def abstract_state(self, state: dict) -> str:
        """
        Abstracts the state of the navigation graph to a more general form by removing the host from the URL.
        """
        url = build_abstract_url(state["url"])
        url = replace_urls(url)

        return url
