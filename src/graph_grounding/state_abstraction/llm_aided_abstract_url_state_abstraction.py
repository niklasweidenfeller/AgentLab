from .abstract_url_state_abstraction import AbstractUrlStateAbstractor

from ..llm import (
    LLM,
    system_message,
    user_message,
    assistant_message,
    escape_url_system_prompt
)

class LLMAidedAbstractUrlStateAbstractor(AbstractUrlStateAbstractor):
    """
    Uses an LLM to abstract the state of the navigation graph to a more general form.
    This is used to create a more general representation of the state of the navigation graph.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def abstract_state(self, state: dict) -> str:
        """
        Abstracts the state of the navigation graph to a more general form by using an LLM.
        """
        url = super().abstract_state(state)

        escaped_url = self.llm.complete([
            system_message(escape_url_system_prompt),
            user_message("https://www.example.com/products/12345"),
            assistant_message("https://www.example.com/products/<product_id>"),

            user_message("https://www.example.com/products/5675"),
            assistant_message("https://www.example.com/products/<product_id>"),

            user_message("https://www.example.com/products/67890"),
            assistant_message("https://www.example.com/products/<product_id>"),

            user_message("https://www.example.com/users/12345/reviews/pages/1"),
            assistant_message("https://www.example.com/users/<user_id>/reviews/pages/<page_number>"),

            user_message("https://shopping.com/health-household.html"),
            assistant_message("https://shopping.com/<category>.html"),

            user_message("https://shopping.com/office-supplies.html"),
            assistant_message("https://shopping.com/<category>.html"),

            user_message("https://shopping.com/3-pack-samsung-galaxy-s6-screen-protector-scratch-resist.html"),
            assistant_message("https://shopping.com/<product_name>.html"),   

            user_message("https://social-forum.com/user/WaffleCactus42/submissions"),
            assistant_message("https://social-forum.com/user/<username>/submissions"),


            user_message(url),
        ])
        print(f"URL {url} escaped to {escaped_url}")
        return escaped_url
