from gen_ai_hub.proxy.native.openai import OpenAI
from gen_ai_hub.proxy.core import get_proxy_client

escape_url_system_prompt = """
given the following URL, please provide a template for the URL.
A template is a string with placeholders for the parts of the URL that are variable.
Please return as a comma-separated list of templates, do not include bullets or any other formatting.
You must include exactly the same number of templates as the number of URLs.
The templates should be in the same order as the URLs. If two URLs are semantically the same, they should have the same template (include the template multiple times, i.e. if two URLs are the same, they should have the same template).
The placeholders should be in the format <placeholder_name>.
"""

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(proxy_client=get_proxy_client())

    def complete(self, conversation: list[dict]):
        messages = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            messages.append({ "role": role, "content": content })
        completion = self.client.chat.completions.create(model_name=self.model_name, messages=messages, temperature=0.0)
        content = completion.choices[0].message.content
        return content

    def __call__(self, prompt: str):
        return self.complete([ user_message(prompt) ])

def user_message(prompt: str):
    return { "role": "user", "content": prompt }
def assistant_message(prompt: str):
    return { "role": "assistant", "content": prompt }
def system_message(prompt: str):
    return { "role": "system", "content": prompt }
