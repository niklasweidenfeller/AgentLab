from gen_ai_hub.proxy.native.openai import embeddings

class Vectorizer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, text: str) -> list[float]:
        return self.vectorize(text)

    def vectorize(self, text: str) -> list[float]:
        response = embeddings.create(
            input=text,
            model_name=self.model_name
        )
        return response.data[0].embedding
