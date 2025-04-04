from gen_ai_hub.proxy.native.openai import OpenAI as AiCoreOpenAI

from agentlab.llm import tracking
from agentlab.llm.chat_api import ChatModel, OpenAIModelArgs

class AiCoreOpenAIModelArgs(OpenAIModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return AiCoreOpenAiChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

class AiCoreOpenAiOverrideClient(AiCoreOpenAI):
    def __init__(self, **kwargs):
        del kwargs["api_key"]
        super().__init__(**kwargs)

class AiCoreOpenAiChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AiCoreOpenAiOverrideClient,
            pricing_func=tracking.get_pricing_openai,
        )
