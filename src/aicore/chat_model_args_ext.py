from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from aicore.llm import AiCoreOpenAIModelArgs

EXTENDED_CHAT_MODEL_ARGS_DICT = CHAT_MODEL_ARGS_DICT.copy()
EXTENDED_CHAT_MODEL_ARGS_DICT.update({
    "aicore/gpt-4o": AiCoreOpenAIModelArgs(
        model_name="gpt-4o",
        max_total_tokens=16_384,
        max_input_tokens=16_384,
        max_new_tokens=4096,
    ),
})
