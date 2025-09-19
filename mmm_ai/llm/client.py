from gai_templates.omni_llm_v2 import OmniChatLLM
from gai_templates.common.constants import OPENAI, GPT_5
from ..config.settings import API_KEY, ENV

# Single shared client for GPT_5 via V2
_LLM = OmniChatLLM(
    api_key=API_KEY,
    provider=OPENAI,
    model_params={"model_name": GPT_5},  # UPDATED: V2 uses "model_name"
    env=ENV,
)

def get_llm() -> OmniChatLLM:
    return _LLM
