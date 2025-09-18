import importlib
import os
import sys
import types

os.environ.setdefault("CONVO_API_KEY", "test-key")

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")

    def load_dotenv(*args, **kwargs):
        return None

    dotenv_module.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv_module

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")

    class _Response:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    def post(*args, **kwargs):
        return _Response()

    requests_module.post = post
    sys.modules["requests"] = requests_module

if "langchain" not in sys.modules:
    langchain_module = types.ModuleType("langchain")
    schema_module = types.ModuleType("langchain.schema")
    prompts_module = types.ModuleType("langchain.prompts")

    class _BaseMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage(_BaseMessage):
        pass

    class SystemMessage(_BaseMessage):
        pass

    class ChatPromptTemplate:
        def __init__(self, messages):
            self._messages = messages

        @classmethod
        def from_messages(cls, messages):
            return cls(messages)

        def format_messages(self):
            return self._messages

    schema_module.HumanMessage = HumanMessage
    schema_module.SystemMessage = SystemMessage
    prompts_module.ChatPromptTemplate = ChatPromptTemplate

    langchain_module.schema = schema_module
    langchain_module.prompts = prompts_module

    sys.modules["langchain"] = langchain_module
    sys.modules["langchain.schema"] = schema_module
    sys.modules["langchain.prompts"] = prompts_module

if "gai_templates" not in sys.modules:
    gai_module = types.ModuleType("gai_templates")
    common_module = types.ModuleType("gai_templates.common")
    constants_module = types.ModuleType("gai_templates.common.constants")
    omni_module = types.ModuleType("gai_templates.omni_llm_v2")

    class OmniChatLLM:
        def __init__(self, *args, **kwargs):
            self._last = None

        def invoke(self, messages):
            self._last = messages

            class _Response:
                def __init__(self):
                    self.content = ""

            return _Response()

    constants_module.OPENAI = "openai"
    constants_module.GPT_5 = "gpt-5"
    omni_module.OmniChatLLM = OmniChatLLM

    gai_module.common = common_module
    gai_module.omni_llm_v2 = omni_module
    common_module.constants = constants_module

    sys.modules["gai_templates"] = gai_module
    sys.modules["gai_templates.common"] = common_module
    sys.modules["gai_templates.common.constants"] = constants_module
    sys.modules["gai_templates.omni_llm_v2"] = omni_module

_pkg = importlib.import_module(".src.mmm_ai", __name__)
sys.modules[__name__] = _pkg
