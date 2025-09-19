import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ..llm.client import get_llm
from ..config.prompts import SUMMARIZER_SYSTEM

def summarize_metrics(diagnostics_json: dict) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SUMMARIZER_SYSTEM),
        HumanMessage(content=json.dumps(diagnostics_json, indent=2))
    ])
    out = get_llm().invoke(prompt.format_messages())
    return out.content
