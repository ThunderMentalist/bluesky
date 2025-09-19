import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ..llm.client import get_llm
from ..config.prompts import COMPARATOR_SYSTEM

def compare_runs(curr_diag: dict, base_diag: dict, curr_summary: str, base_summary: str) -> str:
    payload = {"current": {"diagnostics": curr_diag, "summary": curr_summary},
               "baseline": {"diagnostics": base_diag, "summary": base_summary}}
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=COMPARATOR_SYSTEM),
        HumanMessage(content=json.dumps(payload, indent=2))
    ])
    out = get_llm().invoke(prompt.format_messages())
    return out.content
