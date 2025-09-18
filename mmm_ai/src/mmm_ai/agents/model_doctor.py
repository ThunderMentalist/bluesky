import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ..llm.client import get_llm
from ..config.prompts import MODEL_DOCTOR_SYSTEM

def model_doctor_advice(diagnostics_json: dict, summary_text: str) -> str:
    payload = {"diagnostics": diagnostics_json, "summary": summary_text}
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=MODEL_DOCTOR_SYSTEM),
        HumanMessage(content=json.dumps(payload, indent=2))
    ])
    out = get_llm().invoke(prompt.format_messages())
    return out.content
