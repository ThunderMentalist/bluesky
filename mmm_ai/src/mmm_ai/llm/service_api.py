import json
import re
import requests
from typing import Any, Dict, Optional
from ..config.settings import CONVO_API, HEADERS

def post_v2(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(CONVO_API, headers=HEADERS, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()

def extract_tool_call(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = (resp.get("data", {}) or {}).get("output", {}).get("message", {}).get("content", [])
    for item in content:
        if "toolUse" in item:  # provider variant
            return item["toolUse"]
        if "tool_use" in item:  # provider variant
            return item["tool_use"]
    return None

def extract_text(resp: Dict[str, Any]) -> str:
    content = (resp.get("data", {}) or {}).get("output", {}).get("message", {}).get("content", [])
    for item in content:
        if "text" in item:
            return item["text"]
    return ""

def extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None
