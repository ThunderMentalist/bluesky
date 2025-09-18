import json
from typing import Any, Dict, Optional, Tuple

from ..config.prompts import ORCHESTRATOR_SYSTEM
from ..config.settings import MODEL_NAME_STR, INFERENCE_DEFAULT
from ..llm.service_api import post_v2, extract_tool_call, extract_text, extract_first_json_block
from ..tools.diagnostics import run_mmm_diagnostics_local
from ..tools.exec_tools import exec_mmm_function_local

# Tool schema for function-calling (V2)
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "run_mmm_diagnostics",
                "description": "Compute OLS diagnostics for a given MMM model reference and KPI.",
                "input_schema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "model_ref": {"type": "string", "description": "Identifier/path for fitted model."},
                            "kpi": {"type": "string", "description": "KPI target (e.g., 'total_sales')."}
                        },
                        "required": ["model_ref", "kpi"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "exec_mmm_function",
                "description": "Execute a whitelisted MMM function (e.g., robust SE refit).",
                "input_schema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "function_name": {"type": "string"},
                            "kwargs": {"type": "object"}
                        },
                        "required": ["function_name"]
                    }
                }
            }
        }
    ],
    "tool_choice": {"any": {}}
}

def _handle_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if tool_name == "run_mmm_diagnostics":
        model_ref = tool_input.get("model_ref")
        kpi = tool_input.get("kpi")
        result = run_mmm_diagnostics_local(model_ref, kpi)
        return tool_name, result
    elif tool_name == "exec_mmm_function":
        function_name = tool_input.get("function_name")
        kwargs = tool_input.get("kwargs", {})
        result = exec_mmm_function_local(function_name, kwargs)
        return tool_name, result
    return tool_name, {"error": f"Unknown tool: {tool_name}"}

def orchestrator_analyze(model_ref: str, kpi: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    One full orchestration round:
      1) Ask Orchestrator to analyze the model -> it will request the diagnostics tool
      2) Execute tool locally and return tool_result
      3) Read final assistant message (summary + JSON block)
    """
    user_text = f"Analyze model '{model_ref}' for KPI '{kpi}'. Compute diagnostics and summarize."

    first_req = {
        "model_name": MODEL_NAME_STR,  # raw HTTP uses string
        "client_id": "default",
        "user_id": "default",
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "system_prompt": [{"text": ORCHESTRATOR_SYSTEM}],
        "inference_config": {"temperature": 0.2, "top_p": 1, "max_completion_tokens": 500},
        "tool_config": TOOL_CONFIG,
    }
    if thread_id:
        first_req["thread_id"] = thread_id

    first_resp = post_v2(first_req)
    tool_call = extract_tool_call(first_resp)
    if not tool_call:
        final_text = extract_text(first_resp)
        return {"text": final_text, "json": None, "thread_id": first_resp.get("thread_id")}

    tool_name = tool_call.get("name")
    tool_input = tool_call.get("input", {})
    tool_use_id = tool_call.get("toolUseId") or tool_call.get("tool_use_id")

    # Execute tool locally
    _, tool_result_json = _handle_tool_call(tool_name, tool_input)

    follow_up_req = {
        "model_name": MODEL_NAME_STR,
        "client_id": "default",
        "user_id": "default",
        "messages": [
            {"role": "user", "content": [{"text": user_text}]},
            {"role": "assistant", "content": [
                {"text": f"Calling tool {tool_name}."},
                {"tool_use": {"tool_use_id": tool_use_id, "name": tool_name, "input": tool_input}}
            ]},
            {"role": "user", "content": [
                {"tool_result": {"tool_use_id": tool_use_id, "content": [{"json": tool_result_json}]}}
            ]}
        ],
        "system_prompt": [{"text": ORCHESTRATOR_SYSTEM}],
        "inference_config": INFERENCE_DEFAULT,
        "tool_config": TOOL_CONFIG,
    }
    if thread_id:
        follow_up_req["thread_id"] = thread_id

    final_resp = post_v2(follow_up_req)
    final_text = extract_text(final_resp)
    json_block = extract_first_json_block(final_text)
    return {"text": final_text, "json": json_block or tool_result_json, "thread_id": final_resp.get("thread_id")}
