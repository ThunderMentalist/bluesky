from typing import Any, Dict, Optional
from ..agents.orchestrator import orchestrator_analyze
from ..agents.summarizer import summarize_metrics
from ..agents.model_doctor import model_doctor_advice
from ..agents.comparator import compare_runs

def analyze_model(model_ref: str, kpi: str, baseline_model_ref: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates a full analysis:
      - Orchestrator (tool-calls diagnostics) -> summary + JSON
      - Summarizer (stakeholder summary)
      - Model Doctor (targeted recommendations)
      - Optional Comparator (current vs baseline)
    """
    # Current model
    current = orchestrator_analyze(model_ref, kpi)
    curr_text = current["text"]
    curr_json = current["json"]

    curr_summary = summarize_metrics(curr_json) if curr_json else ""
    curr_doctor = model_doctor_advice(curr_json, curr_summary) if curr_json else ""

    result = {
        "current_summary_orchestrator": curr_text,
        "current_summary_summarizer": curr_summary,
        "current_model_doctor": curr_doctor,
        "current_json": curr_json
    }

    if baseline_model_ref:
        baseline = orchestrator_analyze(baseline_model_ref, kpi)
        base_text = baseline["text"]
        base_json = baseline["json"]
        base_summary = summarize_metrics(base_json) if base_json else ""

        delta = compare_runs(curr_json, base_json, curr_summary, base_summary)

        result.update({
            "baseline_summary_orchestrator": base_text,
            "baseline_summary_summarizer": base_summary,
            "baseline_json": base_json,
            "comparison_delta": delta
        })

    return result
