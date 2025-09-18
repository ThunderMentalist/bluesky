import argparse
from .pipeline import analyze_model

def main():
    parser = argparse.ArgumentParser(description="Run MMM agentic analysis.")
    parser.add_argument("--model-ref", required=True, help="Current model reference")
    parser.add_argument("--kpi", required=True, help="KPI name, e.g., total_sales")
    parser.add_argument("--baseline-model-ref", default=None, help="Optional baseline model reference")
    args = parser.parse_args()

    try:
        result = analyze_model(args.model_ref, args.kpi, args.baseline_model_ref)
        print("\n=== Orchestrator (current) ===\n", result["current_summary_orchestrator"])
        print("\n=== Summarizer (current) ===\n", result["current_summary_summarizer"])
        print("\n=== Model Doctor (current) ===\n", result["current_model_doctor"])
        if "comparison_delta" in result:
            print("\n=== Comparison (current vs baseline) ===\n", result["comparison_delta"])
    except NotImplementedError as e:
        print(f"[ACTION REQUIRED] Integrate your MMM package: {e}")

if __name__ == "__main__":
    main()
