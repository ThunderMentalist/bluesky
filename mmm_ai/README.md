# mmm_ai

Agentic framework for MMM model diagnostics, summaries, recommendations, and comparisons.

## Highlights
- Uses **Conversation Service V2** for tool-calling via raw HTTP in the **Orchestrator**.
- Uses **`gai_templates` V2** with **`GPT_5`** for summarizer, model-doctor, and comparator agents.
- Clear extension points to plug in your MMM package (diagnostics + safe execution whitelist).

## Quick start

1. Copy `.env.example` to `.env` and set your key and env:
```

cp .env.example .env

```
2. Install deps:
```

pip install -r requirements.txt

```
3. Run the CLI (will raise NotImplementedError until you wire `tools/diagnostics.py`):
```

python -m mmm_ai.app.cli --model-ref mm_ols_2025w36 --kpi total_sales

```

## Integrations
- Implement `run_mmm_diagnostics_local(...)` in `tools/diagnostics.py`.
- Whitelist safe actions in `tools/exec_tools.py` (optional).
