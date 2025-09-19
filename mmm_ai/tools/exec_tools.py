from typing import Any, Dict, Optional

def exec_mmm_function_local(function_name: str, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Safely execute a whitelisted MMM function (no arbitrary code).
    Map keys to your real functions below.
    """
    ALLOWED = {
        # "reestimate_with_robust_se": your_pkg.reestimate_with_robust_se,
        # "apply_adstock_and_refit": your_pkg.apply_adstock_and_refit,
        # "group_channels_and_refit": your_pkg.group_channels_and_refit,
    }
    if function_name not in ALLOWED:
        return {"error": f"Function '{function_name}' is not whitelisted."}
    try:
        fn = ALLOWED[function_name]
        out = fn(**(kwargs or {}))
        return {"ok": True, "result": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}
