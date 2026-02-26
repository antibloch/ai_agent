def route_after_validator(state: dict) -> str:
    plan = state.get("plan", {})
    steps = plan.get("steps", [])
    return "executor" if len(steps) > 0 else "responder"

def route_after_gate(state: dict) -> str:
    plan = state.get("plan", {}) or {}
    steps = plan.get("steps", []) or []
    return "validator" if steps else "responder"