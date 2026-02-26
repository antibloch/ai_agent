import json
import re
from typing import Dict, Any

TRUNCATION_TOOL_LIMIT = 2000

# Keys/patterns that should never leak into LLM history
SENSITIVE_KEY_PATTERNS = {
    "auth-token", "auth_token", "authorization", "token",
    "api_key", "apikey", "secret", "password", "credential",
    "private_key", "access_token", "refresh_token", "x-api-key", "bearer"
}

def _sanitize_sensitive_data(obj: Any) -> Any:
    """Recursively REMOVE sensitive keys entirely (do not keep "[REDACTED]")."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            key_lower = str(k).lower().replace("-", "_").replace(" ", "_")
            if any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS):
                continue  # ← KEY CHANGE: drop the whole key-value pair
            sanitized[k] = _sanitize_sensitive_data(v)
        return sanitized
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_sensitive_data(item) for item in obj]
    else:
        return obj
    

def _extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty LLM output")
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No '{' found in LLM output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1].strip()
    raise ValueError("Unbalanced JSON braces in LLM output")

def _parse_plan(raw: str) -> Dict:
    if not raw:
        return {"steps": [], "missing_args": []}
    try:
        json_str = _extract_first_json_object(raw)
        return json.loads(json_str)
    except:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
    return {"steps": [], "missing_args": []}

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _compact_json(obj: Any, max_chars: int = 900) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        return s[:max_chars] + " ...[truncated]"
    return s


def _summarize_tool_output(tool_name: str, tool_content: str) -> str:
    payload = _safe_json_loads(tool_content) if isinstance(tool_content, str) else None
    if not payload:
        return f"TOOL[{tool_name}] -> {tool_content}"

    if isinstance(payload, dict) and payload.get("ok") is False:
        err = payload.get("error") or payload.get("result") or payload
        return f"TOOL[{tool_name}] ERROR -> {_compact_json(err, max_chars=TRUNCATION_TOOL_LIMIT)}"

    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    # === REDACT HERE TOO (in case backend echoes tokens) ===
    result = _sanitize_sensitive_data(result)
    # =======================================================

    if tool_name == "get_charity_stats" and isinstance(result, dict):
        data = result.get("data")
        tool = result.get("tool") or result.get("query")
        if tool == "charity_donor_count" and isinstance(data, list):
            pairs = []
            for row in data:
                name = row.get("charityName")
                cnt = row.get("donorCount")
                if name is not None and cnt is not None:
                    pairs.append(f"{name}: {cnt}")
            if pairs:
                return "TOOL[get_charity_stats:charity_donor_count] -> " + "; ".join(pairs)

    if tool_name in ("Python_REPL", "python_repl", "PythonREPLTool"):
        return f"TOOL[Python_REPL] -> {_compact_json(result, max_chars=TRUNCATION_TOOL_LIMIT)}"

    return f"TOOL[{tool_name}] -> {_compact_json(result, max_chars=TRUNCATION_TOOL_LIMIT)}"