import json
from typing import List, Dict, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from json_utils import _safe_json_loads, _compact_json, _summarize_tool_output, TRUNCATION_TOOL_LIMIT, SENSITIVE_KEY_PATTERNS, _sanitize_sensitive_data

def format_msg(m: BaseMessage) -> str:
    role = m.__class__.__name__
    content = (getattr(m, "content", "") or "").strip()
    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
        content += "\n\n[tool_calls]\n" + json.dumps(getattr(m, "tool_calls"), indent=2, ensure_ascii=False)
    if getattr(m, "name", None):
        content = f"[tool={m.name}]\n{content}"
    return f"{role}:\n{content}\n"


def get_current_round_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    msgs = list(messages or [])
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            return msgs[i:]
    return msgs

def get_latest_tool_per_name(messages: Sequence[BaseMessage]) -> Dict[str, ToolMessage]:
    latest_any: Dict[str, ToolMessage] = {}
    latest_ok: Dict[str, ToolMessage] = {}
    for m in messages or []:
        if isinstance(m, ToolMessage):
            latest_any[m.name] = m
            payload = _safe_json_loads((m.content or "").strip())
            if isinstance(payload, dict) and payload.get("ok") is True:
                latest_ok[m.name] = m
    out = dict(latest_any)
    out.update(latest_ok)
    return out

def get_best_tool_message_by_call_id(messages: Sequence[BaseMessage]) -> Dict[str, ToolMessage]:
    latest_any: Dict[str, ToolMessage] = {}
    latest_ok: Dict[str, ToolMessage] = {}
    for m in messages or []:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if not tcid:
                continue
            latest_any[tcid] = m
            payload = _safe_json_loads((m.content or "").strip())
            if isinstance(payload, dict) and payload.get("ok") is True:
                latest_ok[tcid] = m
    out = dict(latest_any)
    out.update(latest_ok)
    return out

def _compact_err(payload: dict) -> str:
    try:
        s = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        s = str(payload)
    return s[:TRUNCATION_TOOL_LIMIT] + (" ...[truncated]" if len(s) > TRUNCATION_TOOL_LIMIT else "")

def detect_latest_tool_error(messages: Sequence[BaseMessage], max_k: int = 8) -> Dict[str, Any] | None:
    current_round = get_current_round_messages(messages)
    tool_msgs = [m for m in current_round if isinstance(m, ToolMessage)][-max_k:]

    error_markers = [
        "Traceback", "IndentationError", "SyntaxError", "NameError", "KeyError",
        "TypeError", "ValueError", "Exception", "Error:", "ERROR", "invalid", "missing", "failed",
    ]

    for m in reversed(tool_msgs):
        raw = (m.content or "").strip()
        payload = _safe_json_loads(raw)

        if isinstance(payload, dict):
            if payload.get("ok") is False:
                return {"tool": m.name, "error": _compact_err(payload), "tool_message": raw}
            result = payload.get("result", None)
            if isinstance(result, str) and any(k in result for k in error_markers):
                return {"tool": m.name, "error": result[:800], "tool_message": raw}
            err = payload.get("error", None)
            if isinstance(err, str) and any(k in err for k in error_markers):
                return {"tool": m.name, "error": err[:800], "tool_message": raw}

        if any(k in raw for k in error_markers):
            return {"tool": m.name, "error": raw[:800], "tool_message": raw}
    return None


def _format_tool_calls_block(tool_calls: list) -> str:
    """Normalized formatting for ALL tools.
    - Redacts sensitive keys from args before they reach the LLM.
    """
    out = []
    for tc in tool_calls or []:
        tid = tc.get("id") or tc.get("tool_call_id") or ""
        name = tc.get("name") or ""

        # Copy + sanitize
        args = _sanitize_sensitive_data(dict(tc.get("args") or {}))

        # === THE FIX YOU ASKED FOR (still works after sanitization) ===
        if name and name != "get_charity_stats" and "tool_name" not in args:
            args["tool_name"] = name
        # ============================

        out.append(f"TOOL_CALL[{name} id={tid}] args={_compact_json(args, max_chars=TRUNCATION_TOOL_LIMIT)}")
    return "\n".join(out)


def format_history_for_gate(messages: Sequence[BaseMessage]) -> str:
    current_round = get_current_round_messages(messages)
    best_tool_by_call_id = get_best_tool_message_by_call_id(current_round)
    lines = []
    seen_call_ids: set[str] = set()
    for m in current_round:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            if (m.content or "").strip().startswith("System Note:"):
                continue
            if getattr(m, "tool_calls", None):
                lines.append(_format_tool_calls_block(m.tool_calls))
                for tc in m.tool_calls or []:
                    tcid = tc.get("id") or tc.get("tool_call_id")
                    if tcid and tcid in best_tool_by_call_id:
                        tm = best_tool_by_call_id[tcid]
                        seen_call_ids.add(tcid)
                        lines.append(_summarize_tool_output(tm.name, tm.content))
            if (m.content or "").strip():
                lines.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid in seen_call_ids:
                continue
            lines.append(_summarize_tool_output(m.name, m.content))
    return "\n".join(lines) if lines else "(empty)"


def build_cached_tool_outputs(messages: Sequence[BaseMessage], max_chars: int = TRUNCATION_TOOL_LIMIT) -> str:
    current_round = get_current_round_messages(messages)
    last_by_tool: Dict[str, str] = {}
    for m in current_round:
        if isinstance(m, ToolMessage):
            last_by_tool[m.name] = m.content or ""
    if not last_by_tool:
        return "(none)"

    blocks = []
    for tool_name, content in last_by_tool.items():
        c = content.strip()
        # Sanitize before showing in gate prompt
        try:
            payload = _safe_json_loads(c)
            if payload is not None:
                sanitized = _sanitize_sensitive_data(payload)
                c = _compact_json(sanitized, max_chars=max_chars)
            else:
                if len(c) > max_chars:
                    c = c[:max_chars] + " ...[truncated]"
        except Exception:
            if len(c) > max_chars:
                c = c[:max_chars] + " ...[truncated]"

        blocks.append(f"- {tool_name}: {c}")
    return "\n".join(blocks)


def format_history_for_planner(messages: Sequence[BaseMessage], *, drop_last_user: bool = True) -> str:
    msgs = list(messages) if messages else []
    if drop_last_user:
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                msgs = msgs[:i]
                break
    latest_tools_by_name = get_latest_tool_per_name(msgs)
    best_tool_by_call_id = get_best_tool_message_by_call_id(msgs)
    lines = []
    seen_call_ids: set[str] = set()
    for m in msgs:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            if (m.content or "").strip().startswith("System Note:"):
                continue
            if getattr(m, "tool_calls", None):
                visible_calls = []
                for tc in m.tool_calls or []:
                    tcid = tc.get("id") or tc.get("tool_call_id")
                    if tcid and tcid in best_tool_by_call_id:
                        tm = best_tool_by_call_id[tcid]
                        payload = _safe_json_loads((tm.content or "").strip())
                        if isinstance(payload, dict) and payload.get("ok") is True:
                            visible_calls.append(tc)
                    else:
                        visible_calls.append(tc)
                if visible_calls:
                    lines.append(_format_tool_calls_block(visible_calls))
                    for tc in visible_calls:
                        tcid = tc.get("id") or tc.get("tool_call_id")
                        if tcid and tcid in best_tool_by_call_id:
                            tm = best_tool_by_call_id[tcid]
                            lines.append(_summarize_tool_output(tm.name, tm.content))
                            seen_call_ids.add(tcid)
            if (m.content or "").strip():
                lines.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid in seen_call_ids:
                continue
            if m is not latest_tools_by_name.get(m.name):
                continue
            lines.append(_summarize_tool_output(m.name, m.content))
    return "\n".join(lines) if lines else "(no prior history)"


def format_history_for_responder(messages: Sequence[BaseMessage]) -> str:
    latest_tools_by_name = get_latest_tool_per_name(messages)
    best_tool_by_call_id = get_best_tool_message_by_call_id(messages)
    lines = []
    seen_call_ids: set[str] = set()
    for m in messages or []:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            if (m.content or "").strip().startswith("System Note:"):
                continue
            if getattr(m, "tool_calls", None):
                visible_calls = []
                for tc in m.tool_calls or []:
                    tcid = tc.get("id") or tc.get("tool_call_id")
                    if tcid and tcid in best_tool_by_call_id:
                        tm = best_tool_by_call_id[tcid]
                        payload = _safe_json_loads((tm.content or "").strip())
                        if isinstance(payload, dict) and payload.get("ok") is True:
                            visible_calls.append(tc)
                    else:
                        visible_calls.append(tc)
                if visible_calls:
                    lines.append(_format_tool_calls_block(visible_calls))
                    for tc in visible_calls:
                        tcid = tc.get("id") or tc.get("tool_call_id")
                        if not tcid:
                            continue
                        seen_call_ids.add(tcid)
                        tm = best_tool_by_call_id.get(tcid)
                        if tm:
                            lines.append(_summarize_tool_output(tm.name, tm.content))
            if (m.content or "").strip():
                lines.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid in seen_call_ids:
                continue
            if m is not latest_tools_by_name.get(m.name):
                continue
            payload = _safe_json_loads((m.content or "").strip())
            if isinstance(payload, dict) and payload.get("ok") is False:
                continue
            lines.append(_summarize_tool_output(m.name, m.content))
    return "\n".join(lines) if lines else "(empty)"