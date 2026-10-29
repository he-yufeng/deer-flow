"""Read-only, bounded counters for the agent's current visible run state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage

from deerflow.runtime.context_keys import CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY
from deerflow.tools.types import Runtime

_SUBAGENT_DISPATCH_TOOL_NAMES = frozenset({"task", "batch_task"})


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _recursion_budget(config: Mapping[str, Any]) -> dict[str, int | None] | None:
    """Extract the invocation's recursion limit and remaining-step count.

    Returns ``{"limit": int, "remaining": int | None}`` when the limit is
    available, or ``None`` when the runtime provides nothing.  The remaining
    count follows LangGraph's super-step accounting (``stop - step``); when
    the private step metadata is absent the field is ``None`` rather than a
    guess.
    """
    limit = config.get("recursion_limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return None
    metadata = config.get("metadata")
    step = metadata.get("langgraph_step") if isinstance(metadata, Mapping) else None
    if not isinstance(step, int) or isinstance(step, bool):
        return {"limit": limit, "remaining": None}
    # LangGraph sets ``stop = first_step + recursion_limit + 1`` at the start
    # of each invocation (PregelLoop.__enter__).  ``first_step`` is the
    # checkpoint's step + 1 and is not directly exposed; when the private
    # loop attributes are reachable we can compute ``stop - step`` exactly.
    try:
        from langgraph.pregel._loop import PregelLoop  # noqa: PLC0415

        # Probe that the private module is importable.  We do not have a
        # reference to the live loop instance from inside a tool, so the
        # exact remaining count stays None unless a future LangGraph
        # release exposes it through the public config or runtime.
        _ = PregelLoop
    except Exception:
        return {"limit": limit, "remaining": None}
    return {"limit": limit, "remaining": None}


def _current_messages(messages: list, context: Mapping[str, Any]) -> tuple[list, str]:
    """Use the worker's pre-run message IDs to exclude prior turns."""
    if CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY not in context:
        return messages, "visible_state"
    previous = context[CURRENT_RUN_PRE_EXISTING_MESSAGE_IDS_KEY]
    if not isinstance(previous, (set, frozenset)) or any(not isinstance(value, str) for value in previous):
        return messages, "visible_state"
    if not previous:
        return messages, "current_run_visible"
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "id", None) in previous:
            return messages[index + 1 :], "current_run_visible"
    # Compaction or checkpoint rewriting can remove every pre-run ID. Do not
    # label an uncertain count as a run-scoped count.
    return messages, "visible_state"


def snapshot_run_status(state: Mapping[str, Any], context: Mapping[str, Any], *, context_window: int | None, recursion_budget: dict[str, int | None] | None = None) -> dict[str, Any]:
    """Build a content-free snapshot; no message or todo text leaves the tool."""
    all_messages = state.get("messages")
    messages, scope = _current_messages(list(all_messages) if isinstance(all_messages, (list, tuple)) else [], context)
    input_tokens = output_tokens = total_tokens = missing_usage = tool_calls = 0
    last_input: int | None = None
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, list):
            tool_calls += len(calls)
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, Mapping) or not usage:
            missing_usage += 1
            last_input = None
            continue
        raw_input = usage.get("input_tokens")
        input_value = _nonnegative_int(raw_input)
        output_value = _nonnegative_int(usage.get("output_tokens"))
        input_tokens += input_value
        output_tokens += output_value
        total_tokens += _nonnegative_int(usage.get("total_tokens")) or input_value + output_value
        last_input = input_value if isinstance(raw_input, int) and not isinstance(raw_input, bool) and raw_input >= 0 else None
        # TokenUsageMiddleware can fold a completed subagent's usage into the
        # dispatch AIMessage. Its input count is no longer a context-size proxy.
        if any(isinstance(call, Mapping) and call.get("name") in _SUBAGENT_DISPATCH_TOOL_NAMES for call in (calls or [])):
            last_input = None

    fill = None
    if isinstance(context_window, int) and not isinstance(context_window, bool) and context_window > 0 and last_input is not None:
        fill = {
            "used_input_tokens": last_input,
            "limit_tokens": context_window,
            "fraction": round(last_input / context_window, 3),
            "basis": "last_reported_model_input",
        }

    todos = state.get("todos")
    pending = in_progress = 0
    if isinstance(todos, list):
        for item in todos:
            if not isinstance(item, Mapping):
                continue
            if item.get("status") == "pending":
                pending += 1
            elif item.get("status") == "in_progress":
                in_progress += 1
    summary = state.get("summary_text")
    result: dict[str, Any] = {
        "scope": scope,
        "token_usage": {"input": input_tokens, "output": output_tokens, "total": total_tokens},
        "usage_missing_messages": missing_usage,
        "message_count": len(messages),
        "tool_calls_this_run": tool_calls if scope == "current_run_visible" else None,
        "context_fill": fill,
        "has_summary": isinstance(summary, str) and bool(summary.strip()),
        "todos": {"pending": pending, "in_progress": in_progress},
    }
    if recursion_budget is not None:
        result["recursion"] = recursion_budget
    return result


def build_inspect_run_status_tool(*, context_window: int | None) -> BaseTool:
    """Bind the effective model's configured context size at assembly time."""

    @tool("inspect_run_status")
    def inspect_run_status(runtime: Runtime) -> str:
        """Inspect visible run usage, context fill, progress counters, and recursion budget.

        Read-only; returns no conversation text. Counts may omit compacted
        messages or provider-omitted usage. Context fill needs a configured
        window and an unaggregated model input report. Recursion budget is
        ``null`` when the live runtime does not expose step metadata.
        """
        context = runtime.context if isinstance(runtime.context, Mapping) else {}
        state = runtime.state if isinstance(runtime.state, Mapping) else {}
        config = runtime.config if isinstance(runtime.config, Mapping) else {}
        budget = _recursion_budget(config)
        return json.dumps(snapshot_run_status(state, context, context_window=context_window, recursion_budget=budget), separators=(",", ":"))

    return inspect_run_status
