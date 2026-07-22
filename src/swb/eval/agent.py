"""The eval agent: a thin tool-calling loop over OpenRouter.

Receives only the task query (no history inlined) and must use the search tools to
find the answer. Returns the final answer plus the ToolBox's query log.
"""

from __future__ import annotations

import json

from swb.eval.tools import TOOL_SCHEMAS, ToolBox
from swb.llm import chat
from swb.schema import ToolCall

_SYSTEM = (
    "You are answering questions about a group chat you can only access through search "
    "tools. You cannot see the conversation directly. Use the tools to find evidence, "
    "then answer. Keep your final answer as short as possible — just the fact asked for, "
    "with no extra words. For yes/no questions answer exactly 'yes' or 'no'."
)

_MAX_ITERS = 6


def run_agent(query: str, toolbox: ToolBox, model: str, temperature: float = 0.0) -> tuple[str, list[ToolCall]]:
    """Drive the tool-calling loop for one task. Returns (answer, query_log)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": query},
    ]

    for _ in range(_MAX_ITERS):
        msg = chat(messages, model=model, temperature=temperature, tools=TOOL_SCHEMAS)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return (msg.content or "").strip(), toolbox.log

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = toolbox.dispatch(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Ran out of iterations; ask for a final answer with no tools.
    messages.append({"role": "user", "content": "Give your best final answer now, no tools."})
    final = chat(messages, model=model, temperature=temperature)
    return (final.content or "").strip(), toolbox.log
