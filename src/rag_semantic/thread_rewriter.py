from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from src.rag_semantic.sql_generator_gpt import make_llm


def rewrite_followup(
    *,
    user_text: str,
    thread_summary: str,
    thread_state: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Returns:
      {"rewritten": "...", "notes": "..."}  (notes optional)
    """
    llm = make_llm(temperature=0.0)

    system_prompt = """
You are a question rewriter for a data analytics chat app.

Rewrite the user's latest message into a fully specified, standalone analytics question
using the thread context (summary/state/recent messages).

Rules:
- Do NOT answer the question.
- If the user says "same", "now", "also", "instead", "make it top 10", "exclude X", "last month", etc.,
  incorporate the missing context from memory.
- If already standalone, keep it unchanged.
- Output STRICT JSON only:
  {"rewritten":"...","notes":"..."}
""".strip()

    payload = {
        "user_text": user_text,
        "thread_summary": thread_summary,
        "thread_state": thread_state,
        "recent_messages": [{"role": m.get("role"), "content": m.get("content")} for m in recent_messages[-8:]],
    }

    user_prompt = (
        "Rewrite the question as standalone. Return JSON only:\n"
        '{"rewritten":"...","notes":"..."}\n\n'
        f"INPUT:\n{json.dumps(payload)}"
    )

    raw = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    s = (raw or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return {"rewritten": user_text, "notes": "rewriter_no_json"}

    try:
        obj = json.loads(s[start : end + 1])
        rewritten = (obj.get("rewritten") or "").strip() or user_text
        notes = (obj.get("notes") or "").strip()
        return {"rewritten": rewritten, "notes": notes}
    except Exception:
        return {"rewritten": user_text, "notes": "rewriter_parse_error"}
