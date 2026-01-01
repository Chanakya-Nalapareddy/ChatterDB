# src/rag_semantic/question_rewriter.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

from src.rag_semantic.sql_generator_gpt import make_llm  # reuse your make_llm


def rewrite_question(
    *,
    user_question: str,
    thread_summary: str,
    thread_state: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Returns {"rewritten_question": "...", "notes": "..."}.
    Notes are optional and should be short.
    """
    llm: AzureChatOpenAI = make_llm(temperature=0.0)

    system = """
You are a query rewriter for an analytics assistant.

Goal: rewrite the user's new message into a fully specified standalone question,
using prior thread context (summary/state/messages). Keep it faithful to user intent.

Rules:
- Do NOT answer the question. Only rewrite it.
- If user says "same", "now", "also", "instead", "make it top 10", "last month", etc.,
  incorporate relevant prior context (dimensions/metrics/filters/topK/topN).
- If the new message is already standalone, return it unchanged.
- Output STRICT JSON only:
  {"rewritten_question":"...","notes":"..."}
""".strip()

    payload = {
        "user_question": user_question,
        "thread_summary": thread_summary,
        "thread_state": thread_state,
        "recent_messages": recent_messages[-6:],
    }

    user = (
        "Rewrite the question as standalone.\n"
        "Return JSON only:\n"
        '{"rewritten_question":"...","notes":"..."}\n\n'
        f"INPUT:\n{json.dumps(payload)}"
    )

    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content

    # small robust JSON parse
    s = (raw or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return {"rewritten_question": user_question, "notes": "rewriter_fallback_no_json"}

    try:
        obj = json.loads(s[start : end + 1])
        rq = (obj.get("rewritten_question") or "").strip()
        if not rq:
            return {"rewritten_question": user_question, "notes": "rewriter_empty"}
        return {"rewritten_question": rq, "notes": (obj.get("notes") or "").strip()}
    except Exception:
        return {"rewritten_question": user_question, "notes": "rewriter_parse_error"}
