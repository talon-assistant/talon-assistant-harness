"""core/deep_search_agent.py — Agentic RAG via LLM tool calls.

Standard RAG is one-shot: retrieve → generate. The LLM never actually
searches the documents — it reads whatever vector math and keyword
matching handed it. This fails on queries where the relevant content
doesn't semantically match the question's phrasing.

DeepSearchAgent lets the LLM drive retrieval iteratively. Each turn the
LLM emits a JSON tool call. Python executes it and returns the result.
The LLM decides when it has enough and emits {"tool": "done"}.

Tools:
    search(query, top_k)            — run the standard retrieval pipeline
    read_page(chunk_id)             — return full text of one chunk
    list_sources()                  — list all documents + chunk counts
    filter_source(filename, query)  — search restricted to one document
    done(answer_ready)              — end the loop

Triggered by user saying "deep search", "deep research", "research
deeply", etc. before their query.
"""
from __future__ import annotations

import json
import re

import logging
log = logging.getLogger(__name__)


_AGENT_SYSTEM_PROMPT = """You are a document retrieval agent. Your job is to find relevant excerpts from the user's document library to answer their question.

You have these tools. Respond with ONE JSON object per turn — nothing else.

1. search(query, top_k=8)
   Run a keyword + semantic search across ALL documents.
   Example: {"tool": "search", "args": {"query": "feathery shifter karma cost"}}

2. list_sources()
   See which documents exist in the library with chunk counts.
   Example: {"tool": "list_sources", "args": {}}

3. filter_source(filename, query, top_k=8)
   Search within ONE specific document. Use this once you know which book has the content.
   Example: {"tool": "filter_source", "args": {"filename": "E-CAT28801S_Bestial_Nature.pdf", "query": "feathery"}}

4. read_page(chunk_id)
   Fetch the FULL text of a specific chunk (not just the preview). Use this to zoom in on the most promising hit.
   Example: {"tool": "read_page", "args": {"chunk_id": "E-CAT28801S_Bestial_Nature.pdf::p13::s0"}}

5. done(answer_ready)
   Declare you have enough content. Exits the loop.
   Example: {"tool": "done", "args": {"answer_ready": true}}

STRATEGY:
- If the question mentions a specific game system, book, or topic, consider list_sources first to find the right book.
- Start with a specific search that names the entity.
- If a preview looks promising (has the keywords you need), call read_page for the full content.
- Call done when you've read at least one full page with the answer.
- Max 5 iterations. Be efficient.

OUTPUT FORMAT: A single JSON object on one line. No explanation. No markdown fences. No prose around it.
"""


class DeepSearchAgent:
    """Agentic RAG: LLM drives retrieval via tool calls."""

    MAX_ITERATIONS = 5
    MAX_JSON_RETRIES = 2

    def __init__(self, llm, retriever):
        self._llm = llm
        self._retriever = retriever

    def run(self, query: str) -> tuple[str, list[dict]]:
        """Run the agent loop for a query.

        Returns:
            (formatted_context, agent_trace)
            formatted_context: string ready for injection into factual RAG prompt.
            agent_trace: list of dicts, one per iteration, with
                         {tool, args, result_summary, raw_llm_output}.
        """
        history: list[dict] = []  # agent trace
        # chunk_id → {source, page, text} for chunks the agent has fetched
        read_chunks: dict[str, dict] = {}
        # chunk_id → preview (for search results we've seen but not read)
        seen_previews: dict[str, dict] = {}

        for iteration in range(self.MAX_ITERATIONS):
            # Build the prompt with history so far
            prompt = self._build_prompt(query, history)

            # Ask LLM what to do next
            raw = self._call_llm(prompt)
            tool_call = self._parse_tool_call(raw)

            if not tool_call:
                log.warning(f"[Agent] Iter {iteration + 1}: JSON parse failed")
                # Retry once with an explicit correction prompt
                raw2 = self._call_llm(
                    prompt + "\n\nYour last response was not valid JSON. "
                    "Return ONLY a single JSON object like: "
                    '{"tool": "done", "args": {"answer_ready": true}}')
                tool_call = self._parse_tool_call(raw2)
                if not tool_call:
                    log.warning(f"[Agent] Iter {iteration + 1}: "
                              f"gave up after 2 JSON failures")
                    break

            tool = tool_call.get("tool", "").lower()
            args = tool_call.get("args", {}) or {}

            log.info(f"[Agent] Iter {iteration + 1}: tool={tool} args={args}")

            # Execute the tool
            if tool == "done":
                history.append({
                    "tool": "done", "args": args,
                    "result_summary": "Agent declared done.",
                    "raw_llm_output": raw,
                })
                break
            elif tool == "search":
                q = args.get("query", "")
                top_k = int(args.get("top_k", 8))
                result = self._tool_search(q, top_k)
                for item in result:
                    seen_previews[item["chunk_id"]] = item
                summary = self._summarize_search_result(result)
            elif tool == "filter_source":
                fn = args.get("filename", "")
                q = args.get("query", "")
                top_k = int(args.get("top_k", 8))
                result = self._tool_filter_source(fn, q, top_k)
                for item in result:
                    seen_previews[item["chunk_id"]] = item
                summary = self._summarize_search_result(result)
            elif tool == "list_sources":
                result = self._tool_list_sources()
                summary = self._summarize_sources(result)
            elif tool == "read_page":
                cid = args.get("chunk_id", "")
                result = self._tool_read_page(cid)
                if result:
                    read_chunks[cid] = result
                    summary = (f"Full text of {cid} "
                               f"({len(result.get('text', ''))} chars) "
                               f"appended to accumulated chunks.")
                else:
                    summary = f"Chunk {cid} not found."
            else:
                summary = f"Unknown tool: {tool}"

            history.append({
                "tool": tool, "args": args,
                "result_summary": summary,
                "raw_llm_output": raw,
            })

        # Format accumulated chunks for final RAG injection
        formatted = self._format_final_context(read_chunks, seen_previews)
        return formatted, history

    # ── Tool implementations ──────────────────────────────────────────────

    def _tool_search(self, query: str, top_k: int = 8) -> list[dict]:
        return self._retriever.get_raw_chunks(
            query, top_k=top_k, apply_reranker=True)

    def _tool_filter_source(self, filename: str, query: str,
                            top_k: int = 8) -> list[dict]:
        return self._retriever.get_raw_chunks(
            query, top_k=top_k, source_filter=filename,
            apply_reranker=True)

    def _tool_list_sources(self) -> list[dict]:
        return self._retriever.list_sources()

    def _tool_read_page(self, chunk_id: str) -> dict | None:
        return self._retriever.get_chunk_by_id(chunk_id)

    # ── LLM interaction ───────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            return self._llm.generate(
                prompt,
                system_prompt=_AGENT_SYSTEM_PROMPT,
                max_length=200,
                temperature=0.1,
            )
        except Exception as e:
            log.error(f"[Agent] LLM call failed: {e}")
            return ""

    @staticmethod
    def _parse_tool_call(raw: str) -> dict | None:
        """Extract a single JSON object from the LLM response."""
        if not raw:
            return None
        text = raw.strip()
        # Strip markdown code fences
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        # Find the first { ... } block
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group())
            if not isinstance(parsed, dict):
                return None
            return parsed
        except (json.JSONDecodeError, ValueError):
            return None

    # ── Prompt building ──────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(query: str, history: list[dict]) -> str:
        parts = [f"User question: {query}", ""]

        if history:
            parts.append("Your previous tool calls this session:")
            for i, entry in enumerate(history, 1):
                tool = entry["tool"]
                args = entry.get("args", {})
                summary = entry.get("result_summary", "")
                parts.append(f"  {i}. {tool}({args}) -> {summary}")
            parts.append("")

        parts.append("What tool would you like to call next? "
                     "Respond with a single JSON object.")
        return "\n".join(parts)

    # ── Result summarization for history ──────────────────────────────────

    @staticmethod
    def _summarize_search_result(results: list[dict]) -> str:
        if not results:
            return "No results."
        lines = [f"{len(results)} chunk(s) found:"]
        for r in results[:5]:
            lines.append(
                f"    {r['chunk_id']} (kw={r.get('kw_score', 0)}) "
                f"— {r.get('preview', '')[:120]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summarize_sources(sources: list[dict]) -> str:
        if not sources:
            return "No sources found."
        lines = [f"{len(sources)} document(s) in library:"]
        for s in sources[:20]:
            lines.append(f"    {s['filename']} ({s['chunks']} chunks)")
        return "\n".join(lines)

    # ── Final context formatting ──────────────────────────────────────────

    @staticmethod
    def _format_final_context(
        read_chunks: dict[str, dict],
        seen_previews: dict[str, dict],
    ) -> str:
        """Build the formatted context string for the final LLM answer.

        Prefers full-text chunks (from read_page) over previews, but
        includes previews for chunks the agent saw but didn't fully read,
        so the answer generator has broader context.
        """
        if not read_chunks and not seen_previews:
            return ""

        lines = [
            "DOCUMENT EXCERPTS (source of truth — report ALL details found):"
        ]

        # Full-text read chunks first (the agent chose these)
        for chunk_id, chunk in read_chunks.items():
            source = chunk.get("source", "?")
            page = chunk.get("page")
            src = f"{source} (page {page})" if page else source
            lines.append(f"- From {src}: {chunk.get('text', '')}")

        # Previews for chunks the agent saw but didn't read in full
        # (limit to avoid bloat)
        preview_only = [
            (cid, c) for cid, c in seen_previews.items()
            if cid not in read_chunks
        ]
        if preview_only and len(read_chunks) < 4:
            # Only include if we don't already have enough full chunks
            for cid, c in preview_only[:4]:
                source = c.get("source", "?")
                page = c.get("page")
                src = f"{source} (page {page})" if page else source
                lines.append(f"- (preview) From {src}: {c.get('preview', '')}")

        return "\n".join(lines) + "\n"
