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

STRATEGY (IMPORTANT — follow this order):
1. Start with a specific search that names the entity.
2. Look at the previews. Pick the ONE most promising chunk and read it with read_page.
3. AFTER reading that page, evaluate what you have. If you have stats,
   numbers, requirements, and powers → call done. If the page was general
   overview → try a more specific search (e.g. "feathery karma cost",
   "natural weapon beak", "gained powers").
4. If the specific search turns up a different page, read THAT page.
5. Do NOT blindly read every search result. Read → evaluate → decide
   whether to read more, search differently, or declare done.

RULES:
- Call done as soon as you have specific stats/numbers/rules that answer
  the question. Don't waste iterations reading marginal chunks.
- If a search returns a preview page you've already read, skip it —
  search for something different.
- If list_sources helps identify a specific book for the topic, use it
  BEFORE wasting searches on the whole collection.
- Max 7 iterations. Be efficient.

OUTPUT FORMAT: A single JSON object on one line. No explanation. No markdown fences. No prose around it.
"""


class DeepSearchAgent:
    """Agentic RAG: LLM drives retrieval via tool calls."""

    MAX_ITERATIONS = 7
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
                if cid in read_chunks:
                    # Already read — don't waste an iteration re-reading.
                    summary = (f"Already read {cid} in a previous iteration. "
                               f"Pick a different chunk or call done if you "
                               f"have enough information.")
                else:
                    result = self._tool_read_page(cid)
                    if result:
                        read_chunks[cid] = result
                        summary = (f"Full text of {cid} "
                                   f"({len(result.get('text', ''))} chars) "
                                   f"appended to accumulated chunks. "
                                   f"You now have {len(read_chunks)} full "
                                   f"page(s). If this is enough to answer, "
                                   f"call done.")
                    else:
                        summary = f"Chunk {cid} not found."
            else:
                summary = f"Unknown tool: {tool}"

            history.append({
                "tool": tool, "args": args,
                "result_summary": summary,
                "raw_llm_output": raw,
            })

        # Filter out chunks that don't share keywords with the query.
        # The agent sometimes reads tangentially-related chunks (e.g. Greasy
        # Skin from Changelings while searching for feathery shifter) — those
        # pollute the final context and make the LLM hedge or wander.
        filtered_read = self._filter_by_keywords(query, read_chunks)
        if len(filtered_read) < len(read_chunks):
            dropped = len(read_chunks) - len(filtered_read)
            log.info(f"[Agent] Filtered out {dropped} chunk(s) with no "
                     f"keyword overlap (likely tangential)")

        # Format accumulated chunks for final RAG injection
        formatted = self._format_final_context(filtered_read, seen_previews)
        return formatted, history

    @staticmethod
    def _filter_by_keywords(query: str, chunks: dict[str, dict]) -> dict[str, dict]:
        """Remove chunks with zero keyword overlap with the query.

        Keeps chunks with at least one query keyword. Always keeps at least
        one chunk so the final context is never empty, even if the agent
        went completely off-topic.
        """
        _STOPWORDS = {
            "what", "when", "where", "which", "who", "whom", "whose",
            "that", "this", "these", "those", "there", "here",
            "have", "has", "had", "been", "being",
            "will", "would", "could", "should",
            "about", "with", "from", "into",
            "provide", "detail", "explain", "describe", "please",
            "shadowrun",
        }
        keywords = {
            w.lower() for w in query.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        }
        if not keywords:
            return chunks

        def score(text: str) -> int:
            lower = text.lower()
            return sum(1 for kw in keywords if kw in lower)

        scored = [(cid, c, score(c.get("text", "")))
                  for cid, c in chunks.items()]
        relevant = {cid: c for cid, c, s in scored if s > 0}

        # Safety: if filtering removed everything, keep the highest-scoring
        # chunk so we don't return empty context.
        if not relevant and scored:
            scored.sort(key=lambda t: -t[2])
            cid, c, _ = scored[0]
            return {cid: c}
        return relevant

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
                detect_degeneration=False,  # JSON tool call — don't truncate
                no_think=True,  # agent emits JSON directly, no reasoning
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
