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
   See which documents exist in the library with chunk counts. Each entry
   notes whether the book has a parsed table of contents (has_toc=true)
   that you can navigate via lookup_in_toc.
   Example: {"tool": "list_sources", "args": {}}

3. filter_source(filename, query, top_k=8)
   Search within ONE specific document. Use this once you know which book has the content.
   Example: {"tool": "filter_source", "args": {"filename": "E-CAT28801S_Bestial_Nature.pdf", "query": "feathery"}}

4. lookup_in_toc(filename, query)
   Look the term up in a book's table of contents and get the page(s) the
   author tagged for it. This is how a human reader navigates a reference
   book — flip to the right page, don't search every page. ONLY works for
   books where list_sources says has_toc=true.
   Example: {"tool": "lookup_in_toc", "args": {"filename": "CAT28008_Wild_Life.pdf", "query": "Pegasus"}}

5. read_page(chunk_id)
   Fetch the FULL text of a specific chunk (not just the preview). Use this to zoom in on the most promising hit.
   Example: {"tool": "read_page", "args": {"chunk_id": "E-CAT28801S_Bestial_Nature.pdf::p13::s0"}}

6. read_nearby(chunk_id, before=N, after=N)
   Read pages adjacent to a known location. Use when you've found a
   chunk that contains your keyword but you suspect the description
   continues on the next or previous page (spell tables, multi-page
   stat blocks, sections that span pages). Like flipping forward/back
   in a book from a known anchor. Default before=1, after=1.
   Example: {"tool": "read_nearby", "args": {"chunk_id": "Berlin.pdf::p134::s0", "after": 1}}

7. read_section(chunk_id)
   Read the entire TOC section containing `chunk_id`. Use for topical
   questions where the answer spans multiple pages of one section
   (e.g. "explain how Combat Spells work", "what's covered under
   Adept Powers"). Section boundaries come from the book's TOC;
   capped at 8 pages joined for context budget.
   Example: {"tool": "read_section", "args": {"chunk_id": "Berlin.pdf::p134::s0"}}

8. next_section(chunk_id)
   Jump to the start of the section AFTER `chunk_id`'s section.
   Use for "what comes after X" questions and for forward navigation
   through structured docs.
   Example: {"tool": "next_section", "args": {"chunk_id": "Berlin.pdf::p134::s0"}}

10. done(answer_ready)
    Declare you have enough content. Exits the loop.
   Example: {"tool": "done", "args": {"answer_ready": true}}

STRATEGY (IMPORTANT — read carefully and pick the right branch):

A) USER NAMES A SPECIFIC BOOK (e.g. "in Shadowrun Berlin Edition", "in
   Schneier's Applied Cryptography"):
   1. Call list_sources to find the exact filename and check has_toc.
   2. If has_toc is true → try lookup_in_toc FIRST. The TOC gives you
      the page the author chose to put the content on. Read that page
      directly. This is faster and more accurate than search.
   3. If lookup_in_toc returns nothing useful, use filter_source on
      that book — NEVER use plain search when you already know which
      book to look in.

B) USER DOES NOT NAME A SPECIFIC BOOK (e.g. "what is a Shadowrun mana
   bolt", "what is steganography"):
   1. Call search FIRST (global, across all books). The chunk_ids in
      the result reveal which book(s) have content for the query.
      Pick the book with the highest kw_score (keyword match).
   2. THEN filter_source or lookup_in_toc on that book to drill in.
   3. Do NOT guess a book from list_sources without seeing search
      results — your guess will often be wrong (e.g. picking a
      critter book for a spell-rules question).

C) ALWAYS, REGARDLESS OF BRANCH:
   - When calling search/filter_source, query for the SUBJECT only —
     do not include book/series names in the query string. Book
     filtering is handled by filter_source's filename argument.
     GOOD: filter_source(filename="Berlin.pdf", query="mana bolt")
     BAD: search(query="mana bolt shadowrun berlin edition")
   - After reading a page, evaluate. Stats/numbers/rules → call done.
     Overview only → search for the missing detail or lookup_in_toc
     with a more specific term.
   - Do NOT blindly read every result. Read → evaluate → decide.

D) PICKING THE RIGHT READ TOOL:
   - read_page: specific entity/term ("what are Manabolt's stats?")
   - read_nearby: spillover suspected ("description continues to next page")
   - read_section: topical question covering a whole section
     ("explain how Combat Spells work", "what's covered under
     Adept Powers")
   - next_section: forward navigation ("what comes after X?")

RULES:
- Call done as soon as you have specific stats/numbers/rules that answer
  the question. Don't waste iterations reading marginal chunks.
- If a search returns a page you've already read, skip it — search differently.
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
            elif tool == "lookup_in_toc":
                fn = args.get("filename", "")
                q = args.get("query", "")
                result = self._tool_lookup_in_toc(fn, q)
                # Track these like search previews so the agent doesn't
                # re-fetch a page it has only seen the TOC stub for.
                for item in result:
                    page_disp = (item["page_pdf"] + 1
                                 if item.get("page_pdf") is not None else None)
                    seen_previews[item["chunk_id"]] = {
                        "chunk_id": item["chunk_id"],
                        "source": item["filename"],
                        "page": page_disp,
                        "chapter": item.get("chapter_idx"),
                        "preview": f"[TOC: {item['title']}]",
                    }
                summary = self._summarize_toc_result(result, fn, q)
            elif tool == "read_page":
                cid = args.get("chunk_id", "")
                # read_page returns the entire page/chapter (all sub-chunks
                # joined), so any sibling sub_chunk on the same page is a
                # duplicate request. Compare by the page/chapter prefix
                # ("filename::pN" or "filename::chN") rather than the full id.
                page_prefix = cid.rsplit("::s", 1)[0] if "::s" in cid else cid
                already_read_prefixes = {
                    k.rsplit("::s", 1)[0] if "::s" in k else k
                    for k in read_chunks
                }
                if page_prefix in already_read_prefixes:
                    summary = (f"Already read {page_prefix} in a previous "
                               f"iteration (read_page returns the whole "
                               f"page). Pick a different page/chapter or "
                               f"call done if you have enough information.")
                else:
                    result = self._tool_read_page(cid)
                    if result:
                        # Index auto-follow: when the chunk is a back-of-book
                        # index, look up the user's query in it and fetch the
                        # referenced page so the agent gets actual content
                        # rather than just a page reference.
                        from core.document_index import detect_index_chunk
                        followup_note = ""
                        is_index_page = detect_index_chunk(result.get("text", ""))
                        if is_index_page:
                            filename = cid.split("::")[0]
                            followups = self._retriever.follow_index_reference(
                                filename, query, result["text"])
                            for fu in followups:
                                # Append followed page text to the result
                                result["text"] += (
                                    f"\n\n[AUTO-FOLLOWED INDEX REFERENCE → "
                                    f"page {fu['printed_page']} "
                                    f"(chunk {fu['chunk_id']})]\n"
                                    f"{fu['text']}"
                                )
                                # Mark followed chunk as read too
                                read_chunks[fu['chunk_id']] = {
                                    "chunk_id": fu['chunk_id'],
                                    "source": filename,
                                    "page": fu['pdf_idx'] + 1,
                                    "text": fu['text'],
                                }
                            if followups:
                                pages_joined = ", ".join(
                                    str(f['printed_page']) for f in followups)
                                followup_note = (
                                    f" Index detected — auto-followed to "
                                    f"page(s) {pages_joined}.")
                        elif self._keyword_near_end_of_chunk(
                                result.get("text", ""), query):
                            # Auto-extend on near-miss: the user's keyword
                            # sits in the last 400 chars of the chunk, which
                            # usually means the description continues onto
                            # the next page (multi-page spell entries, etc).
                            # Fetch +1 and append it like a turned page.
                            extended = self._retriever.get_neighboring_chunks(
                                cid, before=0, after=1)
                            if extended and extended.get("spans", 0) > 1:
                                # Strip the leading "[• page N]" marker and
                                # the original page's content; keep just the
                                # next-page section so we don't duplicate.
                                ext_text = extended.get("text", "")
                                next_marker_idx = ext_text.find("[→ page")
                                if next_marker_idx >= 0:
                                    extra = ext_text[next_marker_idx:]
                                    result["text"] += (
                                        f"\n\n[AUTO-EXTENDED — keyword near "
                                        f"page boundary, continuing]\n{extra}"
                                    )
                                    pages = extended.get("pages_included") or []
                                    if len(pages) >= 2:
                                        followup_note = (
                                            f" Auto-extended to page "
                                            f"{pages[-1]}.")

                        read_chunks[cid] = result
                        joined = result.get("sub_chunks_joined", 1)
                        joined_note = (f", joined {joined} sub-chunks"
                                       if joined > 1 else "")
                        summary = (f"Full text of {cid} "
                                   f"({len(result.get('text', ''))} chars"
                                   f"{joined_note}) "
                                   f"appended to accumulated chunks.{followup_note} "
                                   f"You now have {len(read_chunks)} full "
                                   f"page(s). If this is enough to answer, "
                                   f"call done.")
                    else:
                        summary = f"Chunk {cid} not found."
            elif tool == "read_section":
                cid = args.get("chunk_id", "")
                result = self._tool_read_section(cid)
                if result:
                    read_chunks[cid] = result
                    title = result.get("section_title", "?")
                    pages = result.get("pages_included") or []
                    page_range = (
                        f"pages {pages[0]}-{pages[-1]}"
                        if len(pages) > 1
                        else (f"page {pages[0]}" if pages else "?")
                    )
                    next_title = result.get("next_section_title")
                    next_note = (f" Next section is '{next_title}'."
                                 if next_title else "")
                    summary = (
                        f"Read section '{title}' covering {page_range} "
                        f"({len(result.get('text', ''))} chars). "
                        f"You now have {len(read_chunks)} full page(s)."
                        f"{next_note}"
                    )
                else:
                    summary = (f"Could not resolve a section for {cid}. "
                               f"The chunk may be outside the parsed TOC "
                               f"range.")
            elif tool == "next_section":
                cid = args.get("chunk_id", "")
                result = self._tool_next_section(cid)
                if result:
                    next_cid = result["chunk_id"]
                    read_chunks[next_cid] = result
                    title = result.get("section_title", "?")
                    summary = (
                        f"Jumped to next section '{title}' (chunk_id: "
                        f"{next_cid}). {len(result.get('text', ''))} chars "
                        f"of content. You now have {len(read_chunks)} "
                        f"full page(s)."
                    )
                else:
                    summary = (f"No next section found for {cid}. May be "
                               f"the last section in the book.")
            elif tool == "read_nearby":
                cid = args.get("chunk_id", "")
                before = int(args.get("before", 1))
                after = int(args.get("after", 1))
                # Same dedup logic as read_page — page-prefix based
                page_prefix = cid.rsplit("::s", 1)[0] if "::s" in cid else cid
                result = self._tool_read_nearby(cid, before=before, after=after)
                if result:
                    read_chunks[cid] = result
                    pages = result.get("pages_included") or []
                    chapters = result.get("chapters_included") or []
                    span_label = (
                        f"pages {pages[0]}-{pages[-1]}" if len(pages) > 1
                        else (f"chapters {chapters[0]}-{chapters[-1]}"
                              if len(chapters) > 1 else page_prefix)
                    )
                    summary = (
                        f"Read {result.get('spans', 0)} adjacent chunk(s) "
                        f"({len(result.get('text', ''))} chars covering "
                        f"{span_label}) appended to accumulated chunks. "
                        f"You now have {len(read_chunks)} full page(s)."
                    )
                else:
                    summary = f"Could not resolve {cid} for read_nearby."
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
    def _keyword_near_end_of_chunk(text: str, query: str,
                                   window: int = 400) -> bool:
        """Is a query keyword sitting in the last `window` chars of `text`?

        When a relevant heading or term appears at the end of an extracted
        chunk, the description usually continues onto the next page. This
        triggers the auto-extend behavior so the agent doesn't have to
        realize "the answer was cut off" and ask for the next page itself.
        """
        if not text or not query:
            return False
        _STOP = {"what", "when", "where", "which", "who", "how", "why",
                 "the", "a", "an", "is", "are", "was", "were",
                 "of", "in", "on", "at", "to", "for", "with",
                 "and", "or", "but", "as", "if",
                 "tell", "about", "explain", "describe", "show"}
        words = re.findall(r"[a-zA-Z][\w'-]*", query.lower())
        keywords = [w for w in words if len(w) >= 4 and w not in _STOP]
        if not keywords:
            return False
        # Also include compound concat for 2-3 word queries
        if 2 <= len(keywords) <= 3:
            keywords.append("".join(keywords))
        end = text[-window:].lower()
        return any(kw in end for kw in keywords)

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

    def _tool_lookup_in_toc(self, filename: str, query: str) -> list[dict]:
        if not filename or not query:
            return []
        return self._retriever.lookup_in_toc(filename, query, max_results=8)

    def _tool_read_page(self, chunk_id: str) -> dict | None:
        return self._retriever.get_chunk_by_id(chunk_id)

    def _tool_read_nearby(self, chunk_id: str,
                          before: int = 1, after: int = 1) -> dict | None:
        return self._retriever.get_neighboring_chunks(
            chunk_id, before=before, after=after)

    def _tool_read_section(self, chunk_id: str) -> dict | None:
        return self._retriever.read_section(chunk_id)

    def _tool_next_section(self, chunk_id: str) -> dict | None:
        return self._retriever.next_section(chunk_id)

    # ── LLM interaction ───────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            return self._llm.generate(
                prompt,
                system_prompt=_AGENT_SYSTEM_PROMPT,
                max_length=200,
                temperature=0.1,
                detect_degeneration=False,  # JSON tool call — don't truncate
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
        # Always surface every has_toc book, plus a sample of the rest.
        toc = [s for s in sources if s.get("has_toc")]
        rest = [s for s in sources if not s.get("has_toc")]
        lines = [f"{len(sources)} document(s) in library "
                 f"({len(toc)} with parsed TOC):"]
        for s in toc:
            lines.append(
                f"    {s['filename']} ({s['chunks']} chunks) [has_toc]")
        # Cap the rest so the prompt doesn't bloat
        for s in rest[:15]:
            lines.append(f"    {s['filename']} ({s['chunks']} chunks)")
        if len(rest) > 15:
            lines.append(f"    ... and {len(rest) - 15} more without TOC")
        return "\n".join(lines)

    @staticmethod
    def _summarize_toc_result(results: list[dict], filename: str,
                              query: str) -> str:
        if not results:
            return (f"No TOC entries match '{query}' in {filename}. "
                    f"Try a different term or fall back to search/filter_source.")
        lines = [f"{len(results)} TOC match(es) in {filename}:"]
        for r in results[:8]:
            if r.get("chapter_idx") is not None:
                loc = f"chapter {r['chapter_idx']}"
            elif r.get("page_printed"):
                loc = f"page {r['page_printed']}"
            else:
                loc = "?"
            lines.append(
                f"    \"{r['title']}\" → {loc} "
                f"(chunk_id: {r['chunk_id']})"
            )
        lines.append("    → Pick the most relevant entry and call read_page "
                     "with its chunk_id.")
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

        def _src_label(c: dict) -> str:
            source = c.get("source", "?")
            page = c.get("page")
            chapter = c.get("chapter")
            if page:
                return f"{source} (page {page})"
            if chapter is not None:
                return f"{source} (chapter {chapter})"
            return source

        # Full-text read chunks first (the agent chose these)
        for chunk_id, chunk in read_chunks.items():
            lines.append(f"- From {_src_label(chunk)}: {chunk.get('text', '')}")

        # Previews for chunks the agent saw but didn't read in full.
        # Filter to only those from the SAME source(s) the agent actually
        # read. Previews from other books cause cross-source pollution —
        # the LLM treats them as authoritative and fabricates synthesized
        # answers attributing claims to books the agent never opened.
        read_sources = {c.get("source") for c in read_chunks.values()
                        if c.get("source")}
        preview_only = [
            (cid, c) for cid, c in seen_previews.items()
            if cid not in read_chunks
            and (not read_sources or c.get("source") in read_sources)
        ]
        if preview_only and len(read_chunks) < 4:
            # Only include if we don't already have enough full chunks
            for cid, c in preview_only[:4]:
                lines.append(
                    f"- (preview) From {_src_label(c)}: {c.get('preview', '')}")

        return "\n".join(lines) + "\n"
