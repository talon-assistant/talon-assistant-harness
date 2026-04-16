"""rquery.py — RAG retrieval pipeline debugger.

Runs a query through the same pipeline Talon uses and dumps every stage
in detail so you can see what the reranker actually scored, which chunks
got dropped, and why source concentration picked the document it did.

Usage:
    python rquery.py "in shadowrun what is a feathery shifter"
    python rquery.py --no-rerank "your query"
    python rquery.py --pool 40 "your query"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Silence noisy loggers
import logging
logging.basicConfig(level=logging.WARNING)


def _load_config() -> dict:
    """Load settings.json or settings.example.json."""
    for name in ("config/settings.json", "config/settings.example.json"):
        p = Path(name)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _c(text: str, color: str) -> str:
    """ANSI color wrapper. Disabled if output is redirected."""
    if not sys.stdout.isatty():
        return text
    codes = {
        "bold": "\033[1m", "dim": "\033[2m",
        "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    }
    return f"{codes.get(color, '')}{text}\033[0m"


def _short(text: str, n: int = 80) -> str:
    """Show first N chars of a chunk, one line, escape newlines."""
    t = text.replace("\n", " ").replace("\r", " ")
    return t[:n].rstrip() + ("..." if len(t) > n else "")


def _keyword_score(chunk_text: str, keywords: set[str]) -> int:
    lower = chunk_text.lower()
    return sum(1 for kw in keywords if kw in lower)


# Stopwords matching document_retriever.py
_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose",
    "that", "this", "these", "those", "there", "here",
    "have", "has", "had", "been", "being", "were", "was",
    "will", "would", "could", "should", "shall", "might",
    "does", "did", "done", "doing", "make", "made",
    "give", "gave", "given", "take", "took", "taken",
    "tell", "told", "about", "with", "from", "into",
    "much", "many", "more", "most", "some", "such",
    "also", "just", "than", "then", "only", "very",
    "like", "well", "even", "back", "over", "after",
    "each", "every", "other", "both", "same", "know",
    "provide", "detail", "details", "detailed", "explain",
    "describe", "please", "possible", "list",
    "show", "find", "help", "need", "want", "look",
    "information", "info",
    "shadowrun",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The query string to test")
    parser.add_argument("--pool", type=int, default=20,
                        help="Reranker pool size (default: 20)")
    parser.add_argument("--final", type=int, default=8,
                        help="Final chunk cap (default: 8)")
    parser.add_argument("--n-semantic", type=int, default=12,
                        help="Chunks per semantic query (default: 12)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip reranker (show semantic ranking only)")
    parser.add_argument("--alt-queries", nargs="*", default=None,
                        help="Alt queries (skip LLM expansion)")
    parser.add_argument("--no-alt", action="store_true",
                        help="Skip alt query expansion entirely")
    parser.add_argument("--chunk-chars", type=int, default=100,
                        help="Preview chars per chunk (default: 100)")
    parser.add_argument("--kw-boost", type=float, default=0.0,
                        help="Add (kw_hits/n_kw)*BOOST to rerank scores "
                             "before final sort. Try 0.2 to see effect. "
                             "Default 0 (off).")
    parser.add_argument("--contains-sticky", action="store_true",
                        help="Guarantee all $contains chunks make it into "
                             "the reranker pool (expands pool if needed).")
    args = parser.parse_args()

    query = args.query
    cfg = _load_config()
    mem_cfg = cfg.get("memory", {})
    chroma_path = mem_cfg.get("chroma_path", "data/chroma_db")
    embed_model = mem_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
    reranker_model = mem_cfg.get(
        "reranker_model", "BAAI/bge-reranker-base")

    print(_c("=" * 78, "dim"))
    print(_c(f"QUERY: {query}", "bold"))
    print(_c(f"ChromaDB: {chroma_path}", "dim"))
    print(_c(f"Embed:    {embed_model}", "dim"))
    print(_c(f"Reranker: {reranker_model}", "dim"))
    print(_c("=" * 78, "dim"))
    print()

    import chromadb
    from core import embeddings as _emb
    from core import reranker as _reranker

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        docs = client.get_collection("talon_documents")
    except Exception as e:
        print(_c(f"ERROR: {e}", "red"))
        sys.exit(1)

    total = docs.count()
    print(f"Total chunks in collection: {total:,}")
    print()

    # Keywords
    raw_tokens = {w.lower() for w in query.split() if len(w) > 3}
    keywords = {w for w in raw_tokens if w not in _STOPWORDS}
    print(_c("[Keywords after stopword filter]", "cyan"),
          sorted(keywords))
    print()

    # ── PHASE 1: Semantic retrieval (primary query) ──
    print(_c("━━━ PHASE 1: Semantic retrieval (primary query) ━━━", "bold"))
    qvec = _emb.embed_queries([query], embed_model)
    results = docs.query(
        query_embeddings=qvec,
        n_results=args.n_semantic,
        include=["documents", "metadatas", "distances"],
    )
    semantic_chunks = []  # (filename, text, dist, page)
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        fn = meta.get("filename", "?")
        pg = meta.get("page_number")
        semantic_chunks.append((fn, doc, dist, pg))

    for i, (fn, txt, dist, pg) in enumerate(semantic_chunks, 1):
        kw = _keyword_score(txt, keywords)
        pg_s = f"p{pg + 1}" if pg is not None else "-"
        print(f"  {i:>2}. dist={dist:.3f} kw={kw}  "
              f"{fn[:45]:<45} {pg_s:>5}  {_short(txt, args.chunk_chars)}")
    print()

    # ── PHASE 2: Alt queries ──
    alt_queries = args.alt_queries
    if args.no_alt:
        alt_queries = []
    elif alt_queries is None:
        # Generate via LLM if available
        print(_c("━━━ PHASE 2: Alt query expansion (via LLM) ━━━", "bold"))
        try:
            from core.llm_client import LLMClient
            import re
            llm_cfg = cfg.get("llm", {})
            if not llm_cfg:
                raise RuntimeError("No LLM config in settings")
            client_llm = LLMClient(cfg)
            raw = client_llm.generate(
                f"Generate 3 search queries to find relevant document chunks. "
                f"Rules: (1) 4-8 words each. "
                f"(2) If the request mentions game entities, spells, characters, or "
                f"stat blocks, at least one query must name the specific entity and "
                f"include words like 'powers', 'attributes', 'statistics', or 'type'. "
                f"(3) Include synonyms and related terms. "
                f"Return a JSON array of 3 strings, nothing else.\n\n"
                f"Request: {query}\nQueries:",
                max_length=80,
                temperature=0.0,
            )
            raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
            alt_queries = json.loads(raw)
            print(f"  LLM generated: {alt_queries}")
        except Exception as e:
            print(_c(f"  LLM unavailable ({e}) — skipping alt queries", "yellow"))
            alt_queries = []
    else:
        print(_c("━━━ PHASE 2: Alt queries (user-provided) ━━━", "bold"))
        print(f"  {alt_queries}")

    alt_pool = []  # (filename, text, dist, page)
    seen = {t[:100] for _, t, _, _ in semantic_chunks}
    for aq in alt_queries or []:
        if len(aq.strip()) < 4:
            continue
        aqvec = _emb.embed_queries([aq], embed_model)
        r = docs.query(
            query_embeddings=aqvec,
            n_results=args.n_semantic,
            include=["documents", "metadatas", "distances"],
        )
        added = 0
        for doc, meta, dist in zip(
            r["documents"][0], r["metadatas"][0], r["distances"][0]
        ):
            key = doc[:100]
            if key not in seen:
                seen.add(key)
                alt_pool.append((
                    meta.get("filename", "?"), doc, dist,
                    meta.get("page_number"),
                ))
                added += 1
        print(f"  + {aq!r}: +{added} new chunks")
    print(f"  Alt-query unique additions: {len(alt_pool)}")
    print()

    # ── PHASE 3: $contains keyword fallback ──
    print(_c("━━━ PHASE 3: $contains keyword fallback ━━━", "bold"))
    all_terms = [query] + (alt_queries or [])
    text_kws = sorted(
        {w for t in all_terms for w in t.split() if len(w) > 3},
        key=len, reverse=True,
    )[:8]
    print(f"  Search terms: {text_kws}")
    contains_pool = []
    contains_seen = set()
    for kw in text_kws:
        for variant in {kw, kw.title(), kw.capitalize()}:
            try:
                hits = docs.get(
                    where_document={"$contains": variant},
                    limit=6,
                    include=["documents", "metadatas"],
                )
                added = 0
                for doc, meta in zip(
                    hits.get("documents", []),
                    hits.get("metadatas", []),
                ):
                    key = doc[:100]
                    if key in contains_seen:
                        continue
                    contains_seen.add(key)
                    contains_pool.append((
                        meta.get("filename", "?"), doc, 1.0,
                        meta.get("page_number"),
                    ))
                    added += 1
                if added:
                    print(f"  $contains \"{variant}\": +{added} chunks")
            except Exception as e:
                print(_c(f"  $contains {variant}: error: {e}", "yellow"))
    print(f"  Total $contains unique: {len(contains_pool)}")
    print()

    # ── PHASE 4: Candidate pool summary ──
    print(_c("━━━ PHASE 4: Combined candidate pool ━━━", "bold"))
    # Track which keys came from $contains so we can mark them sticky
    contains_keys = {c[1][:100] for c in contains_pool}

    all_candidates = semantic_chunks + alt_pool + contains_pool
    # Dedup by text prefix
    seen_all = set()
    dedup_candidates = []
    for c in all_candidates:
        key = c[1][:100]
        if key not in seen_all:
            seen_all.add(key)
            dedup_candidates.append(c)
    print(f"  Semantic (primary):    {len(semantic_chunks)}")
    print(f"  Alt queries:           {len(alt_pool)}")
    print(f"  $contains keyword:     {len(contains_pool)}")
    print(f"  Unique after dedup:    {len(dedup_candidates)}")

    # Source breakdown in candidate pool
    source_counts = Counter(c[0] for c in dedup_candidates)
    print()
    print(_c("  Candidate pool by source:", "cyan"))
    for fn, n in source_counts.most_common():
        # Does this source contain our exact keywords?
        kw_chunks = [c for c in dedup_candidates
                     if c[0] == fn and _keyword_score(c[1], keywords) > 0]
        print(f"    {n:>3}x  {fn[:60]}   (with kw hits: {len(kw_chunks)})")
    print()

    # Pool selection: by distance ascending, top-K.
    # With --contains-sticky: guarantee all $contains chunks make the pool,
    # expanding the pool if needed.
    dedup_candidates.sort(key=lambda c: c[2])

    if args.contains_sticky:
        sticky = [c for c in dedup_candidates
                  if c[1][:100] in contains_keys]
        non_sticky = [c for c in dedup_candidates
                      if c[1][:100] not in contains_keys]
        # Keep all sticky chunks plus fill remaining pool slots with
        # top-distance non-sticky chunks.
        remaining = max(0, args.pool - len(sticky))
        rerank_pool = sticky + non_sticky[:remaining]
        print(_c(f"  [contains-sticky active]", "magenta"))
        print(f"  Sticky $contains chunks guaranteed: {len(sticky)}")
        print(f"  Non-sticky fill slots:              {len(non_sticky[:remaining])}")
        print(f"  Final reranker pool:                {len(rerank_pool)}")
    else:
        rerank_pool = dedup_candidates[:args.pool]
        print(f"  Sending top {len(rerank_pool)} (by distance) to reranker")
    print()

    # ── PHASE 5: Reranker ──
    if not args.no_rerank and len(rerank_pool) > 1:
        print(_c("━━━ PHASE 5: Cross-encoder reranker (ALL scores) ━━━", "bold"))
        model = _reranker._get_model(reranker_model)
        pairs = [(query, c[1]) for c in rerank_pool]
        scores = model.predict(pairs).tolist()

        # Optional keyword boost (AND-like reranking)
        n_kw = max(1, len(keywords))
        boost = args.kw_boost
        if boost > 0:
            print(_c(f"  [kw-boost active: +(kw/{n_kw})*{boost:.2f} per chunk]",
                     "magenta"))
            adjusted = []
            for s, c in zip(scores, rerank_pool):
                kw = _keyword_score(c[1], keywords)
                bonus = (kw / n_kw) * boost
                adjusted.append((s + bonus, s, bonus, c))
            adjusted.sort(key=lambda x: x[0], reverse=True)
        else:
            adjusted = [(s, s, 0.0, c)
                        for s, c in zip(scores, rerank_pool)]
            adjusted.sort(key=lambda x: x[0], reverse=True)

        if boost > 0:
            print(f"  {'rank':>4} {'final':>8} {'rerank':>8} {'+bonus':>7}  "
                  f"{'kw':>3}  {'source':<40} {'pg':>5}  preview")
            print(f"  {'-' * 4} {'-' * 8} {'-' * 8} {'-' * 7}  "
                  f"{'-' * 3}  {'-' * 40} {'-' * 5}  {'-' * 40}")
        else:
            print(f"  {'rank':>4} {'score':>8}  {'kw':>3}  "
                  f"{'source':<45} {'pg':>5}  preview")
            print(f"  {'-' * 4} {'-' * 8}  {'-' * 3}  "
                  f"{'-' * 45} {'-' * 5}  {'-' * 40}")
        min_score = -1.0
        cutoff_shown = False
        for rank, (final_s, raw_s, bonus_s, c) in enumerate(adjusted, 1):
            fn, txt, dist, pg = c
            kw = _keyword_score(txt, keywords)
            pg_s = f"p{pg + 1}" if pg is not None else "-"

            if rank == args.final + 1:
                print(_c(f"  --- above: top {args.final} kept --- "
                         f"below: dropped ---", "yellow"))
            if raw_s < min_score and not cutoff_shown:
                print(_c(f"  --- below reranker min_score={min_score} "
                         f"(would be dropped without kw boost) ---", "red"))
                cutoff_shown = True

            color = "green" if raw_s >= min_score and rank <= args.final else (
                "yellow" if raw_s >= min_score else "red")
            if boost > 0:
                line = (f"  {rank:>4} {final_s:>+8.3f} {raw_s:>+8.3f} "
                        f"{bonus_s:>+7.3f}  {kw:>3}  "
                        f"{fn[:40]:<40} {pg_s:>5}  "
                        f"{_short(txt, args.chunk_chars)}")
            else:
                line = (f"  {rank:>4} {raw_s:>+8.3f}  {kw:>3}  "
                        f"{fn[:45]:<45} {pg_s:>5}  "
                        f"{_short(txt, args.chunk_chars)}")
            print(_c(line, color))
        print()

        # Apply min_score against the RAW reranker score, not boosted.
        # Keep top N by final (boosted) score among those that passed.
        passing = [(final_s, c) for final_s, raw_s, _, c in adjusted
                   if raw_s >= min_score]
        final_chunks = [c for _s, c in passing[:args.final]]
    else:
        print(_c("━━━ PHASE 5: Skipped reranker ━━━", "bold"))
        final_chunks = rerank_pool[:args.final]
        print()

    # ── PHASE 6: Source concentration analysis ──
    print(_c("━━━ PHASE 6: Source concentration ━━━", "bold"))
    source_kw_totals: dict[str, int] = defaultdict(int)
    source_max_kw: dict[str, int] = defaultdict(int)
    source_chunk_counts: dict[str, int] = defaultdict(int)
    for fn, txt, _, _ in final_chunks:
        kw = _keyword_score(txt, keywords)
        source_kw_totals[fn] += kw
        source_chunk_counts[fn] += 1
        if kw > source_max_kw[fn]:
            source_max_kw[fn] = kw

    print(f"  {'source':<55} {'chunks':>6} {'tot_kw':>7} {'max_kw':>7}")
    for fn in sorted(source_kw_totals,
                     key=lambda k: (-source_kw_totals[k], k)):
        print(f"  {fn[:55]:<55} {source_chunk_counts[fn]:>6} "
              f"{source_kw_totals[fn]:>7} {source_max_kw[fn]:>7}")

    # What would current logic pick?
    if source_kw_totals:
        top_by_total = max(
            source_kw_totals,
            key=lambda fn: (source_kw_totals[fn], source_chunk_counts[fn]))
        top_by_max = max(
            source_max_kw, key=lambda fn: source_max_kw[fn])
        print()
        print(_c(f"  Current logic (total kw): would pick {top_by_total}",
                 "cyan"))
        print(_c(f"  Alternative (max kw):     would pick {top_by_max}",
                 "cyan"))
    print()

    # ── PHASE 7: Final output ──
    print(_c("━━━ PHASE 7: Final chunks ━━━", "bold"))
    for i, (fn, txt, dist, pg) in enumerate(final_chunks, 1):
        kw = _keyword_score(txt, keywords)
        pg_s = f"p{pg + 1}" if pg is not None else "-"
        print(f"  {i}. kw={kw}  {fn[:50]:<50} {pg_s:>5}")
        print(f"     {_short(txt, 200)}")
    print()


if __name__ == "__main__":
    main()
