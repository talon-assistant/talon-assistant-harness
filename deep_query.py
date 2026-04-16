"""deep_query.py — Agentic RAG tester.

Runs a query through DeepSearchAgent and dumps every iteration:
the LLM's raw output, the tool call it chose, the tool result,
and the final formatted context.

Usage:
    python deep_query.py "in shadowrun what is a feathery shifter"
    python deep_query.py --max-iter 3 "your query"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)


def _load_config() -> dict:
    for name in ("config/settings.json", "config/settings.example.json"):
        p = Path(name)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _c(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    codes = {
        "bold": "\033[1m", "dim": "\033[2m",
        "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    }
    return f"{codes.get(color, '')}{text}\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The query string")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Max agent iterations (default: 5)")
    args = parser.parse_args()

    query = args.query
    cfg = _load_config()
    mem_cfg = cfg.get("memory", {})
    chroma_path = mem_cfg.get("chroma_path", "data/chroma_db")
    embed_model = mem_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
    reranker_model = mem_cfg.get(
        "reranker_model", "BAAI/bge-reranker-base")

    print(_c("=" * 78, "dim"))
    print(_c(f"DEEP SEARCH: {query}", "bold"))
    print(_c(f"ChromaDB: {chroma_path}", "dim"))
    print(_c("=" * 78, "dim"))
    print()

    import chromadb
    from core.document_retriever import DocumentRetriever
    from core.deep_search_agent import DeepSearchAgent
    from core.llm_client import LLMClient

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        docs = client.get_collection("talon_documents")
    except Exception as e:
        print(_c(f"ERROR: {e}", "red"))
        sys.exit(1)

    llm = LLMClient(cfg)
    retriever = DocumentRetriever(docs, embed_model, reranker_model)
    agent = DeepSearchAgent(llm, retriever)
    agent.MAX_ITERATIONS = args.max_iter

    print(_c("━━━ Running agent loop ━━━", "bold"))
    print()
    context, trace = agent.run(query)

    # Dump trace
    for i, entry in enumerate(trace, 1):
        print(_c(f"─── Iteration {i} ───", "cyan"))
        print(_c(f"  Tool:    {entry['tool']}", "bold"))
        print(f"  Args:    {entry.get('args', {})}")
        print()
        print(_c("  Raw LLM output:", "dim"))
        raw = entry.get("raw_llm_output", "")
        for line in raw.split("\n")[:8]:
            print(f"    {line}")
        print()
        print(_c("  Result:", "dim"))
        for line in entry.get("result_summary", "").split("\n")[:12]:
            print(f"    {line}")
        print()

    # Final context
    print(_c("━━━ Final formatted context (for RAG injection) ━━━", "bold"))
    if context:
        print(context)
    else:
        print(_c("(empty)", "dim"))
    print()


if __name__ == "__main__":
    main()
