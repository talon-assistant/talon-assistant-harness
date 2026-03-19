"""manage_documents.py — ChromaDB document collection management.

Usage:
    python manage_documents.py list                        # List all ingested documents
    python manage_documents.py remove <name>               # Remove a specific document by filename
    python manage_documents.py info <name>                 # Show chunk count and metadata for a doc
    python manage_documents.py sample <name>               # Print N random full chunks (default 3)
    python manage_documents.py sample <name> -n 5          # Print 5 random chunks
    python manage_documents.py sample <name> -n 5 --seq    # Print first 5 chunks in order
    python manage_documents.py sample <name> --chunk <id>  # Print one specific chunk by ID
    python manage_documents.py clear                       # Remove ALL documents (with confirmation)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import chromadb


def _load_config():
    config_path = Path("config/settings.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        chroma_path = config.get("memory", {}).get("chroma_path", "data/chroma_db")
    else:
        chroma_path = "data/chroma_db"
    return chroma_path


def _get_collection(chroma_path: str):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name="talon_documents",
        metadata={"description": "User documents for RAG retrieval"}
    )


def _iter_all(collection, include: list[str], batch_size: int = 500):
    """Paginate through the entire collection in batches.

    Yields dicts with keys matching ``include`` plus "ids".
    Avoids SQLite 'too many variables' on large collections.
    """
    offset = 0
    while True:
        result = collection.get(include=include, limit=batch_size, offset=offset)
        batch_ids = result.get("ids", [])
        if not batch_ids:
            break
        yield result
        offset += len(batch_ids)
        if len(batch_ids) < batch_size:
            break


def _all_metadatas(collection) -> tuple[list[str], list[dict]]:
    """Return (ids, metadatas) for the entire collection, paginated."""
    all_ids: list[str] = []
    all_metas: list[dict] = []
    for batch in _iter_all(collection, include=["metadatas"]):
        all_ids.extend(batch["ids"])
        all_metas.extend(batch["metadatas"])
    return all_ids, all_metas


def cmd_list(collection):
    """List all ingested documents with chunk counts."""
    all_ids, all_metas = _all_metadatas(collection)
    if not all_ids:
        print("No documents ingested.")
        return

    # Aggregate by filename
    counts: dict[str, int] = {}
    types: dict[str, set] = {}
    for meta in all_metas:
        fname = meta.get("filename", "<unknown>")
        counts[fname] = counts.get(fname, 0) + 1
        chunk_type = meta.get("type", "")
        types.setdefault(fname, set()).add(chunk_type)

    print(f"\n{'Document':<60} {'Chunks':>6}  Types")
    print("-" * 80)
    for fname in sorted(counts):
        type_str = ", ".join(sorted(types[fname]))
        print(f"{fname:<60} {counts[fname]:>6}  {type_str}")
    print(f"\nTotal: {len(counts)} document(s), {len(all_ids)} chunk(s)")


def cmd_info(collection, name: str):
    """Show detailed chunk info for a specific document."""
    result = collection.get(
        where={"filename": name},
        include=["metadatas", "documents"]
    )
    if not result["ids"]:
        # Try partial match
        _, all_metas = _all_metadatas(collection)
        matches = set(
            m.get("filename", "") for m in all_metas
            if name.lower() in m.get("filename", "").lower()
        )
        if matches:
            print(f"No exact match for '{name}'. Did you mean:")
            for m in sorted(matches):
                print(f"  {m}")
        else:
            print(f"No document found matching '{name}'.")
        return

    print(f"\n{name} — {len(result['ids'])} chunk(s)\n")
    for i, (doc_id, meta, doc) in enumerate(
            zip(result["ids"], result["metadatas"], result["documents"]), 1):
        preview = doc[:80].replace("\n", " ") + ("..." if len(doc) > 80 else "")
        chunk_type = meta.get("type", "")
        chapter = meta.get("chapter", meta.get("page_number", ""))
        print(f"  [{i:>3}] {doc_id}")
        print(f"        type={chunk_type}  chapter/page={chapter}")
        print(f"        {preview}\n")


def cmd_sample(collection, name: str, n: int = 3, sequential: bool = False,
               chunk_id: str = None):
    """Print full raw text of N chunks from a document."""
    # Single chunk by ID
    if chunk_id:
        result = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        if not result["ids"]:
            print(f"No chunk found with id '{chunk_id}'.")
            return
        _print_chunk(1, result["ids"][0], result["metadatas"][0], result["documents"][0])
        return

    result = collection.get(
        where={"filename": name},
        include=["metadatas", "documents"]
    )
    if not result["ids"]:
        # Try partial match suggestion
        _, all_metas = _all_metadatas(collection)
        matches = set(
            m.get("filename", "") for m in all_metas
            if name.lower() in m.get("filename", "").lower()
        )
        if matches:
            print(f"No exact match for '{name}'. Did you mean:")
            for m in sorted(matches):
                print(f"  {m}")
        else:
            print(f"No document found matching '{name}'.")
        return

    total = len(result["ids"])
    ids = result["ids"]
    metas = result["metadatas"]
    docs = result["documents"]

    if sequential:
        indices = list(range(min(n, total)))
    else:
        indices = random.sample(range(total), min(n, total))
        indices.sort()  # keep them in document order even when random

    print(f"\n{name} — showing {len(indices)} of {total} chunk(s)"
          f" ({'first' if sequential else 'random'})\n")
    for rank, idx in enumerate(indices, 1):
        _print_chunk(rank, ids[idx], metas[idx], docs[idx])


def _print_chunk(rank: int, chunk_id: str, meta: dict, doc: str):
    """Pretty-print a single chunk with its full text."""
    chunk_type = meta.get("type", "")
    chapter = meta.get("chapter", meta.get("page_number", ""))
    char_count = len(doc)
    sep = "─" * 72
    print(f"{sep}")
    print(f"  Chunk #{rank}  │  id: {chunk_id}")
    print(f"  type={chunk_type}  chapter/page={chapter}  ({char_count} chars)")
    print(sep)
    print(doc)
    print()


def cmd_remove(collection, name: str):
    """Remove all chunks for a specific document."""
    result = collection.get(
        where={"filename": name},
        include=[]
    )
    if not result["ids"]:
        # Try partial match
        _, all_metas = _all_metadatas(collection)
        matches = set(
            m.get("filename", "") for m in all_metas
            if name.lower() in m.get("filename", "").lower()
        )
        if matches:
            print(f"No exact match for '{name}'. Did you mean:")
            for m in sorted(matches):
                print(f"  {m}")
            print("\nUse the exact filename shown above.")
        else:
            print(f"No document found matching '{name}'.")
        return

    count = len(result["ids"])
    confirm = input(f"Remove {count} chunk(s) for '{name}'? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    collection.delete(ids=result["ids"])
    print(f"✓ Removed {count} chunk(s) for '{name}'.")


def cmd_clear(collection):
    """Remove ALL documents from the collection."""
    # Count via paginated fetch to avoid SQLite variable limit
    all_ids: list[str] = []
    for batch in _iter_all(collection, include=[]):
        all_ids.extend(batch["ids"])

    count = len(all_ids)
    if count == 0:
        print("Collection is already empty.")
        return

    confirm = input(
        f"This will permanently delete ALL {count} chunk(s) from talon_documents.\n"
        f"Type 'yes' to confirm: "
    ).strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    # Delete in batches to avoid the same SQLite variable limit on delete
    batch_size = 500
    for i in range(0, count, batch_size):
        collection.delete(ids=all_ids[i:i + batch_size])
    print(f"✓ Cleared {count} chunk(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Talon's ChromaDB document collection.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all ingested documents")

    p_info = sub.add_parser("info", help="Show chunks for a specific document")
    p_info.add_argument("name", help="Filename of the document")

    p_sample = sub.add_parser("sample", help="Print full raw text of sampled chunks")
    p_sample.add_argument("name", nargs="?", default="",
                          help="Filename of the document (omit when using --chunk)")
    p_sample.add_argument("-n", type=int, default=3,
                          help="Number of chunks to show (default: 3)")
    p_sample.add_argument("--seq", action="store_true",
                          help="Show first N chunks in order instead of random")
    p_sample.add_argument("--chunk", metavar="ID", default=None,
                          help="Show one specific chunk by its ChromaDB ID")

    p_remove = sub.add_parser("remove", help="Remove a specific document")
    p_remove.add_argument("name", help="Filename of the document to remove")

    sub.add_parser("clear", help="Remove ALL documents (requires confirmation)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    chroma_path = _load_config()
    collection = _get_collection(chroma_path)

    if args.command == "list":
        cmd_list(collection)
    elif args.command == "info":
        cmd_info(collection, args.name)
    elif args.command == "sample":
        cmd_sample(collection, args.name, n=args.n,
                   sequential=args.seq, chunk_id=args.chunk)
    elif args.command == "remove":
        cmd_remove(collection, args.name)
    elif args.command == "clear":
        cmd_clear(collection)


if __name__ == "__main__":
    main()
