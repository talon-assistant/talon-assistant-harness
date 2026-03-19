"""manage_documents.py — ChromaDB document collection management.

Usage:
    python manage_documents.py list              # List all ingested documents
    python manage_documents.py remove <name>     # Remove a specific document by filename
    python manage_documents.py info <name>       # Show chunk count and metadata for a doc
    python manage_documents.py clear             # Remove ALL documents (with confirmation)
"""

import argparse
import json
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


def cmd_list(collection):
    """List all ingested documents with chunk counts."""
    result = collection.get(include=["metadatas"])
    if not result["ids"]:
        print("No documents ingested.")
        return

    # Aggregate by filename
    counts: dict[str, int] = {}
    types: dict[str, set] = {}
    for meta in result["metadatas"]:
        fname = meta.get("filename", "<unknown>")
        counts[fname] = counts.get(fname, 0) + 1
        chunk_type = meta.get("type", "")
        types.setdefault(fname, set()).add(chunk_type)

    print(f"\n{'Document':<60} {'Chunks':>6}  Types")
    print("-" * 80)
    for fname in sorted(counts):
        type_str = ", ".join(sorted(types[fname]))
        print(f"{fname:<60} {counts[fname]:>6}  {type_str}")
    print(f"\nTotal: {len(counts)} document(s), {len(result['ids'])} chunk(s)")


def cmd_info(collection, name: str):
    """Show detailed chunk info for a specific document."""
    result = collection.get(
        where={"filename": name},
        include=["metadatas", "documents"]
    )
    if not result["ids"]:
        # Try partial match
        all_result = collection.get(include=["metadatas"])
        matches = set(
            m.get("filename", "") for m in all_result["metadatas"]
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


def cmd_remove(collection, name: str):
    """Remove all chunks for a specific document."""
    result = collection.get(
        where={"filename": name},
        include=[]
    )
    if not result["ids"]:
        # Try partial match
        all_result = collection.get(include=["metadatas"])
        matches = set(
            m.get("filename", "") for m in all_result["metadatas"]
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
    result = collection.get(include=[])
    count = len(result["ids"])
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

    collection.delete(ids=result["ids"])
    print(f"✓ Cleared {count} chunk(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Talon's ChromaDB document collection.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all ingested documents")

    p_info = sub.add_parser("info", help="Show chunks for a specific document")
    p_info.add_argument("name", help="Filename of the document")

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
    elif args.command == "remove":
        cmd_remove(collection, args.name)
    elif args.command == "clear":
        cmd_clear(collection)


if __name__ == "__main__":
    main()
