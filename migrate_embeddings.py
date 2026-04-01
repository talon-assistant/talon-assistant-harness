#!/usr/bin/env python3
"""Re-embed all ChromaDB collections with the upgraded BGE embedding model.

Run this ONCE after upgrading from all-MiniLM-L6-v2 to BAAI/bge-base-en-v1.5
(or any other embedding model change).  Without migration, vector queries
return garbage results because the stored embeddings and query embeddings are
in different vector spaces.

Usage:
    python migrate_embeddings.py

The script reads config/settings.json to determine the target model and
chroma_path.  All five collections are migrated in-place:
  - talon_memory
  - talon_documents
  - talon_notes
  - talon_rules
  - talon_corrections

Documents, metadatas, and IDs are preserved; only the embedding vectors
are replaced.
"""
import json
import sys
from pathlib import Path

import logging
log = logging.getLogger(__name__)


SETTINGS_PATH = Path("config/settings.json")
BATCH = 500   # documents per ChromaDB batch


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    # Fall back to example config for defaults only
    ex = Path("config/settings.example.json")
    if ex.exists():
        with open(ex) as f:
            return json.load(f)
    return {}


def migrate():
    settings = load_settings()
    mem_cfg = settings.get("memory", {})
    chroma_path = mem_cfg.get("chroma_path", "data/chroma_db")
    embed_model = mem_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")

    log.info("=" * 60)
    log.info("Talon embedding migration")
    log.info("=" * 60)
    log.info(f"  chroma_path   : {chroma_path}")
    log.info(f"  embed_model   : {embed_model}")
    log.debug("")

    import chromadb
    from core.embeddings import embed_documents

    client = chromadb.PersistentClient(path=chroma_path)

    # Collection name → recreation metadata
    collection_defs = [
        ("talon_memory",      {"description": "Talon conversation and preference memory"}),
        ("talon_documents",   {"description": "User documents for RAG retrieval"}),
        ("talon_notes",       {"description": "User notes for semantic search"}),
        ("talon_rules",       {"description": "Behavioral rules: trigger phrase semantic matching"}),
        ("talon_corrections", {"hnsw:space": "cosine",
                               "description": "Correction memory: maps bad commands to correct intent"}),
    ]

    for coll_name, coll_meta in collection_defs:
        log.info(f"── {coll_name} ──")

        try:
            old_coll = client.get_collection(coll_name)
        except Exception:
            log.warning("Not found — skipping.\n")
            continue

        total = old_coll.count()
        if total == 0:
            log.info("Empty — nothing to migrate.\n")
            continue

        log.info(f"Fetching {total} document(s)...")
        all_docs, all_ids, all_metas = [], [], []
        offset = 0
        while offset < total:
            batch = old_coll.get(
                limit=BATCH,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_docs.extend(batch["documents"])
            all_ids.extend(batch["ids"])
            all_metas.extend(batch["metadatas"])
            offset += BATCH

        log.info(f"Re-embedding {len(all_docs)} document(s) with {embed_model}...")
        new_embeddings = embed_documents(all_docs, embed_model)

        log.info("Replacing collection...")
        client.delete_collection(coll_name)
        new_coll = client.get_or_create_collection(
            name=coll_name,
            metadata=coll_meta,
        )

        log.info("Inserting with new embeddings...")
        for i in range(0, len(all_docs), BATCH):
            new_coll.add(
                embeddings=new_embeddings[i:i + BATCH],
                documents=all_docs[i:i + BATCH],
                metadatas=all_metas[i:i + BATCH],
                ids=all_ids[i:i + BATCH],
            )
            done = min(i + BATCH, len(all_docs))
            log.info(f"{done}/{len(all_docs)}")

        log.info(f"Done.\n")

    log.info("Migration complete — all collections re-embedded.")
    log.info("You can now start Talon normally.")


if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    migrate()
