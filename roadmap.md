# Talon Assistant — Roadmap

## Recently Completed
- Vision-enhanced PDF ingestion (`--vision` flag, one chunk per page via Qwen2.5-VL)
- Multi-query RAG retrieval with alt_queries expansion
- Text-match `$contains` fallback for sparse stat-block chunks
- Keyword re-ranking, 8-chunk hard cap, page number citations in responses
- Training-knowledge contamination prevention in explicit RAG mode
- `_extract_arg()` helper on BaseTalent — standardised single-value LLM extraction

## Queued
- Session reflection / auto-distill — Talon summarises and learns from conversation patterns over time
- Self-writing talents — Talon generates new talent plugins from a user description
- RAG: corpus-frequency stop words for `$contains` — skip generic terms like "damage" or "shadowrun" that match too broadly across the corpus
