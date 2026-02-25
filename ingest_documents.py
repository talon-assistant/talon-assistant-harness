import os
import json
from pathlib import Path
import chromadb
import pymupdf4llm
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
import markdown
from datetime import datetime


class DocumentIngester:
    """Ingests documents into ChromaDB for RAG"""

    def __init__(self, documents_dir=None, chroma_path=None):
        # Try to load paths from config
        config_path = Path("config/settings.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            documents_dir = documents_dir or config.get("documents", {}).get("directory", "documents")
            chroma_path = chroma_path or config.get("memory", {}).get("chroma_path", "data/chroma_db")
        else:
            documents_dir = documents_dir or "documents"
            chroma_path = chroma_path or "data/chroma_db"

        self.documents_dir = Path(documents_dir)
        self.chroma_path = chroma_path

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="talon_documents",
            metadata={"description": "User documents for RAG retrieval"}
        )

        self.supported_extensions = {
            '.pdf', '.txt', '.docx', '.md', '.markdown',
            '.py', '.js', '.java', '.cpp', '.cs', '.html', '.htm',
            '.csv', '.xlsx', '.xls'
        }

        print(f"✓ Document ingester ready!")
        print(f"  Documents directory: {self.documents_dir}")
        print(f"  Supported types: {', '.join(self.supported_extensions)}\n")

    def chunk_text(self, text, chunk_size=400, overlap=50):
        """Split text into overlapping chunks.

        Default chunk_size raised to 400 words (from 200) to keep cross-
        referenced content (e.g. Mana Bolt described under Manaball) within
        the same chunk, improving RAG retrieval on structured reference books.
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def extract_pdf(self, filepath):
        """Extract text from PDF.

        Primary: pymupdf4llm.to_markdown() — preserves reading order in
        multi-column layouts, tables, and headings.

        Fallback: raw fitz page-by-page extraction — used when pymupdf4llm
        returns suspiciously little content (some PDFs with complex layouts
        cause pymupdf4llm to silently drop most pages).
        """
        import fitz  # PyMuPDF — bundled with pymupdf4llm

        # Primary: pymupdf4llm
        try:
            result = pymupdf4llm.to_markdown(str(filepath))
            doc = fitz.open(str(filepath))
            page_count = doc.page_count
            doc.close()
            # Sanity check: ~100 chars per page minimum
            if result and len(result.strip()) >= page_count * 100:
                return result
            print(f"    ⚠ pymupdf4llm returned only {len(result or '')} chars "
                  f"for {page_count} pages — falling back to raw fitz extraction")
        except Exception as e:
            print(f"    ⚠ pymupdf4llm failed ({e}) — falling back to raw fitz")

        # Fallback: raw fitz page-by-page
        try:
            doc = fitz.open(str(filepath))
            pages = []
            for page in doc:
                text = page.get_text(flags=fitz.TEXT_PRESERVE_WHITESPACE)
                if text.strip():
                    pages.append(text)
            doc.close()
            full_text = "\n\n".join(pages)
            print(f"    → fitz fallback: extracted {len(full_text)} chars "
                  f"from {len(pages)} pages")
            return full_text if full_text.strip() else None
        except Exception as e:
            print(f"  ✗ fitz fallback also failed: {e}")
            return None

    def extract_docx(self, filepath):
        """Extract text from DOCX"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"  ✗ Error reading DOCX: {e}")
            return None

    def extract_txt(self, filepath):
        """Extract text from TXT/code files"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"  ✗ Error reading text file: {e}")
            return None

    def extract_markdown(self, filepath):
        """Extract text from Markdown"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                md_text = file.read()
                html = markdown.markdown(md_text)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            print(f"  ✗ Error reading Markdown: {e}")
            return None

    def extract_csv_excel(self, filepath):
        """Extract data from CSV/Excel"""
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            text = f"File: {filepath.name}\n\n"
            text += f"Columns: {', '.join(df.columns)}\n\n"
            text += df.to_string()
            return text
        except Exception as e:
            print(f"  ✗ Error reading data file: {e}")
            return None

    def extract_html(self, filepath):
        """Extract text from HTML"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        except Exception as e:
            print(f"  ✗ Error reading HTML: {e}")
            return None

    def extract_text(self, filepath):
        """Route to appropriate extractor"""
        ext = filepath.suffix.lower()

        if ext == '.pdf':
            return self.extract_pdf(filepath)
        elif ext == '.docx':
            return self.extract_docx(filepath)
        elif ext in ['.txt', '.py', '.js', '.java', '.cpp', '.cs']:
            return self.extract_txt(filepath)
        elif ext in ['.md', '.markdown']:
            return self.extract_markdown(filepath)
        elif ext in ['.csv', '.xlsx', '.xls']:
            return self.extract_csv_excel(filepath)
        elif ext in ['.html', '.htm']:
            return self.extract_html(filepath)
        else:
            return None

    def ingest_file(self, filepath, chunk_size=400, overlap=50):
        """Ingest a single file"""
        print(f"  Processing: {filepath.name}")

        text = self.extract_text(filepath)
        if not text or len(text.strip()) < 50:
            print(f"    ⚠ Skipped (no content or too short)")
            return 0

        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"    → Created {len(chunks)} chunks ({chunk_size}w/{overlap}w overlap)")

        doc_id_base = f"doc_{filepath.stem}_{int(datetime.now().timestamp())}"

        for i, chunk in enumerate(chunks):
            doc_id = f"{doc_id_base}_chunk_{i}"

            self.collection.add(
                documents=[chunk],
                metadatas=[{
                    "type": "document",
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "file_type": filepath.suffix,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "ingested_at": datetime.now().isoformat()
                }],
                ids=[doc_id]
            )

        print(f"    ✓ Ingested {len(chunks)} chunks")
        return len(chunks)

    def ingest_directory(self, chunk_size=400, overlap=50):
        """Ingest all supported files"""
        if not self.documents_dir.exists():
            print(f"Creating documents directory: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✓ Created! Add your documents to: {self.documents_dir}")
            return

        files = []
        for ext in self.supported_extensions:
            files.extend(self.documents_dir.rglob(f"*{ext}"))

        if not files:
            print(f"No documents found in {self.documents_dir}")
            return

        print(f"Found {len(files)} files to ingest\n")

        total_chunks = 0
        successful = 0

        for filepath in files:
            try:
                chunks = self.ingest_file(filepath, chunk_size=chunk_size, overlap=overlap)
                total_chunks += chunks
                successful += 1
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        print(f"\n{'=' * 50}")
        print(f"✓ Ingestion complete!")
        print(f"  Files: {successful}/{len(files)}")
        print(f"  Chunks: {total_chunks}")
        print(f"{'=' * 50}\n")

    def list_documents(self):
        """List all ingested documents"""
        results = self.collection.get()

        if not results['ids']:
            print("No documents in database")
            return

        files = {}
        for metadata in results['metadatas']:
            filename = metadata.get('filename', 'unknown')
            if filename not in files:
                files[filename] = {
                    'type': metadata.get('file_type', 'unknown'),
                    'chunks': 0,
                    'ingested_at': metadata.get('ingested_at', 'unknown')
                }
            files[filename]['chunks'] += 1

        print(f"\nDocuments in database: {len(files)}\n")

        for filename, info in sorted(files.items()):
            print(f"  {filename}")
            print(f"    Type: {info['type']}, Chunks: {info['chunks']}")

    def clear_documents(self):
        """Clear all documents"""
        self.client.delete_collection(name="talon_documents")
        self.collection = self.client.get_or_create_collection(
            name="talon_documents",
            metadata={"description": "User documents for RAG retrieval"}
        )
        print("✓ Cleared all documents")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Talon document ingester")
    parser.add_argument("command", nargs="?", default="ingest",
                        choices=["ingest", "list", "clear"],
                        help="Command to run (default: ingest)")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Words per chunk (default: 400)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap words between chunks (default: 50)")
    args = parser.parse_args()

    ingester = DocumentIngester()

    if args.command == "list":
        ingester.list_documents()
    elif args.command == "clear":
        confirm = input("Clear all documents? (yes/no): ")
        if confirm.lower() == "yes":
            ingester.clear_documents()
    else:
        print(f"Chunk size: {args.chunk_size} words, overlap: {args.overlap} words")
        ingester.ingest_directory(chunk_size=args.chunk_size, overlap=args.overlap)