"""
Ingestion Component
Handles document loading, parsing, and chunking for the RAG pipeline.
Supports: PDF, DOCX, HTML, TXT, Markdown, CSV, JSON, URLs
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    page_number: Optional[int] = None
    chunk_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseLoader:
    """Base class for document loaders."""

    def load(self, source: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PDFLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        try:
            import pypdf
            docs = []
            with open(source, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append({
                            "content": text,
                            "metadata": {"page": i + 1, "source": source, "format": "pdf"},
                        })
            return docs
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf")


class DocxLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        try:
            from docx import Document
            doc = Document(source)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [{"content": full_text, "metadata": {"source": source, "format": "docx"}}]
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")


class HTMLLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
            with open(source, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return [{"content": text, "metadata": {"source": source, "format": "html"}}]
        except ImportError:
            raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")


class TextLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        fmt = Path(source).suffix.lstrip(".")
        return [{"content": content, "metadata": {"source": source, "format": fmt}}]


class CSVLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        import csv
        docs = []
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                docs.append({
                    "content": content,
                    "metadata": {"source": source, "format": "csv", "row": i + 1, **row},
                })
        return docs


class JSONLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            docs = []
            for i, item in enumerate(data):
                content = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                docs.append({"content": content, "metadata": {"source": source, "format": "json", "index": i}})
            return docs
        return [{"content": json.dumps(data, indent=2), "metadata": {"source": source, "format": "json"}}]


class URLLoader(BaseLoader):
    def load(self, source: str) -> List[Dict[str, Any]]:
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove nav, footer, scripts
            for tag in soup(["nav", "footer", "script", "style", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return [{"content": text, "metadata": {"source": source, "format": "url", "url": source}}]
        except ImportError:
            raise ImportError("requests and beautifulsoup4 required: pip install requests beautifulsoup4")


LOADER_MAP = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".doc": DocxLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".jsonl": JSONLoader,
    "url": URLLoader,
}


class TextSplitter:
    """Splits documents into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, strategy: str = "recursive"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def split(self, text: str) -> List[str]:
        if self.strategy == "recursive":
            return self._recursive_split(text)
        elif self.strategy == "character":
            return self._character_split(text)
        elif self.strategy == "sentence":
            return self._sentence_split(text)
        else:
            return self._recursive_split(text)

    def _recursive_split(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        self._recursive(text, separators, chunks)
        return chunks

    def _recursive(self, text: str, separators: List[str], chunks: List[str]):
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]
        parts = text.split(sep) if sep else list(text)
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) > self.chunk_size and remaining_seps:
                        self._recursive(current, remaining_seps, chunks)
                    else:
                        chunks.append(current.strip())
                    # Handle overlap
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + (sep if sep else "") + part
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())

    def _character_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current = current + " " + sent if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks


class IngestionComponent:
    """Main ingestion orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.splitter_strategy = config.get("splitter", "recursive")
        self.splitter = TextSplitter(self.chunk_size, self.chunk_overlap, self.splitter_strategy)

    def _get_loader(self, source: str) -> BaseLoader:
        if source.startswith("http://") or source.startswith("https://"):
            return URLLoader()
        ext = Path(source).suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if not loader_cls:
            logger.warning(f"No loader for extension {ext}, defaulting to TextLoader")
            return TextLoader()
        return loader_cls()

    def _make_chunk_id(self, content: str, source: str, index: int) -> str:
        h = hashlib.md5(f"{source}:{index}:{content[:100]}".encode()).hexdigest()[:12]
        return f"chunk_{h}"

    def ingest(self, sources: List[str]) -> List[DocumentChunk]:
        """Ingest documents from a list of file paths or URLs."""
        all_chunks: List[DocumentChunk] = []

        for source in sources:
            logger.info(f"Loading source: {source}")
            try:
                loader = self._get_loader(source)
                raw_docs = loader.load(source)

                for doc in raw_docs:
                    content = doc["content"]
                    metadata = doc.get("metadata", {})
                    page = metadata.get("page")
                    text_chunks = self.splitter.split(content)

                    for i, chunk_text in enumerate(text_chunks):
                        chunk_id = self._make_chunk_id(chunk_text, source, len(all_chunks))
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            metadata=metadata,
                            source=source,
                            page_number=page,
                            chunk_index=i,
                        )
                        all_chunks.append(chunk)

                logger.info(f"Source {source}: created {len(text_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                continue

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def ingest_to_json(self, sources: List[str], output_path: str) -> str:
        """Ingest and save chunks to JSON file (for pipeline artifact passing)."""
        chunks = self.ingest(sources)
        output = [c.to_dict() for c in chunks]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(output)} chunks to {output_path}")
        return output_path


def main():
    """CLI entrypoint for the ingestion component."""
    import argparse
    parser = argparse.ArgumentParser(description="RAG Ingestion Component")
    parser.add_argument("--sources", nargs="+", required=True, help="File paths or URLs to ingest")
    parser.add_argument("--output-path", required=True, help="Output JSON path for chunks")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--splitter", default="recursive", choices=["recursive", "character", "sentence"])
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        config = full_cfg.get("ingestion", {})

    config.update({
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "splitter": args.splitter,
    })

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    component = IngestionComponent(config)
    component.ingest_to_json(args.sources, args.output_path)


if __name__ == "__main__":
    main()
