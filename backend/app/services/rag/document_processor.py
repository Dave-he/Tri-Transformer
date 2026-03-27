import io
from typing import Optional

import fitz


class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process_file(self, file_content: bytes, filename: str) -> dict:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "pdf":
            text = self._extract_pdf(file_content)
            fmt = "pdf"
        elif ext in ("md", "markdown", "txt"):
            text = file_content.decode("utf-8", errors="replace")
            fmt = "markdown" if ext in ("md", "markdown") else "text"
        elif ext in ("docx",):
            text = self._extract_docx(file_content)
            fmt = "docx"
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return {"text": text, "format": fmt, "filename": filename}

    async def process_text(self, content: str, filename: str = "") -> dict:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "text"
        fmt = "markdown" if ext in ("md", "markdown") else "text"
        return {"text": content, "format": fmt, "filename": filename}

    async def chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += step
        return chunks

    def _extract_pdf(self, content: bytes) -> str:
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)

    def _extract_docx(self, content: bytes) -> str:
        try:
            import docx

            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return content.decode("utf-8", errors="replace")

    @staticmethod
    def supported_extensions() -> set[str]:
        return {"pdf", "md", "markdown", "txt", "docx"}
