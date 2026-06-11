"""PDF document loader (requires PyPDF2)."""

from __future__ import annotations

from pathlib import Path

from ..errors import DependencyError, LoaderError, ValidationError
from ..models import Document
from .base import BaseLoader


class PDFLoader(BaseLoader):
    """Load documents from PDF files.

    Requires: ``pip install rag-bridge-kit[pdf]``
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pages: tuple[int, ...] | None = None,
        one_doc_per_page: bool = False,
    ) -> None:
        self.path = Path(path)
        self.pages = pages
        self.one_doc_per_page = one_doc_per_page

    def load(self) -> list[Document]:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "PyPDF2 is required for PDFLoader. Install with: pip install rag-bridge-kit[pdf]"
            ) from exc

        if not self.path.exists():
            raise ValidationError(f"PDF file does not exist: {self.path}")

        if self.path.is_dir():
            documents: list[Document] = []
            pdf_files = sorted(self.path.rglob("*.pdf"))
            if not pdf_files:
                raise LoaderError(f"No PDF files found in {self.path}")
            for pdf_path in pdf_files:
                documents.extend(self._load_pdf(pdf_path, PdfReader))
            return documents

        return self._load_pdf(self.path, PdfReader)

    def _load_pdf(self, file_path: Path, reader_cls: type) -> list[Document]:
        try:
            reader = reader_cls(str(file_path))
        except Exception as exc:
            raise LoaderError(f"Failed to read PDF {file_path}: {exc}") from exc

        total_pages = len(reader.pages)
        page_indices = (
            list(self.pages) if self.pages else list(range(total_pages))
        )

        if self.one_doc_per_page:
            documents: list[Document] = []
            for page_num in page_indices:
                if page_num >= total_pages:
                    continue
                text = reader.pages[page_num].extract_text() or ""
                documents.append(
                    Document(
                        content=text.strip(),
                        source=str(file_path),
                        metadata={
                            "filename": file_path.name,
                            "page": page_num,
                            "total_pages": total_pages,
                        },
                    )
                )
            return documents

        # One document for the whole PDF
        all_text_parts: list[str] = []
        for page_num in page_indices:
            if page_num >= total_pages:
                continue
            text = reader.pages[page_num].extract_text() or ""
            all_text_parts.append(text.strip())

        return [
            Document(
                content="\n\n".join(all_text_parts),
                source=str(file_path),
                metadata={
                    "filename": file_path.name,
                    "total_pages": total_pages,
                    "pages_loaded": len(page_indices),
                },
            )
        ]

    def __repr__(self) -> str:
        return f"PDFLoader(path={self.path!r})"
