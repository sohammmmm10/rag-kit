"""Plain text and directory loader."""

from __future__ import annotations

from pathlib import Path

from ..errors import LoaderError, ValidationError
from ..models import Document
from .base import BaseLoader


class TextLoader(BaseLoader):
    """Load plain text files (.txt, .md, or any text file)."""

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        glob_pattern: str = "*.txt",
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.glob_pattern = glob_pattern

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise ValidationError(f"Path does not exist: {self.path}")

        if self.path.is_file():
            return [self._load_file(self.path)]

        if self.path.is_dir():
            documents: list[Document] = []
            files = sorted(self.path.rglob(self.glob_pattern))
            if not files:
                raise LoaderError(
                    f"No files matching '{self.glob_pattern}' found in {self.path}"
                )
            for file_path in files:
                documents.append(self._load_file(file_path))
            return documents

        raise LoaderError(f"Path is neither a file nor a directory: {self.path}")

    def _load_file(self, file_path: Path) -> Document:
        try:
            content = file_path.read_text(encoding=self.encoding)
        except Exception as exc:
            raise LoaderError(f"Failed to read {file_path}: {exc}") from exc

        return Document(
            content=content,
            source=str(file_path),
            metadata={
                "filename": file_path.name,
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
            },
        )

    def __repr__(self) -> str:
        return f"TextLoader(path={self.path!r})"
