"""CSV document loader."""

from __future__ import annotations

import csv
from pathlib import Path

from ..errors import LoaderError, RagKitError, ValidationError
from ..models import Document
from .base import BaseLoader


class CSVLoader(BaseLoader):
    """Load documents from a CSV file.

    Each row becomes a separate Document. Use ``content_columns`` to
    specify which columns to join as the document content.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.delimiter = delimiter
        self.encoding = encoding

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise ValidationError(f"CSV file does not exist: {self.path}")
        if not self.path.is_file():
            raise ValidationError(f"Path is not a file: {self.path}")

        try:
            with open(self.path, newline="", encoding=self.encoding) as fh:
                reader = csv.DictReader(fh, delimiter=self.delimiter)
                if reader.fieldnames is None:
                    raise LoaderError(f"CSV file has no header row: {self.path}")

                content_cols = self.content_columns or list(reader.fieldnames)
                meta_cols = self.metadata_columns or []

                documents: list[Document] = []
                for row_num, row in enumerate(reader):
                    parts = [
                        str(row.get(col, ""))
                        for col in content_cols
                        if row.get(col)
                    ]
                    content = " | ".join(parts)
                    if not content.strip():
                        continue

                    metadata: dict[str, str] = {
                        "source": str(self.path),
                        "row": str(row_num),
                    }
                    for col in meta_cols:
                        metadata[col] = str(row.get(col, ""))

                    documents.append(
                        Document(
                            content=content,
                            source=str(self.path),
                            metadata=metadata,
                        )
                    )

                return documents
        except RagKitError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to read CSV {self.path}: {exc}") from exc

    def __repr__(self) -> str:
        return f"CSVLoader(path={self.path!r})"
