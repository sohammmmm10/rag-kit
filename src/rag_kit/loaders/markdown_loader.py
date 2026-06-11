"""Markdown document loader."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import LoaderError, ValidationError
from ..models import Document
from .base import BaseLoader


class MarkdownLoader(BaseLoader):
    """Load Markdown files, optionally splitting by headings."""

    def __init__(
        self,
        path: str | Path,
        *,
        split_by_heading: bool = False,
        heading_level: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.split_by_heading = split_by_heading
        self.heading_level = heading_level
        self.encoding = encoding

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise ValidationError(f"Path does not exist: {self.path}")

        if self.path.is_dir():
            documents: list[Document] = []
            md_files = sorted(self.path.rglob("*.md"))
            if not md_files:
                raise LoaderError(f"No Markdown files found in {self.path}")
            for md_path in md_files:
                documents.extend(self._load_file(md_path))
            return documents

        return self._load_file(self.path)

    def _load_file(self, file_path: Path) -> list[Document]:
        try:
            content = file_path.read_text(encoding=self.encoding)
        except Exception as exc:
            raise LoaderError(f"Failed to read {file_path}: {exc}") from exc

        if not self.split_by_heading:
            return [
                Document(
                    content=content,
                    source=str(file_path),
                    metadata={
                        "filename": file_path.name,
                        "extension": ".md",
                    },
                )
            ]

        return self._split_by_heading(content, file_path)

    def _split_by_heading(
        self, content: str, file_path: Path
    ) -> list[Document]:
        pattern = rf"^(#{{1,{self.heading_level}}})\s+(.+)$"
        sections: list[Document] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in content.split("\n"):
            match = re.match(pattern, line)
            if match:
                # Save the previous section
                if current_lines:
                    sections.append(
                        Document(
                            content="\n".join(current_lines).strip(),
                            source=str(file_path),
                            metadata={
                                "filename": file_path.name,
                                "heading": current_heading,
                                "section_index": len(sections),
                            },
                        )
                    )
                current_heading = match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            sections.append(
                Document(
                    content="\n".join(current_lines).strip(),
                    source=str(file_path),
                    metadata={
                        "filename": file_path.name,
                        "heading": current_heading,
                        "section_index": len(sections),
                    },
                )
            )

        return sections if sections else [
            Document(
                content=content,
                source=str(file_path),
                metadata={"filename": file_path.name},
            )
        ]

    def __repr__(self) -> str:
        return f"MarkdownLoader(path={self.path!r})"
