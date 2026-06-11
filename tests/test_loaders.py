"""Tests for rag_kit loaders."""

import csv
import pytest
from pathlib import Path

from rag_kit.errors import LoaderError, ValidationError
from rag_kit.loaders import TextLoader, CSVLoader, MarkdownLoader


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("Hello, this is a test document.\nLine two.")
    return f


@pytest.fixture
def tmp_text_dir(tmp_path: Path) -> Path:
    for i in range(3):
        (tmp_path / f"doc_{i}.txt").write_text(f"Content of document {i}.")
    return tmp_path


@pytest.fixture
def tmp_csv_file(tmp_path: Path) -> Path:
    f = tmp_path / "data.csv"
    with open(f, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "answer", "category"])
        writer.writeheader()
        writer.writerow({"question": "What is AI?", "answer": "Artificial Intelligence", "category": "tech"})
        writer.writerow({"question": "What is ML?", "answer": "Machine Learning", "category": "tech"})
    return f


@pytest.fixture
def tmp_md_file(tmp_path: Path) -> Path:
    f = tmp_path / "readme.md"
    f.write_text(
        "# Title\n\nIntro text.\n\n## Section 1\n\nContent one.\n\n## Section 2\n\nContent two."
    )
    return f


class TestTextLoader:
    def test_load_single_file(self, tmp_text_file: Path) -> None:
        loader = TextLoader(tmp_text_file)
        docs = loader.load()
        assert len(docs) == 1
        assert "Hello" in docs[0].content
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_load_directory(self, tmp_text_dir: Path) -> None:
        loader = TextLoader(tmp_text_dir)
        docs = loader.load()
        assert len(docs) == 3

    def test_nonexistent_path(self) -> None:
        with pytest.raises(ValidationError):
            TextLoader("/nonexistent/path").load()

    def test_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError):
            TextLoader(tmp_path).load()


class TestCSVLoader:
    def test_load_csv(self, tmp_csv_file: Path) -> None:
        loader = CSVLoader(tmp_csv_file)
        docs = loader.load()
        assert len(docs) == 2
        assert "AI" in docs[0].content

    def test_content_columns(self, tmp_csv_file: Path) -> None:
        loader = CSVLoader(tmp_csv_file, content_columns=["question"])
        docs = loader.load()
        assert len(docs) == 2
        assert "Artificial Intelligence" not in docs[0].content

    def test_metadata_columns(self, tmp_csv_file: Path) -> None:
        loader = CSVLoader(
            tmp_csv_file,
            content_columns=["question", "answer"],
            metadata_columns=["category"],
        )
        docs = loader.load()
        assert docs[0].metadata["category"] == "tech"

    def test_nonexistent_csv(self) -> None:
        with pytest.raises(ValidationError):
            CSVLoader("/nonexistent/data.csv").load()


class TestMarkdownLoader:
    def test_load_whole_file(self, tmp_md_file: Path) -> None:
        loader = MarkdownLoader(tmp_md_file)
        docs = loader.load()
        assert len(docs) == 1
        assert "# Title" in docs[0].content

    def test_split_by_heading(self, tmp_md_file: Path) -> None:
        loader = MarkdownLoader(tmp_md_file, split_by_heading=True, heading_level=2)
        docs = loader.load()
        assert len(docs) >= 2

    def test_nonexistent_path(self) -> None:
        with pytest.raises(ValidationError):
            MarkdownLoader("/nonexistent/readme.md").load()
