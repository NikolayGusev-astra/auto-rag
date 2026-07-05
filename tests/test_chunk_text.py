import pytest
from indexer import chunk_text, parse_frontmatter


class TestChunkText:
    def test_simple_chunk(self):
        text = "# Title\n\nSome content here."
        chunks = chunk_text(text)
        assert len(chunks) >= 1

    def test_long_text_split(self):
        text = "# Title\n\n" + "x" * 3000
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c["text"]) <= 2000

    def test_frontmatter_parsing(self):
        text = "---\ntitle: Test\ntags: [a, b]\n---\n\nContent"
        meta, body = parse_frontmatter(text)
        assert meta.get("title") == "Test"
        assert "Content" in body

    def test_no_frontmatter(self):
        text = "Just content"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Just content"

    def test_code_block_not_split(self):
        code = "```python\n" + "\n".join([f"line{i}" for i in range(200)]) + "\n```"
        text = f"# Title\n\n{code}"
        chunks = chunk_text(text)
        assert any("```python" in c["text"] and "line199" in c["text"] for c in chunks)
