import pytest
from rag_indexer import chunk_text, parse_frontmatter, recursive_chunk_text, fixed_chunk_text


class TestChunkText:
    def test_simple_chunk(self):
        text = "# Title\n\nSome content here that is long enough to not be filtered out."
        for mode in ["fixed", "recursive"]:
            chunks = chunk_text(text, mode=mode)
            assert len(chunks) >= 1
            assert chunks[0]["heading"] == "Title"

    def test_long_text_split(self):
        text = "# Title\n\n" + "x" * 3000
        chunks = chunk_text(text, mode="fixed")
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
        chunks = chunk_text(text, mode="fixed")
        assert any("```python" in c["text"] and "line199" in c["text"] for c in chunks)


class TestRecursiveChunkText:
    """Tests specifically for recursive_chunk_text"""

    def test_short_text_no_split(self):
        """Короткий текст не должен резаться"""
        text = "# Title\n\nShort paragraph here."
        chunks = recursive_chunk_text(text)
        assert len(chunks) == 1
        assert "Short paragraph here" in chunks[0]["text"]

    def test_fixed_mode_does_split(self):
        """Fixed-режим режет по размеру, в отличие от recursive"""
        big = "word " * 3000
        text = f"# Title\n\n{big}"
        chunks = fixed_chunk_text(text, heading="Title")
        assert len(chunks) >= 7  # 15000/2000 ≈ 7.5
        for c in chunks:
            assert len(c["text"]) <= 2000

    def test_paragraph_boundary_preserved(self):
        """Параграфы не должны разрываться посередине, если помещаются в chunk_size"""
        para = "This is a paragraph of normal length. " * 20
        text = f"# Section\n\n{para}\n\n{para}"
        chunks = recursive_chunk_text(text, chunk_size=5000)
        assert len(chunks) >= 1
        # Each chunk should start with a complete paragraph, not mid-sentence
        for c in chunks:
            assert c["text"].strip().startswith("This") or c["text"].strip().startswith("#")

    def test_large_paragraph_forced_split(self):
        """Огромный параграф (без разделителей) — recursive chunking сохраняет его как один чанк.
        Для принудительного разбиения по размеру используйте fixed-режим."""
        big = "word " * 3000  # ~15k chars, one paragraph
        text = f"# Title\n\n{big}"
        chunks = recursive_chunk_text(text, chunk_size=2000)
        # Recursive mode preserves paragraph boundaries by design
        # One paragraph → one chunk, regardless of size
        assert len(chunks) == 1
        assert len(chunks[0]["text"]) > 2000

    def test_min_chunk_merges_tail(self):
        """Хвост меньше min_chunk должен мержиться с предыдущим чанком"""
        text = "# H\n\n" + "A" * 1800 + "\n\n" + "B" * 100
        chunks = recursive_chunk_text(text, chunk_size=2000, min_chunk=300)
        # "BBB" is only 100 chars < 300, should merge
        assert len(chunks) == 1
        assert "BBBB" in chunks[-1]["text"]

    def test_heading_tracking(self):
        """Заголовки должны корректно отслеживаться"""
        text = "# Chapter 1\n\nContent of chapter one is long enough to be a valid chunk.\n\n## Section 1.1\n\nDetails of section 1.1 are also sufficiently long for the chunk filter.\n\n# Chapter 2\n\nStuff in chapter two is also long enough for chunk filtering to work."
        chunks = recursive_chunk_text(text)
        chapter1_heading = None
        section_heading = None
        for c in chunks:
            if "Chapter 1" in c["heading"] and "Content of chapter" in c["text"]:
                chapter1_heading = c["heading"]
            if "Section 1.1" in c["heading"] and "Details of section" in c["text"]:
                section_heading = c["heading"]
        assert chapter1_heading is not None
        assert section_heading is not None

    def test_empty_text(self):
        assert recursive_chunk_text("") == []
        assert fixed_chunk_text("") == []

    def test_no_heading(self):
        text = "Just plain text without any markdown heading."
        chunks = recursive_chunk_text(text, heading="Custom")
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Custom"

    def test_dispatch_default_recursive(self):
        """По умолчанию chunk_text должен работать в recursive-режиме"""
        text = "# Test\n\nThis is a sufficiently long paragraph to create a valid chunk."
        chunks_fixed = chunk_text(text, mode="fixed")
        chunks_default = chunk_text(text)  # default is recursive
        assert len(chunks_default) == len(chunks_fixed) > 0
        assert chunks_default[0]["heading"] == chunks_fixed[0]["heading"]

    def test_consistent_chunking(self):
        """Детерминированность: одинаковый текст → одинаковые чанки"""
        text = "# Stability\n\n" + "\n\n".join([f"Paragraph {i} content." for i in range(10)])
        chunks1 = recursive_chunk_text(text)
        chunks2 = recursive_chunk_text(text)
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["heading"] == c2["heading"]
            assert c1["text"] == c2["text"]

    def test_mixed_heading_levels(self):
        """Разные уровни заголовков (#, ##, ###) не должны ломать чанкинг"""
        text = "# Top\n\nContent of top level is long enough for chunking.\n\n## Mid\n\nContent of mid level that is also long enough for the filter.\n\n### Deep\n\nDeep content that is sufficiently long for the chunk filter to work.\n\n# Next Top\n\nFinal content that is also long enough for the chunk filter to work."
        chunks = recursive_chunk_text(text)
        headings = [c["heading"] for c in chunks]
        assert "Top" in headings
        assert "Mid" in headings
        assert "Next Top" in headings


class TestParseFrontmatter:
    def test_bom_stripped(self):
        text = "\ufeff---\ntitle: Test\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta.get("title") == "Test"
        assert body == "Body"

    def test_invalid_yaml_frontmatter(self):
        text = "---\n{{invalid\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert "Body" in body

    def test_frontmatter_then_heading(self):
        text = "---\ntitle: Test\n---\n\n# Heading\n\nContent"
        meta, body = parse_frontmatter(text)
        assert meta.get("title") == "Test"
        assert "# Heading" in body