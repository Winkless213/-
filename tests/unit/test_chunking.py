from app.chunking.splitter import split_document


class TestSplitMarkdown:
    def test_simple_headings(self):
        text = "# Title One\nContent A\n\n## Title Two\nContent B"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 2
        assert chunks[0].metadata["heading"] == "Title One"
        assert chunks[1].metadata["heading"] == "Title Two"
        assert chunks[0].metadata["filename"] == "test.md"
        assert chunks[0].metadata["position"] == 0
        assert chunks[1].metadata["position"] == 1

    def test_long_section_splits_by_size(self):
        long_content = "A" * 1000
        text = f"# Long\n{long_content}"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 450  # max_size + some tolerance for overlap
            assert chunk.metadata["heading"] == "Long"

    def test_no_headings_plain_text(self):
        text = "Hello world. " * 200  # ~2600 chars
        chunks = split_document(text, filename="test.txt", max_size=400, overlap=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.metadata["heading"] == ""

    def test_empty_text(self):
        chunks = split_document("", filename="test.md", max_size=400, overlap=50)
        assert chunks == []

    def test_whitespace_only(self):
        chunks = split_document("   \n\n  ", filename="test.md", max_size=400, overlap=50)
        assert chunks == []

    def test_single_short_section(self):
        text = "# Title\nShort content"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short content"
        assert chunks[0].metadata["heading"] == "Title"

    def test_overlap_within_heading_only(self):
        """Overlap should only apply within a heading section, not across headings."""
        text = "# A\n" + "X" * 500 + "\n# B\n" + "Y" * 500
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        a_chunks = [c for c in chunks if c.metadata["heading"] == "A"]
        b_chunks = [c for c in chunks if c.metadata["heading"] == "B"]
        assert len(a_chunks) >= 1
        assert len(b_chunks) >= 1
        # No chunk should mix content from both sections
        for c in a_chunks:
            assert "Y" * 10 not in c.text
        for c in b_chunks:
            assert "X" * 10 not in c.text

    def test_multiple_heading_levels(self):
        text = "# H1\nContent\n## H2\nSub\n### H3\nDeep"
        chunks = split_document(text, filename="test.md", max_size=400, overlap=50)
        assert len(chunks) == 3
        assert chunks[0].metadata["heading"] == "H1"
        assert chunks[1].metadata["heading"] == "H2"
        assert chunks[2].metadata["heading"] == "H3"
