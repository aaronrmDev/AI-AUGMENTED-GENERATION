import io

import pytest
from pypdf import PdfWriter

from src.rag.domain.errors import UnsupportedFileType
from src.rag.infrastructure.text_extractor import TextExtractor


def _make_minimal_pdf_bytes(text: str) -> bytes:
    # Build a tiny real PDF in memory rather than committing a binary fixture file.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf's PdfWriter has no direct "write text" helper; a blank-page PDF is
    # sufficient to prove extraction runs without error against a real PDF
    # structure. The extraction test below checks pypdf.PdfReader is invoked
    # correctly, not that specific text round-trips — that would require a
    # heavier PDF-generation dependency this project doesn't otherwise need.
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_txt_content_is_decoded_directly():
    extractor = TextExtractor()
    result = extractor.extract("notes.txt", b"hello world")
    assert result == "hello world"


def test_txt_content_that_is_not_valid_utf8_does_not_raise():
    extractor = TextExtractor()
    # 0xFF is not a legal UTF-8 byte anywhere -- a strict decode raises
    # UnicodeDecodeError here, which reaches the client as an unhandled 500.
    result = extractor.extract("latin1.txt", b"caf\xe9 \xff au lait")
    assert "caf" in result
    assert "au lait" in result


def test_pdf_content_is_extracted_via_pypdf():
    extractor = TextExtractor()
    pdf_bytes = _make_minimal_pdf_bytes("irrelevant")
    # A blank page extracts to an empty string — this proves the pypdf path
    # runs without raising, which is what this test is actually checking.
    result = extractor.extract("report.pdf", pdf_bytes)
    assert result == ""


def test_unsupported_extension_raises():
    extractor = TextExtractor()
    with pytest.raises(UnsupportedFileType):
        extractor.extract("archive.docx", b"whatever")
