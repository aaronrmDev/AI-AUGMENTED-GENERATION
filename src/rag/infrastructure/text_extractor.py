import io

from pypdf import PdfReader

from src.rag.domain.errors import UnsupportedFileType


class TextExtractor:
    def extract(self, filename: str, content: bytes) -> str:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension == ".txt":
            # errors="replace", not a strict decode: a .txt file in some other
            # encoding is a routine upload, not a programming error, and a
            # strict decode turns it into an unhandled UnicodeDecodeError and
            # a 500. Substituting U+FFFD for the undecodable bytes keeps the
            # rest of the document usable.
            return content.decode("utf-8", errors="replace")

        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        raise UnsupportedFileType(extension or filename)
