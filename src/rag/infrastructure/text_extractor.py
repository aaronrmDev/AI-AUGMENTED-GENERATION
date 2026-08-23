import io

from pypdf import PdfReader

from src.rag.domain.errors import UnsupportedFileType


class TextExtractor:
    def extract(self, filename: str, content: bytes) -> str:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension == ".txt":
            return content.decode("utf-8")

        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        raise UnsupportedFileType(extension or filename)
