import uuid
from pathlib import Path


class LocalFileStorage:
    def save(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, filename: str, content: bytes
    ) -> str:
        # Path(filename).name strips every directory component the client can
        # smuggle through the multipart Content-Disposition header -- "..",
        # an absolute prefix, embedded separators -- leaving only the final
        # path segment. Without it, a filename like "../../src/api/main.py"
        # escapes the tenant's directory and overwrites arbitrary files.
        safe_name = Path(filename).name
        directory = Path("storage") / str(tenant_id) / str(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe_name
        path.write_bytes(content)
        return str(path)
