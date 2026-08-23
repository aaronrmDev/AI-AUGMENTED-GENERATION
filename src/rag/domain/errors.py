class UnsupportedFileType(Exception):
    def __init__(self, extension: str) -> None:
        super().__init__(f"Unsupported file type: {extension}")
        self.extension = extension
