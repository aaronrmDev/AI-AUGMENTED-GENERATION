class InvalidCredentials(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid credentials")


class EmailAlreadyRegistered(Exception):
    def __init__(self, email: str) -> None:
        super().__init__(f"Email already registered: {email}")
        self.email = email


class TokenExpired(Exception):
    def __init__(self) -> None:
        super().__init__("Token expired")


class TokenAlreadyUsed(Exception):
    def __init__(self) -> None:
        super().__init__("Token already used or invalid")
