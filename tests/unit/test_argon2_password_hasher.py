from src.identity.infrastructure.argon2_password_hasher import Argon2PasswordHasher


def test_hash_produces_an_argon2id_prefixed_value():
    hasher = Argon2PasswordHasher()
    result = hasher.hash("correct horse battery staple")
    assert str(result).startswith("$argon2id$")


def test_hashing_the_same_password_twice_produces_different_hashes():
    hasher = Argon2PasswordHasher()
    a = hasher.hash("correct horse battery staple")
    b = hasher.hash("correct horse battery staple")
    assert str(a) != str(b)


def test_verify_succeeds_for_the_correct_password():
    hasher = Argon2PasswordHasher()
    hashed = hasher.hash("correct horse battery staple")
    assert hasher.verify("correct horse battery staple", hashed) is True


def test_verify_fails_for_the_wrong_password():
    hasher = Argon2PasswordHasher()
    hashed = hasher.hash("correct horse battery staple")
    assert hasher.verify("wrong password", hashed) is False
