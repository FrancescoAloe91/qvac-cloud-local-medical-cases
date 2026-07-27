from cryptography.fernet import Fernet

from lib.secure_account_store import decrypt_text, encrypt_text


def test_sensitive_values_are_encrypted_at_rest(monkeypatch):
    monkeypatch.setenv("APP_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    plaintext = "sk-or-v1-example-secret"
    ciphertext = encrypt_text(plaintext)
    assert plaintext not in ciphertext
    assert decrypt_text(ciphertext) == plaintext
