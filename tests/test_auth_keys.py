import os
import pytest

from app.auth.keys import APIKeyStore


def test_empty_api_keys_fail_closed(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    store = APIKeyStore()
    assert len(store) == 0
    assert store.validate("anything") is None


def test_api_key_requires_minimum_entropy_and_safe_user_id(monkeypatch):
    monkeypatch.setenv("API_KEYS", "short:user")
    with pytest.raises(ValueError):
        APIKeyStore()
    monkeypatch.setenv("API_KEYS", ("x" * 40) + ":bad user")
    with pytest.raises(ValueError):
        APIKeyStore()


def test_api_key_validates_and_is_scoped(monkeypatch):
    key = "K" * 40
    monkeypatch.setenv("API_KEYS", f"{key}:operator")
    store = APIKeyStore()
    assert store.validate(key) == {"user_id": "operator"}
    assert store.validate("wrong") is None


def test_duplicate_api_key_is_rejected(monkeypatch):
    key = "K" * 40
    monkeypatch.setenv("API_KEYS", f"{key}:one,{key}:two")
    with pytest.raises(ValueError):
        APIKeyStore()
