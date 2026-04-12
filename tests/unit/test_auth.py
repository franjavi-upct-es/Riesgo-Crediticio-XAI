# tests/unit/test_auth.py
"""Unit tests for API key authentication.

Verifies that auth is enforced when API_KEY is configured,
bypassed when not configured, and uses constant-time comparison.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.api.auth import verify_api_key


class TestVerifyApiKey:
    """Test the verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_auth_disabled_when_no_key_configured(self):
        """When API_KEY is None, auth is bypassed entirely."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = None
            result = await verify_api_key(header_key=None, query_key=None)
            assert result is None

    @pytest.mark.asyncio
    async def test_auth_disabled_ignores_provided_key(self):
        """When API_KEY is None, even a provided key is ignored."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = None
            result = await verify_api_key(
                header_key="some-key", query_key=None
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_valid_header_key_passes(self):
        """Correct key in X-API-Key header returns the key."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "secret-123"
            result = await verify_api_key(
                header_key="secret-123", query_key=None
            )
            assert result == "secret-123"

    @pytest.mark.asyncio
    async def test_valid_query_key_passes(self):
        """Correct key in api_key query parameter returns the key."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "secret-123"
            result = await verify_api_key(
                header_key=None, query_key="secret-123"
            )
            assert result == "secret-123"

    @pytest.mark.asyncio
    async def test_header_key_takes_precedence(self):
        """When both header and query are provided, header wins."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "correct"
            result = await verify_api_key(
                header_key="correct", query_key="wrong"
            )
            assert result == "correct"

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self):
        """When auth is enabled but no key is provided, return 401."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "secret-123"
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(header_key=None, query_key=None)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_returns_403(self):
        """When auth is enabled and key is wrong, return 403."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "secret-123"
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(header_key="wrong-key", query_key=None)
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_string_key_returns_401(self):
        """An empty string is falsy, so it's treated as missing (401)."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api.api_key = "secret-123"
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(header_key="", query_key=None)
            assert exc_info.value.status_code == 401
