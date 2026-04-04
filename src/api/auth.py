"""API key authentication.

Provides a FastAPI dependency that validates API key credentials
from the X-API-Key header or api_key query parameter. When no
API_KEY is configured in settings, authentication is disabled
(open access) — suitable for local development.

Uses constant-time comparison to prevent timing attacks.
"""

import hmac

import structlog
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from src.config import settings

logger = structlog.get_logger(__name__)

# Accept API key from header (preferred) or query parameter (convenience)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def verify_api_key(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> str | None:
    """Validate the API key from header or query parameter.

    When API_KEY is not set in configuration, authentication is
    bypassed entirely (returns None). This allows local development
    without credentials while requiring them in production.

    Args:
        header_key: Value from X-API-Key header.
        query_key: Value from api_key query parameter.

    Returns:
        The validated API key string, or None if auth is disabled.

    Raises:
        HTTPException 401: If auth is enabled and no valid key is provided.
        HTTPException 403: If auth is enabled and the key is invalid.
    """
    configured_key = settings.api.api_key

    # Auth disabled — open access
    if configured_key is None:
        return None

    # Resolve the provided key (header takes precedence)
    provided_key = header_key or query_key

    if provided_key is None:
        logger.warning("auth_missing_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide it via X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison prevents timing attacks
    if not hmac.compare_digest(provided_key, configured_key):
        logger.warning("auth_invalid_key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return provided_key
