import secrets

from fastapi import HTTPException, Request, status


def require_api_key(request: Request) -> None:
    """Protect the API with an optional shared key.

    Authentication is disabled when no key is configured, which keeps local
    development frictionless. When a key is configured it must be supplied
    either as ``X-API-Key`` or as a bearer token.
    """

    settings = getattr(request.app.state, "settings", None)
    expected = getattr(settings, "api_key", None)

    if not expected:
        return

    provided = request.headers.get("x-api-key")

    if not provided:
        authorization = request.headers.get("authorization", "")

        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
