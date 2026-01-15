from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from persistence.models import Client
from persistence.repositories import validate_api_key


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API keys for protected routes.

    Exempts the following paths from authentication:
    - / (root)
    - /docs
    - /redoc
    - /openapi.json
    - /health
    - /info
    - /clients/register (public registration endpoint)
    """

    EXEMPT_PATHS = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/info",
        "/api/v1/clients/register",
    }

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide it in the X-API-Key header.",
            )

        result = validate_api_key(api_key)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )

        client, api_key_obj = result

        request.state.client = client
        request.state.api_key = api_key_obj

        return await call_next(request)


async def get_current_client(request: Request) -> Client:
    """
    Dependency to get the current authenticated client.

    Use this in route handlers to access client information:
    ```
    @router.get("/protected")
    async def protected_route(client: Client = Depends(get_current_client)):
        return {"client": client.company_name}
    ```
    """
    if not hasattr(request.state, "client"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return request.state.client
