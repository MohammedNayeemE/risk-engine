from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware that validates origins against registered client domains.

    For authenticated requests, this middleware checks if the request origin
    matches the domain registered for the client's API key.

    For unauthenticated requests (like /clients/register), it allows requests
    from configured default origins.
    """

    def __init__(self, app: ASGIApp, allowed_origins: list[str] = None):
        super().__init__(app)
        self.default_allowed_origins = allowed_origins or [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
        ]

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")

        response = await call_next(request)

        allowed = False
        allowed_origin = None

        if hasattr(request.state, "client") and request.state.client:
            client_domain = request.state.client.domain
            if origin and self._origin_matches_domain(origin, client_domain):
                allowed = True
                allowed_origin = origin
        # Otherwise, check default allowed origins
        elif (
            origin in self.default_allowed_origins
            or origin
            and any(
                origin.startswith(default_origin)
                for default_origin in self.default_allowed_origins
            )
        ):
            allowed = True
            allowed_origin = origin

        # Set CORS headers if allowed
        if allowed and allowed_origin:
            response.headers["Access-Control-Allow-Origin"] = allowed_origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-API-Key"
            )

        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=200)
            if allowed and allowed_origin:
                response.headers["Access-Control-Allow-Origin"] = allowed_origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                )
                response.headers["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization, X-API-Key"
                )

        return response

    def _origin_matches_domain(self, origin: str, domain: str) -> bool:
        """
        Check if the request origin matches the registered domain.

        Handles variations like:
        - https://example.com matches https://example.com
        - http://localhost:3000 matches http://localhost:3000
        - Wildcards in domain (*.example.com)
        """
        # Exact match
        if origin == domain:
            return True

        # Handle wildcard subdomains (e.g., *.example.com)
        if domain.startswith("*."):
            base_domain = domain[2:]  # Remove "*."
            if origin.endswith(base_domain):
                return True

        # Handle different ports on same domain
        # e.g., domain="http://localhost:3000" should match "http://localhost:3000"
        if origin.startswith(domain.rstrip("/")):
            return True

        return False
