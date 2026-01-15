import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routes import router as api_router
from app.middleware.auth import AuthMiddleware
from app.middleware.cors import DynamicCORSMiddleware
from app.settings import settings

logging.basicConfig(
    level=getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.APP_NAME,
    description="Medicine Safety Risk Assessment Engine - Public API with API Key Authentication",
    version=settings.APP_VERSION,
)

app.add_middleware(AuthMiddleware)
app.add_middleware(DynamicCORSMiddleware, allowed_origins=settings.CORS_ORIGINS)


@app.get("/")
async def root() -> dict:
    """Root endpoint for basic service metadata."""
    return {"service": settings.APP_NAME, "version": settings.APP_VERSION}


app.include_router(api_router)


# ============================================================================
# Info Endpoint
# ============================================================================


@app.get("/info")
async def get_info():
    """Get API information"""
    return {
        "name": "Risk Engine API",
        "description": "Medicine Safety Risk Assessment Engine - Public API",
        "version": "1.0.0",
        "authentication": "API Key required (X-API-Key header)",
        "endpoints": {
            "health": "/api/v1/health (GET) - Health check",
            "register": "/api/v1/clients/register (POST) - Register new client and get API key",
            "list_keys": "/api/v1/clients/api-keys (GET) - List your API keys",
            "create_key": "/api/v1/clients/api-keys (POST) - Create new API key",
            "revoke_key": "/api/v1/clients/api-keys (DELETE) - Revoke an API key",
            "assess_start": "/api/v1/assess/start (POST) - Start assessment",
            "assess_approve": "/api/v1/assess/approve (POST) - Submit approval decision",
            "info": "/info (GET) - This endpoint",
        },
        "docs": "/docs (Swagger UI)",
        "redoc": "/redoc (ReDoc)",
        "getting_started": {
            "step_1": "Register at /api/v1/clients/register to get your API key",
            "step_2": "Include your API key in the X-API-Key header for all requests",
            "step_3": "Use the /api/v1/assess/* endpoints to perform risk assessments",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )
