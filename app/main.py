import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routes import router as api_router
from app.settings import settings

logging.basicConfig(
    level=getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.APP_NAME,
    description="Medicine Safety Risk Assessment Engine",
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "description": "Medicine Safety Risk Assessment Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "status": "/status (GET)",
            "assess_start": "/assess/start (POST) - Start assessment (returns thread_id if approval needed)",
            "assess_approve": "/assess/approve (POST) - Submit approval decision",
            "assess_file": "/assess/file (POST) - File upload input",
            "assess_batch": "/assess/batch (POST) - Batch processing",
            "info": "/info (GET)",
        },
        "docs": "/docs (Swagger UI)",
        "redoc": "/redoc (ReDoc)",
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
