from fastapi import APIRouter

from api.v1.schemas import HealthResponse
from app.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", service=settings.APP_NAME)


@router.get("/status")
async def get_status() -> dict:
    """Lightweight service status payload."""
    return {
        "status": "running",
        "graph_compiled": True,
        "version": settings.APP_VERSION,
    }
