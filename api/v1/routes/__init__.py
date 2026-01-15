"""API v1 routes module."""

from fastapi import APIRouter

from api.v1.routes.assess import router as assess_router
from api.v1.routes.clients import router as clients_router
from api.v1.routes.health import router as health_router

# Create a combined router that includes all v1 route routers
router = APIRouter(prefix="/api/v1")

# Include all sub-routers
router.include_router(health_router)
router.include_router(assess_router)
router.include_router(clients_router)

__all__ = ["router"]