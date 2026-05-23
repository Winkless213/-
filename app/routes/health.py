from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()

_health_data = {}


def set_health_data(data: dict):
    global _health_data
    _health_data = data


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(**_health_data)
