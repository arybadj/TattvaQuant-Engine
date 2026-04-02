from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.routes.analytics import router as analytics_router
from app.routes.auth import router as auth_router
from app.routes.candidates import router as candidate_router
from app.routes.dashboard import router as dashboard_router
from app.routes.interviews import router as interview_router
from app.routes.ui import router as ui_router
from app.routes.webhooks import router as webhook_router
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger.info("application_starting", environment=settings.environment)
    yield
    logger.info("application_stopping")


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(ui_router)
app.include_router(auth_router, prefix="/api")
app.include_router(candidate_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(interview_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(webhook_router, prefix="/api")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
