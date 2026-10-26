from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from backend.security.rate_limiter import limiter

from backend.routes.admin_routes import router as admin_router
from backend.routes.auth_routes import router as auth_router
from backend.routes.document_routes import router as document_router
from backend.routes.form_routes import router as form_router
from backend.routes.profile_routes import router as profile_router
from backend.security.headers_middleware import SecurityHeadersMiddleware
from backend.services.token_service import setup_token_blacklist_ttl_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure MongoDB TTL index for revoked tokens
    setup_token_blacklist_ttl_index()
    yield



app = FastAPI(
    title="AutoForm Unified API",
    version="1.0.0",
    description="Privacy-focused, encrypted document extraction and autofill engine.",
    lifespan=lifespan,
)

# SlowAPI rate limiting configuration
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(form_router)
app.include_router(admin_router)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


@app.get("/", include_in_schema=False)
async def serve_landing_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True,
    ),
    name="frontend",
)