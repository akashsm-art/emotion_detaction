"""
Main FastAPI Application Entry Point.
Initializes the Multi-Modal Emotion Recognition API, configures CORS,
mounts modular routes under /api, and provides OpenAPI documentation.
"""

import os
import logging
from typing import List
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.routes import health, text, audio, vision, multimodal

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")

# Load CORS origins from environment
frontend_origin_env = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins: List[str] = [orig.strip() for orig in frontend_origin_env.split(",") if orig.strip()]

app = FastAPI(
    title="Multi-Modal Emotion Recognition API",
    description=(
        "Production-grade REST API for Multi-Modal Emotion Recognition. "
        "Integrates deep RoBERTa sequence classification for Text, 2D CNN acoustic inference for Speech, "
        "and confidence-aware late decision fusion. "
        "Note: Vision emotion classification is currently blocked due to missing trained weights in the source repository."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Exception handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "validation_error",
            "detail": exc.errors(),
            "body": str(exc.body),
        },
    )


# Global unhandled exception handler to prevent stack traces leaking to clients
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled internal server error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "internal_error",
            "detail": "An internal server error occurred while processing the request.",
        },
    )


# Mount API routers
app.include_router(health.router, prefix="/api")
app.include_router(text.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(vision.router, prefix="/api")
app.include_router(multimodal.router, prefix="/api")


# Root health redirect / convenience route
@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Multi-Modal Emotion Recognition API",
        "version": "1.0.0",
        "docs_url": "/api/docs",
        "health_url": "/api/health",
        "status": "operational",
    }
