import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.routes.dashboard import router as dashboard_router
except ImportError:
    from routes.dashboard import router as dashboard_router


app = FastAPI(title="Smart Grid Forecasting Dashboard API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
