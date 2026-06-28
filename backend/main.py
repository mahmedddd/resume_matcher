"""
FastAPI application entry point.
"""
import asyncio
import sys

# Playwright (and asyncio subprocesses in general) require the Proactor
# event loop on Windows. Must be set before any loop is created.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from routers import applications, cv, internships, scraper, user


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables
    await init_db()
    # Heal any zombie applications (agent process died without updating DB)
    await _heal_zombie_applications()
    yield
    # Shutdown: nothing to clean up


async def _heal_zombie_applications():
    """Mark applications stuck in Applying/Pending as Failed on startup."""
    from database import AsyncSessionLocal, Application
    from sqlalchemy import update
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            update(Application)
            .where(Application.status.in_(["Applying", "Pending"]))
            .values(
                status="Failed",
                notes="Agent process was interrupted (server restarted). Please re-apply.",
            )
        )
        if result.rowcount:
            print(f"[Startup] Healed {result.rowcount} zombie application(s).")
        await db.commit()



app = FastAPI(
    title="SkillSync PK API",
    description="AI-powered Pakistan internship discovery & CV matching",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    # Log OPTIONS requests specifically
    if request.method == "OPTIONS":
        print(f"DEBUG PREFLIGHT: {request.url.path}")
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"DEBUG: {request.method} {request.url.path} - Status: {response.status_code} - {duration:.2f}s")
    return response

@app.get("/api/debug")
async def debug_routes():
    return {"routes": [route.path for route in app.routes]}

@app.post("/api/test-upload")
async def test_upload():
    return {"status": "reached"}

app.include_router(scraper.router, prefix="/api/v1/scraper", tags=["Scraper"])
app.include_router(internships.router, prefix="/api/v1/internships", tags=["Internships"])
app.include_router(cv.router, prefix="/api/v1/cv", tags=["CV Matching"])
app.include_router(applications.router, prefix="/api/v1/applications", tags=["Autonomous Applications"])
app.include_router(user.router, prefix="/api/v1/profile", tags=["User Profile"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "SkillSync PK API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
