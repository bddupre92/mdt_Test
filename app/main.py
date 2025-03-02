"""
Main FastAPI application.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.api.routes import router as api_router
from app.core.middleware.auth import AuthMiddleware
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Setup Jinja templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
app.add_middleware(
    AuthMiddleware,
    exclude_paths=[
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/register",
        "/api/dashboard",  
        "/static",
        "/",  # Allow access to root URL
        "/test",  # Allow access to test dashboard
        "/benchmark-test"  # Allow access to benchmark test
    ]
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Add root route to serve dashboard
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard."""
    try:
        return templates.TemplateResponse("pages/dashboard.html", {"request": request})
    except Exception as e:
        import logging
        logging.error(f"Error serving dashboard: {str(e)}")
        return HTMLResponse(content=f"<html><body><h1>Error loading dashboard</h1><p>{str(e)}</p></body></html>", status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard page."""
    return templates.TemplateResponse("pages/dashboard.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def test_dashboard(request: Request):
    """Serve the test dashboard page."""
    return templates.TemplateResponse("test_dashboard.html", {"request": request})

@app.get("/benchmark-test", response_class=HTMLResponse)
async def benchmark_test(request: Request):
    """Serve the benchmark test page."""
    return templates.TemplateResponse("benchmark_test.html", {"request": request})
