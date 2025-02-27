"""
Simple standalone server for dashboard development.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Import only the dashboard routes
from app.api.routes.dashboard import router as dashboard_router

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "app" / "static"), name="static")

# Setup Jinja templates
templates = Jinja2Templates(directory=Path(__file__).parent / "app" / "templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dashboard routes
app.include_router(dashboard_router, prefix="/api/dashboard")

# Add root route to serve dashboard
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse("pages/dashboard.html", {"request": request})

# Run with: uvicorn dashboard_server:app --reload --host 0.0.0.0 --port 8002
