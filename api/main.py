# api/main.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Migraine Digital Twin API",
    description="API for Migraine Digital Twin application",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js development server
        "http://localhost:4001",  # Custom port
        "http://127.0.0.1:4001"   # Alternative address
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from api.routers import benchmarks, optimization, visualization, prediction, framework

app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["visualization"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["prediction"])
app.include_router(framework.router, prefix="/api/framework", tags=["framework"])

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Migraine Digital Twin API",
        "version": "0.1.0",
        "documentation": "/docs",
        "endpoints": [
            "/api/benchmarks",
            "/api/optimization", 
            "/api/visualization",
            "/api/prediction",
            "/api/framework"
        ]
    }