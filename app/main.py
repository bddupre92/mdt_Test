"""
Main FastAPI application.
"""
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.middleware.auth import AuthMiddleware
from benchmarking.benchmark_runner import run_comprehensive_benchmark

app = FastAPI(title="Migraine Prediction Service")

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
        "/api/auth/token"  # Keep this for backward compatibility
    ]
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Add new routes for benchmark results and comparison
benchmark_router = APIRouter()

@benchmark_router.get("/benchmarks")
async def get_benchmark_results():
    # Placeholder for fetching benchmark results
    return {"message": "Benchmark results"}

@benchmark_router.post("/dashboard/run-benchmark-comparison")
async def run_benchmark_comparison():
    # Placeholder for running benchmark comparison
    run_comprehensive_benchmark()
    return {"message": "Benchmark comparison triggered"}

app.include_router(benchmark_router, prefix="/api")
