"""
Main FastAPI application.
"""
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import json
import os

from app.api.routes import router as api_router
from app.core.middleware.auth import AuthMiddleware
from benchmarking.benchmark_runner import run_comprehensive_benchmark, run_benchmark_comparison

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
    benchmark_file = 'benchmark_comparison_results/theoretical_results.csv'
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r') as f:
            benchmarks = json.load(f)
        return benchmarks
    return {"error": "Benchmark data not found."}, 404

@benchmark_router.post("/dashboard/run-benchmark-comparison")
async def run_benchmark_comparison():
    results = run_benchmark_comparison()
    return results

app.include_router(benchmark_router, prefix="/api")
