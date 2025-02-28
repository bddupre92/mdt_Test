"""
Main Flask application.
"""
from flask import Flask, render_template, jsonify, request
import json
import os
from benchmarking.benchmark_runner import BenchmarkRunner

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    benchmark_file = os.path.join('data', 'benchmarks', 'benchmark_comparison.json')
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r') as f:
            benchmarks = json.load(f)
        return jsonify(benchmarks)
    return jsonify({"error": "Benchmark data not found."}), 404

@app.route('/api/dashboard/run-benchmark-comparison', methods=['POST'])
def run_benchmark_comparison():
    # Initialize benchmark runner with desired settings
    runner = BenchmarkRunner(
        optimizers=[],  # Add your optimizers here
        n_runs=1,
        max_evaluations=1000,
        use_ray=False
    )
    # Run benchmark comparison
    results = runner.run_theoretical_benchmarks()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
