"""
Main Flask application.
"""
from flask import Flask, render_template, jsonify, request
import json
import os
from benchmarking.benchmark_runner import run_benchmark_comparison

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmark_results():
    benchmark_file = os.path.join('benchmark_comparison_results', 'theoretical_results.csv')
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r') as f:
            benchmarks = json.load(f)
        return jsonify(benchmarks)
    return jsonify({"error": "Benchmark data not found."}), 404

@app.route('/api/dashboard/run-benchmark-comparison', methods=['POST'])
def run_benchmark_comparison_route():
    results = run_benchmark_comparison()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
