// v0test/components/benchmark-results.tsx
'use client';
import { useState, useEffect } from 'react';
import { getBenchmark, BenchmarkResult, OptimizerResult } from '../lib/api/benchmarks';
import { useRouter } from 'next/navigation';

interface BenchmarkResultsProps {
  benchmarkId?: string;
}

export default function BenchmarkResults({ benchmarkId }: BenchmarkResultsProps) {
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null);
  const router = useRouter();

  // Fetch benchmark data
  const fetchBenchmark = async () => {
    if (!benchmarkId) return;
    
    try {
      setLoading(true);
      const data = await getBenchmark(benchmarkId);
      setBenchmark(data);
      
      // If benchmark is still running, set up polling
      if (data.status === 'pending' || data.status === 'running') {
        if (!refreshInterval) {
          const intervalId = window.setInterval(() => {
            fetchBenchmark();
          }, 2000); // Poll every 2 seconds
          setRefreshInterval(intervalId);
        }
      } else if (refreshInterval) {
        // Clear interval if benchmark is completed
        clearInterval(refreshInterval);
        setRefreshInterval(null);
      }
      
      setError(null);
    } catch (err) {
      setError('Failed to fetch benchmark results');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch and cleanup
  useEffect(() => {
    fetchBenchmark();
    
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [benchmarkId]);

  // Render loading state
  if (loading && !benchmark) {
    return <div className="p-4">Loading benchmark results...</div>;
  }

  // Render error state
  if (error) {
    return (
      <div className="p-4 text-red-500">
        <p>{error}</p>
        <button 
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded"
          onClick={fetchBenchmark}
        >
          Retry
        </button>
      </div>
    );
  }

  // Render empty state
  if (!benchmark) {
    return <div className="p-4">No benchmark results found.</div>;
  }

  // Render results
  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">
        Benchmark Results: {benchmark.function_name}
      </h2>
      
      <div className="mb-4">
        <p><strong>Status:</strong> {benchmark.status}</p>
        <p><strong>Dimension:</strong> {benchmark.dimension}</p>
        <p><strong>Max Evaluations:</strong> {benchmark.max_evaluations}</p>
        <p><strong>Repetitions:</strong> {benchmark.repetitions}</p>
      </div>
      
      {benchmark.status === 'pending' || benchmark.status === 'running' ? (
        <div className="mb-4">
          <p>Benchmark is currently running...</p>
          <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
            <div className="h-full bg-blue-500 rounded-full animate-pulse"></div>
          </div>
        </div>
      ) : null}
      
      {benchmark.error ? (
        <div className="p-4 bg-red-50 border border-red-200 rounded mb-4">
          <p className="text-red-500"><strong>Error:</strong> {benchmark.error}</p>
        </div>
      ) : null}
      
      {benchmark.results ? (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Optimizer Results</h3>
          
          <table className="min-w-full border border-gray-200">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 border">Optimizer</th>
                <th className="p-2 border">Mean Score</th>
                <th className="p-2 border">Std Score</th>
                <th className="p-2 border">Mean Time (s)</th>
                <th className="p-2 border">Success Rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(benchmark.results).map(([optimizer, data]) => (
                <tr key={optimizer}>
                  <td className="p-2 border font-medium">{optimizer}</td>
                  <td className="p-2 border">
                    {data.mean_score !== undefined 
                      ? data.mean_score.toExponential(4) 
                      : 'N/A'}
                  </td>
                  <td className="p-2 border">
                    {data.std_score !== undefined 
                      ? data.std_score.toExponential(4) 
                      : 'N/A'}
                  </td>
                  <td className="p-2 border">
                    {data.mean_time !== undefined 
                      ? data.mean_time.toFixed(2) 
                      : 'N/A'}
                  </td>
                  <td className="p-2 border">
                    {data.success_rate !== undefined 
                      ? `${(data.success_rate * 100).toFixed(0)}%` 
                      : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      
      <button 
        className="mt-6 px-4 py-2 bg-blue-500 text-white rounded"
        onClick={() => router.back()}
      >
        Back
      </button>
    </div>
  );
}