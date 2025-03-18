import React from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import BenchmarkSuite from "@/components/meta-optimizer/BenchmarkSuite";

// Import sample data
const SAMPLE_DATASETS = [
  { id: 'migraineA', name: 'Migraine Dataset A', description: 'Clinical migraine dataset with physiological signals' },
  { id: 'migraineB', name: 'Migraine Dataset B', description: 'Patient-reported symptoms with environmental factors' },
  { id: 'benchmark1', name: 'Benchmark Function 1', description: 'Rastrigin function (multimodal)' },
  { id: 'benchmark2', name: 'Benchmark Function 2', description: 'Rosenbrock function (unimodal valley)' },
  { id: 'benchmark3', name: 'Benchmark Function 3', description: 'Ackley function (multimodal noise)' }
];

const SAMPLE_ALGORITHMS = [
  { id: 'ga', name: 'Genetic Algorithm', category: 'evolutionary', description: 'Good for discrete problems' },
  { id: 'de', name: 'Differential Evolution', category: 'evolutionary', description: 'Efficient for continuous problems' },
  { id: 'pso', name: 'Particle Swarm Optimization', category: 'swarm', description: 'Fast convergence' },
  { id: 'es', name: 'Evolution Strategy', category: 'evolutionary', description: 'Robust to noise' },
  { id: 'nm', name: 'Nelder-Mead', category: 'classical', description: 'Local search method' },
  { id: 'bo', name: 'Bayesian Optimization', category: 'bayesian', description: 'Sample-efficient for expensive functions' }
];

const SAMPLE_METRICS = [
  { id: 'accuracy', name: 'Accuracy', description: 'Classification accuracy (higher is better)' },
  { id: 'f1', name: 'F1 Score', description: 'Harmonic mean of precision and recall' },
  { id: 'rmse', name: 'RMSE', description: 'Root mean squared error (lower is better)' },
  { id: 'mae', name: 'MAE', description: 'Mean absolute error (lower is better)' },
  { id: 'convergence', name: 'Convergence Rate', description: 'Speed of convergence' },
  { id: 'robustness', name: 'Robustness', description: 'Performance stability across datasets' }
];

// Sample benchmark results
const SAMPLE_BENCHMARK_RESULTS = [
  {
    id: 'benchmark1',
    algorithmId: 'ga',
    datasetId: 'migraineA',
    metrics: { accuracy: 0.865, f1: 0.872, rmse: 0.124, convergence: 0.78 },
    executionTime: 45.2,
    status: 'completed',
    timestamp: '2023-06-15T14:22:31Z'
  },
  {
    id: 'benchmark2',
    algorithmId: 'de',
    datasetId: 'migraineA',
    metrics: { accuracy: 0.891, f1: 0.885, rmse: 0.102, convergence: 0.85 },
    executionTime: 38.7,
    status: 'completed',
    timestamp: '2023-06-15T14:22:31Z'
  },
  {
    id: 'benchmark3',
    algorithmId: 'pso',
    datasetId: 'migraineA',
    metrics: { accuracy: 0.842, f1: 0.838, rmse: 0.147, convergence: 0.92 },
    executionTime: 29.5,
    status: 'completed',
    timestamp: '2023-06-15T14:22:31Z'
  },
  {
    id: 'benchmark4',
    algorithmId: 'bo',
    datasetId: 'migraineA',
    metrics: { accuracy: 0.879, f1: 0.874, rmse: 0.112, convergence: 0.67 },
    executionTime: 62.1,
    status: 'completed',
    timestamp: '2023-06-15T14:22:31Z'
  },
  {
    id: 'benchmark5',
    algorithmId: 'ga',
    datasetId: 'migraineB',
    metrics: { accuracy: 0.831, f1: 0.827, rmse: 0.152, convergence: 0.71 },
    executionTime: 47.8,
    status: 'completed',
    timestamp: '2023-06-15T15:42:11Z'
  },
  {
    id: 'benchmark6',
    algorithmId: 'de',
    datasetId: 'migraineB',
    metrics: { accuracy: 0.862, f1: 0.858, rmse: 0.128, convergence: 0.79 },
    executionTime: 41.3,
    status: 'completed',
    timestamp: '2023-06-15T15:42:11Z'
  },
  {
    id: 'benchmark7',
    algorithmId: 'es',
    datasetId: 'migraineB',
    metrics: { accuracy: 0.845, f1: 0.841, rmse: 0.135, convergence: 0.76 },
    executionTime: 43.9,
    status: 'completed',
    timestamp: '2023-06-15T15:42:11Z'
  },
  {
    id: 'benchmark8',
    algorithmId: 'nm',
    datasetId: 'benchmark1',
    metrics: { rmse: 0.043, convergence: 0.91, robustness: 0.88 },
    executionTime: 18.5,
    status: 'completed',
    timestamp: '2023-06-16T09:15:42Z'
  },
  {
    id: 'benchmark9',
    algorithmId: 'pso',
    datasetId: 'benchmark1',
    metrics: { rmse: 0.037, convergence: 0.95, robustness: 0.72 },
    executionTime: 22.1,
    status: 'completed',
    timestamp: '2023-06-16T09:15:42Z'
  },
  {
    id: 'benchmark10',
    algorithmId: 'de',
    datasetId: 'benchmark2',
    metrics: { rmse: 0.021, convergence: 0.89, robustness: 0.94 },
    executionTime: 35.7,
    status: 'running',
    timestamp: '2023-06-16T10:05:17Z'
  }
];

export default function BenchmarksPage() {
  // In a real implementation, these would be connected to API calls
  const handleRunBenchmark = async (config: any) => {
    console.log('Running benchmark with config:', config);
    // This would normally call an API
    return new Promise<void>(resolve => setTimeout(resolve, 2000));
  };

  const handleExportResults = (format: 'csv' | 'json') => {
    console.log(`Exporting results as ${format}`);
  };

  return (
    <div className="container py-6 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Benchmark Testing</h1>
        <p className="text-muted-foreground mt-2">
          Test and compare algorithm performance across multiple datasets and metrics
        </p>
      </div>

      <Tabs defaultValue="dashboard" className="space-y-4">
        <TabsList className="bg-background">
          <TabsTrigger value="dashboard">Benchmark Dashboard</TabsTrigger>
          <TabsTrigger value="configure">Configure New Benchmark</TabsTrigger>
          <TabsTrigger value="history">Benchmark History</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Benchmark Results Summary</CardTitle>
              <CardDescription>
                Overview of performance across algorithms and datasets
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex justify-between items-center">
                <div className="space-x-2">
                  <Select defaultValue="accuracy">
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select Metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {SAMPLE_METRICS.map(metric => (
                        <SelectItem key={metric.id} value={metric.id}>
                          {metric.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select defaultValue="migraineA">
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select Dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {SAMPLE_DATASETS.map(dataset => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button variant="outline">Update View</Button>
              </div>

              <div className="rounded-md border">
                <table className="min-w-full divide-y divide-border">
                  <thead>
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">Algorithm</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Accuracy</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">F1 Score</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">RMSE</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Convergence</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Time (s)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-background">
                    {SAMPLE_ALGORITHMS.slice(0, 5).map((algorithm, index) => {
                      const result = SAMPLE_BENCHMARK_RESULTS.find(
                        r => r.algorithmId === algorithm.id && r.datasetId === 'migraineA'
                      );
                      return (
                        <tr key={algorithm.id}>
                          <td className="px-4 py-2 text-sm font-medium">{algorithm.name}</td>
                          <td className="px-4 py-2 text-sm">
                            {result?.metrics.accuracy ? result.metrics.accuracy.toFixed(4) : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {result?.metrics.f1 ? result.metrics.f1.toFixed(4) : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {result?.metrics.rmse ? result.metrics.rmse.toFixed(4) : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {result?.metrics.convergence ? result.metrics.convergence.toFixed(4) : '-'}
                          </td>
                          <td className="px-4 py-2 text-sm">
                            {result?.executionTime ? result.executionTime.toFixed(2) : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <Separator />

              <div>
                <h3 className="text-lg font-medium mb-4">Visual Comparison</h3>
                <div className="flex items-center justify-center bg-muted/20 rounded-md border p-12">
                  <div className="text-center text-muted-foreground">
                    <p>
                      Visualization of algorithm performance would appear here,
                      including bar charts, radar plots, and convergence curves.
                    </p>
                    <p className="mt-2">
                      Select algorithms and metrics above to update the visualization.
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Performance Insights</h3>
                  <div className="rounded-md border p-4 space-y-4">
                    <div>
                      <h4 className="font-medium">Best Overall: Differential Evolution</h4>
                      <p className="text-sm text-muted-foreground">
                        Highest accuracy (89.1%) and lowest RMSE (0.102) on Migraine Dataset A
                      </p>
                    </div>
                    <div>
                      <h4 className="font-medium">Fastest Convergence: Particle Swarm Optimization</h4>
                      <p className="text-sm text-muted-foreground">
                        Quickest to reach optimal values with convergence rate of 0.92
                      </p>
                    </div>
                    <div>
                      <h4 className="font-medium">Most Consistent: Bayesian Optimization</h4>
                      <p className="text-sm text-muted-foreground">
                        Lowest variance in performance across datasets (0.87 ± 0.02)
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Statistical Analysis</h3>
                  <div className="rounded-md border p-4 space-y-4">
                    <div>
                      <h4 className="font-medium">Statistical Significance Tests</h4>
                      <p className="text-sm text-muted-foreground">
                        DE vs GA: p-value = 0.032 (statistically significant)
                      </p>
                      <p className="text-sm text-muted-foreground">
                        DE vs PSO: p-value = 0.047 (statistically significant)
                      </p>
                      <p className="text-sm text-muted-foreground">
                        GA vs PSO: p-value = 0.184 (not statistically significant)
                      </p>
                    </div>
                    <div>
                      <h4 className="font-medium">Critical Difference Diagram</h4>
                      <p className="text-sm text-muted-foreground">
                        Ranking: DE (1.2) {`>`} BO (2.3) {`>`} ES (2.8) {`>`} GA (3.1) {`>`} PSO (3.4) {`>`} NM (4.2)
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline">Export Report</Button>
              <Button>Create New Benchmark</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="configure" className="space-y-6">
          <BenchmarkSuite
            availableDatasets={SAMPLE_DATASETS}
            availableAlgorithms={SAMPLE_ALGORITHMS}
            availableMetrics={SAMPLE_METRICS}
            onRunBenchmark={handleRunBenchmark}
            onExportResults={handleExportResults}
            results={[]}
            isRunning={false}
            progress={0}
          />
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Benchmark History</CardTitle>
              <CardDescription>
                Previous benchmark runs and their results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <table className="min-w-full divide-y divide-border">
                  <thead>
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">Date</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Dataset</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Algorithms</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Metrics</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-background">
                    <tr>
                      <td className="px-4 py-2 text-sm">2023-06-16 10:05</td>
                      <td className="px-4 py-2 text-sm">Benchmark Function 2</td>
                      <td className="px-4 py-2 text-sm">DE</td>
                      <td className="px-4 py-2 text-sm">RMSE, Convergence, Robustness</td>
                      <td className="px-4 py-2 text-sm">
                        <span className="inline-flex items-center rounded-full bg-yellow-100 px-2.5 py-0.5 text-xs font-medium text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                          Running
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm">
                        <Button variant="ghost" size="sm">View</Button>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 text-sm">2023-06-16 09:15</td>
                      <td className="px-4 py-2 text-sm">Benchmark Function 1</td>
                      <td className="px-4 py-2 text-sm">NM, PSO</td>
                      <td className="px-4 py-2 text-sm">RMSE, Convergence, Robustness</td>
                      <td className="px-4 py-2 text-sm">
                        <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900 dark:text-green-200">
                          Completed
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm">
                        <Button variant="ghost" size="sm">View</Button>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 text-sm">2023-06-15 15:42</td>
                      <td className="px-4 py-2 text-sm">Migraine Dataset B</td>
                      <td className="px-4 py-2 text-sm">GA, DE, ES</td>
                      <td className="px-4 py-2 text-sm">Accuracy, F1, RMSE, Convergence</td>
                      <td className="px-4 py-2 text-sm">
                        <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900 dark:text-green-200">
                          Completed
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm">
                        <Button variant="ghost" size="sm">View</Button>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 text-sm">2023-06-15 14:22</td>
                      <td className="px-4 py-2 text-sm">Migraine Dataset A</td>
                      <td className="px-4 py-2 text-sm">GA, DE, PSO, BO</td>
                      <td className="px-4 py-2 text-sm">Accuracy, F1, RMSE, Convergence</td>
                      <td className="px-4 py-2 text-sm">
                        <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800 dark:bg-green-900 dark:text-green-200">
                          Completed
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm">
                        <Button variant="ghost" size="sm">View</Button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
            <CardFooter>
              <div className="flex justify-between w-full">
                <Button variant="outline">Export All Results</Button>
                <Button>Configure New Benchmark</Button>
              </div>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 