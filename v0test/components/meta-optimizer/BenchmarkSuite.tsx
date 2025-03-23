import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { BarChart, CheckCircle, CircleSlash, Clock, HelpCircle, RotateCw, Download, RefreshCw } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { 
  runMetaOptimizerBenchmark, 
  compareAlgorithms,
  getPerformanceImprovementSuggestions,
  BenchmarkRequest as MetaOptimizerBenchmarkRequest, 
  BenchmarkResult as ImportedBenchmarkResult, 
  AlgorithmComparisonResult
} from '@/lib/api/meta-optimizer/benchmark';
import { ProblemCharacteristics } from '@/lib/utils/meta-optimizer/problem-classification';

// Extend the imported benchmark request type
interface BenchmarkConfig extends Omit<MetaOptimizerBenchmarkRequest, 'function_name'> {
  datasetIds: string[];
  algorithmIds: string[];
  metrics: string[];
  replications: number;
  timeLimit: number;
  parallelRuns: number;
  problemCharacteristics?: ProblemCharacteristics;
}

// Define our local benchmark result type
interface BenchmarkResult {
  algorithmId: string;
  datasetId: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  executionTime: number;
  metrics: Record<string, number>;
}

// Update the suggestions type
interface OptimizationSuggestion {
  parameter: string;
  currentValue: string | number;
  suggestedValue: string | number;
  expectedImprovement: string;
  confidence: number;
}

export interface BenchmarkSuiteProps {
  availableDatasets: Array<{ id: string; name: string; description?: string }>;
  availableAlgorithms: Array<{ id: string; name: string; description?: string }>;
  availableMetrics: Array<{ id: string; name: string; description?: string }>;
  onRunBenchmark?: (config: BenchmarkConfig) => Promise<void>;
  onExportResults?: (format: 'csv' | 'json') => void;
  results?: BenchmarkResult[];
  isRunning?: boolean;
  progress?: number;
  problemCharacteristics?: ProblemCharacteristics;
}

// Add these interfaces at the top with the other interfaces
interface BenchmarkRequest {
  datasetIds: string[];
  algorithmIds: string[];
  metrics: string[];
  replications: number;
  timeLimit: number;
  parallelRuns: number;
  problemCharacteristics?: ProblemCharacteristics;
}

export default function BenchmarkSuite({
  availableDatasets = [],
  availableAlgorithms = [],
  availableMetrics = [],
  onRunBenchmark,
  onExportResults,
  results: initialResults = [],
  isRunning: initialIsRunning = false,
  progress: initialProgress = 0,
  problemCharacteristics
}: BenchmarkSuiteProps) {
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [replications, setReplications] = useState(5);
  const [timeLimit, setTimeLimit] = useState(60); // seconds
  const [parallelRuns, setParallelRuns] = useState(1);
  const [activeTab, setActiveTab] = useState('configuration');
  
  // State for API-related functionality
  const [isRunning, setIsRunning] = useState(initialIsRunning);
  const [progress, setProgress] = useState(initialProgress);
  const [results, setResults] = useState<BenchmarkResult[]>(initialResults);
  const [benchmarkId, setBenchmarkId] = useState<string>('');
  const [comparison, setComparison] = useState<AlgorithmComparisonResult | null>(null);
  const [suggestions, setSuggestions] = useState<Record<string, OptimizationSuggestion[]>>({});
  const [isLoading, setIsLoading] = useState<Record<string, boolean>>({
    comparison: false,
    suggestions: false,
    results: false
  });
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  // Helper to toggle selection in array
  const toggleSelection = (array: string[], item: string): string[] => {
    return array.includes(item)
      ? array.filter(i => i !== item)
      : [...array, item];
  };

  // Run benchmark with current configuration
  const runBenchmark = async () => {
    if (selectedDatasets.length === 0 || selectedAlgorithms.length === 0 || selectedMetrics.length === 0) {
      alert('Please select at least one dataset, algorithm, and metric');
      return;
    }

    const config: BenchmarkConfig = {
      datasetIds: selectedDatasets,
      algorithmIds: selectedAlgorithms,
      metrics: selectedMetrics,
      replications,
      timeLimit,
      parallelRuns,
      problemCharacteristics
    };

    try {
      setIsRunning(true);
      setProgress(0);
      setActiveTab('results');
      
      if (onRunBenchmark) {
        await onRunBenchmark(config);
      } else {
        // Convert to API format
        const benchmarkRequest: MetaOptimizerBenchmarkRequest = {
          ...config,
          function_name: 'run_benchmark'
        };
        
        const result = await runMetaOptimizerBenchmark(benchmarkRequest);
        setBenchmarkId(result.id);
        
        // Start progress polling
        startProgressPolling(result.id);
      }
    } catch (error) {
      console.error('Error running benchmark:', error);
      setIsRunning(false);
    }
  };
  
  // Start polling for benchmark progress
  const startProgressPolling = (id: string) => {
    // Clear any existing interval
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
    
    // Set up new polling interval
    const interval = setInterval(async () => {
      try {
        setIsLoading(prev => ({ ...prev, results: true }));
        // Fetch latest results (this would be a real API call)
        const latestResults = await fetchResults(id);
        setResults(latestResults);
        
        // Update progress
        const completedRuns = latestResults.filter(r => r.status === 'completed').length;
        const totalRuns = selectedDatasets.length * selectedAlgorithms.length * replications;
        const newProgress = Math.round((completedRuns / totalRuns) * 100);
        setProgress(newProgress);
        
        // Check if complete
        if (newProgress >= 100 || latestResults.every(r => r.status !== 'running')) {
          setIsRunning(false);
          clearInterval(interval);
          setRefreshInterval(null);
          
          // Get comparison if we have multiple algorithms
          if (selectedAlgorithms.length > 1) {
            fetchComparison(id);
          }
        }
      } catch (error) {
        console.error('Error polling benchmark progress:', error);
      } finally {
        setIsLoading(prev => ({ ...prev, results: false }));
      }
    }, 2000);
    
    setRefreshInterval(interval);
  };
  
  // Update the fetchResults function to handle type conversion
  const fetchResults = async (id: string): Promise<BenchmarkResult[]> => {
    try {
      // In a real implementation, this would call an API and convert the response
      // For now, just return the current results with updated progress
      return results.map(result => ({
        ...result,
        // Simulate progress by randomly completing some results
        status: result.status === 'running' && Math.random() > 0.3 ? 'completed' : result.status,
      }));
    } catch (error) {
      console.error('Error fetching results:', error);
      return [];
    }
  };
  
  // Fetch algorithm comparison
  const fetchComparison = async (id: string) => {
    try {
      setIsLoading(prev => ({ ...prev, comparison: true }));
      const comparisonResult = await compareAlgorithms(id);
      setComparison(comparisonResult);
    } catch (error) {
      console.error('Error fetching algorithm comparison:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, comparison: false }));
    }
  };
  
  // Fetch improvement suggestions for an algorithm
  const fetchSuggestions = async (algorithmId: string) => {
    if (!benchmarkId) return;
    
    try {
      setIsLoading(prev => ({ ...prev, suggestions: true }));
      const result = await getPerformanceImprovementSuggestions(benchmarkId, algorithmId);
      setSuggestions(prev => ({
        ...prev,
        [algorithmId]: result.suggestions
      }));
    } catch (error) {
      console.error('Error fetching improvement suggestions:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, suggestions: false }));
    }
  };
  
  // Export results to CSV or JSON
  const handleExportResults = (format: 'csv' | 'json') => {
    if (onExportResults) {
      onExportResults(format);
    } else {
      // Client-side export
      const data = format === 'json' 
        ? JSON.stringify(results, null, 2)
        : convertResultsToCSV(results);
      
      const blob = new Blob([data], { type: format === 'json' ? 'application/json' : 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `benchmark-results.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };
  
  // Convert results to CSV format
  const convertResultsToCSV = (resultsData: BenchmarkResult[]): string => {
    const headers = ['Algorithm', 'Dataset', 'Status', 'Execution Time'];
    
    // Add all metrics as headers
    const metricNames = new Set<string>();
    resultsData.forEach(result => {
      Object.keys(result.metrics).forEach(metric => metricNames.add(metric));
    });
    
    const allHeaders = [...headers, ...Array.from(metricNames)];
    
    // Create CSV content
    let csv = allHeaders.join(',') + '\n';
    
    resultsData.forEach(result => {
      const row = [
        result.algorithmId,
        result.datasetId,
        result.status,
        result.executionTime.toString()
      ];
      
      // Add metric values
      Array.from(metricNames).forEach(metric => {
        row.push((result.metrics[metric] || '').toString());
      });
      
      csv += row.join(',') + '\n';
    });
    
    return csv;
  };

  // Group results by algorithm for easier display
  const resultsByAlgorithm = results.reduce<Record<string, BenchmarkResult[]>>((acc, result) => {
    if (!acc[result.algorithmId]) {
      acc[result.algorithmId] = [];
    }
    acc[result.algorithmId].push(result);
    return acc;
  }, {});

  // Calculate statistics across runs
  const calculateStats = (algorithmId: string, metricId: string) => {
    const algorithmResults = resultsByAlgorithm[algorithmId] || [];
    const values = algorithmResults
      .filter(r => r.status === 'completed')
      .map(r => r.metrics[metricId] || 0);

    if (values.length === 0) return { mean: 0, stdDev: 0, min: 0, max: 0 };

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);

    return { mean, stdDev, min, max };
  };

  // Get status icon for benchmark result
  const getStatusIcon = (status: BenchmarkResult['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running':
        return <RotateCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <CircleSlash className="h-4 w-4 text-red-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-gray-500" />;
      default:
        return <HelpCircle className="h-4 w-4 text-gray-500" />;
    }
  };
  
  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [refreshInterval]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Benchmark Testing Suite</CardTitle>
        <CardDescription>
          Evaluate and compare algorithm performance across multiple datasets
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="configuration">Configuration</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="visualization">Visualization</TabsTrigger>
          </TabsList>

          <TabsContent value="configuration" className="space-y-6">
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">Select Datasets</h3>
                <div className="grid grid-cols-2 gap-2">
                  {availableDatasets.map(dataset => (
                    <div
                      key={dataset.id}
                      className="flex items-start space-x-2 p-2 border rounded-md"
                    >
                      <Checkbox
                        id={`dataset-${dataset.id}`}
                        checked={selectedDatasets.includes(dataset.id)}
                        onCheckedChange={() => setSelectedDatasets(prev => toggleSelection(prev, dataset.id))}
                      />
                      <div className="flex flex-col">
                        <Label
                          htmlFor={`dataset-${dataset.id}`}
                          className="font-medium"
                        >
                          {dataset.name}
                        </Label>
                        {dataset.description && (
                          <p className="text-sm text-muted-foreground">{dataset.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <h3 className="text-lg font-medium mb-2">Select Algorithms</h3>
                <div className="grid grid-cols-2 gap-2">
                  {availableAlgorithms.map(algorithm => (
                    <div
                      key={algorithm.id}
                      className="flex items-start space-x-2 p-2 border rounded-md"
                    >
                      <Checkbox
                        id={`algorithm-${algorithm.id}`}
                        checked={selectedAlgorithms.includes(algorithm.id)}
                        onCheckedChange={() => setSelectedAlgorithms(prev => toggleSelection(prev, algorithm.id))}
                      />
                      <div className="flex flex-col">
                        <Label
                          htmlFor={`algorithm-${algorithm.id}`}
                          className="font-medium"
                        >
                          {algorithm.name}
                        </Label>
                        {algorithm.description && (
                          <p className="text-sm text-muted-foreground">{algorithm.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <h3 className="text-lg font-medium mb-2">Metrics</h3>
                <div className="grid grid-cols-2 gap-2">
                  {availableMetrics.map(metric => (
                    <div
                      key={metric.id}
                      className="flex items-start space-x-2 p-2 border rounded-md"
                    >
                      <Checkbox
                        id={`metric-${metric.id}`}
                        checked={selectedMetrics.includes(metric.id)}
                        onCheckedChange={() => setSelectedMetrics(prev => toggleSelection(prev, metric.id))}
                      />
                      <div className="flex flex-col">
                        <Label
                          htmlFor={`metric-${metric.id}`}
                          className="font-medium"
                        >
                          {metric.name}
                        </Label>
                        {metric.description && (
                          <p className="text-sm text-muted-foreground">{metric.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="replications">Replications</Label>
                  <Input
                    id="replications"
                    type="number"
                    min={1}
                    max={50}
                    value={replications}
                    onChange={e => setReplications(parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Number of times to run each algorithm</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="timeLimit">Time Limit (seconds)</Label>
                  <Input
                    id="timeLimit"
                    type="number"
                    min={1}
                    max={3600}
                    value={timeLimit}
                    onChange={e => setTimeLimit(parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Maximum execution time per run</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="parallelRuns">Parallel Runs</Label>
                  <Input
                    id="parallelRuns"
                    type="number"
                    min={1}
                    max={16}
                    value={parallelRuns}
                    onChange={e => setParallelRuns(parseInt(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Number of concurrent executions</p>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button onClick={runBenchmark} disabled={isRunning}>
                {isRunning ? (
                  <>
                    <RotateCw className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  'Run Benchmark'
                )}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="results">
            {isRunning && (
              <div className="mb-6 space-y-2">
                <div className="flex justify-between items-center">
                  <h3 className="font-medium">Benchmark in progress</h3>
                  <span className="text-sm text-muted-foreground">{progress}% complete</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            )}
            
            {results.length > 0 ? (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-medium">Results Summary</h3>
                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleExportResults('csv')}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Export CSV
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleExportResults('json')}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Export JSON
                    </Button>
                  </div>
                </div>
                
                {comparison && (
                  <div className="bg-muted p-4 rounded-md mb-4">
                    <h4 className="font-medium mb-2">Algorithm Comparison</h4>
                    <div className="space-y-2">
                      <p className="text-sm">
                        <span className="font-medium">Best Overall Algorithm:</span> {comparison.bestAlgorithm}
                      </p>
                      
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Rankings by Metric:</p>
                        {Object.entries(comparison.rankings).map(([metric, rankings]) => (
                          <p key={metric} className="text-sm ml-4">
                            <span className="font-medium">{metric}:</span> {rankings.join(' > ')}
                          </p>
                        ))}
                      </div>
                      
                      <p className="text-sm">
                        <span className="font-medium">Statistical Significance:</span> {
                          comparison.statisticalTests.some(test => test.significant) 
                            ? 'Results show statistically significant differences' 
                            : 'No statistically significant differences detected'
                        }
                      </p>
                    </div>
                  </div>
                )}
                
                <div className="space-y-6">
                  {Object.entries(resultsByAlgorithm).map(([algorithmId, algorithmResults]) => {
                    const algorithmInfo = availableAlgorithms.find(a => a.id === algorithmId);
                    const totalRuns = algorithmResults.length;
                    const completedRuns = algorithmResults.filter(r => r.status === 'completed').length;
                    const failedRuns = algorithmResults.filter(r => r.status === 'error').length;
                    
                    return (
                      <div key={algorithmId} className="border rounded-md p-4">
                        <div className="flex justify-between mb-4">
                          <div>
                            <h4 className="text-lg font-medium">{algorithmInfo?.name || algorithmId}</h4>
                            <p className="text-sm text-muted-foreground">
                              {completedRuns} of {totalRuns} runs completed
                              {failedRuns > 0 && `, ${failedRuns} failed`}
                            </p>
                          </div>
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => fetchSuggestions(algorithmId)}
                            disabled={isLoading.suggestions}
                          >
                            {isLoading.suggestions ? (
                              <RotateCw className="mr-2 h-4 w-4 animate-spin" />
                            ) : (
                              <RefreshCw className="mr-2 h-4 w-4" />
                            )}
                            Get Improvement Suggestions
                          </Button>
                        </div>
                        
                        {suggestions[algorithmId] && (
                          <div className="bg-muted p-3 rounded-md mb-4">
                            <h5 className="font-medium mb-2">Improvement Suggestions</h5>
                            <ul className="space-y-1">
                              {suggestions[algorithmId].map((suggestion, idx) => (
                                <li key={idx} className="text-sm flex justify-between">
                                  <span>
                                    Change <span className="font-medium">{suggestion.parameter}</span> from {suggestion.currentValue} to {suggestion.suggestedValue}
                                  </span>
                                  <span className="text-xs">
                                    {suggestion.expectedImprovement} (confidence: {(suggestion.confidence * 100).toFixed(0)}%)
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        <div className="space-y-4">
                          <h5 className="font-medium">Performance Metrics</h5>
                          
                          <div className="space-y-3">
                            {selectedMetrics.map(metricId => {
                              const metricInfo = availableMetrics.find(m => m.id === metricId);
                              const stats = calculateStats(algorithmId, metricId);
                              
                              return (
                                <div key={metricId} className="space-y-1">
                                  <p className="text-sm font-medium">{metricInfo?.name || metricId}</p>
                                  <div className="grid grid-cols-4 gap-2 text-sm">
                                    <div className="bg-background p-2 rounded-md">
                                      <p className="text-xs text-muted-foreground">Mean</p>
                                      <p className="font-mono">{stats.mean.toFixed(4)}</p>
                                    </div>
                                    <div className="bg-background p-2 rounded-md">
                                      <p className="text-xs text-muted-foreground">Std Dev</p>
                                      <p className="font-mono">{stats.stdDev.toFixed(4)}</p>
                                    </div>
                                    <div className="bg-background p-2 rounded-md">
                                      <p className="text-xs text-muted-foreground">Min</p>
                                      <p className="font-mono">{stats.min.toFixed(4)}</p>
                                    </div>
                                    <div className="bg-background p-2 rounded-md">
                                      <p className="text-xs text-muted-foreground">Max</p>
                                      <p className="font-mono">{stats.max.toFixed(4)}</p>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <p>No benchmark results yet.</p>
                <p className="text-sm mt-1">Configure and run a benchmark to see results here.</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="visualization">
            {results.length > 0 ? (
              <div className="space-y-8">
                <div>
                  <h3 className="text-lg font-medium mb-4">Performance Comparison</h3>
                  <div className="h-64 bg-muted rounded-md flex items-center justify-center">
                    <p className="text-muted-foreground">[Bar Chart Visualization]</p>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-4">Convergence Comparison</h3>
                  <div className="h-64 bg-muted rounded-md flex items-center justify-center">
                    <p className="text-muted-foreground">[Line Chart Visualization]</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <p>No visualization data available.</p>
                <p className="text-sm mt-1">Run a benchmark first to see visualizations.</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 