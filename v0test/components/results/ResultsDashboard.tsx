"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { 
  Download, 
  Share2, 
  BarChart3, 
  LineChart, 
  ThumbsUp, 
  ThumbsDown, 
  Search,
  AlertCircle,
  Loader2,
  BarChart4,
  GitCompare
} from "lucide-react"

// Import our new visualization components
import { ComparativeVisualization } from "@/components/visualizations/ComparativeVisualization"
import { ConvergenceVisualization } from "@/components/visualizations/ConvergenceVisualization"

import { 
  listResults, 
  getOptimizationResult, 
  getBenchmarkResult, 
  exportResultAsCSV,
  compareResults,
  getResultVisualization,
  OptimizationResult,
  BenchmarkResult,
  ResultsComparison
} from "@/lib/api/results"

type Result = OptimizationResult | BenchmarkResult;

export interface ResultsDashboardProps {
  initialResultId?: string;
  resultType?: 'optimization' | 'benchmark' | 'all';
  showSearch?: boolean;
  showCompare?: boolean;
}

export function ResultsDashboard({ 
  initialResultId,
  resultType = 'all',
  showSearch = true,
  showCompare = true
}: ResultsDashboardProps) {
  // State for results list and selection
  const [results, setResults] = useState<Result[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(initialResultId || null);
  const [selectedResult, setSelectedResult] = useState<Result | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  
  // State for compare mode
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [selectedResultIds, setSelectedResultIds] = useState<string[]>(initialResultId ? [initialResultId] : []);
  const [comparisonResults, setComparisonResults] = useState<ResultsComparison | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  
  // Load results on mount
  useEffect(() => {
    loadResults();
  }, [resultType]);
  
  // Update selected result when selectedResultId changes
  useEffect(() => {
    if (selectedResultId) {
      loadResult(selectedResultId);
    } else {
      setSelectedResult(null);
    }
  }, [selectedResultId]);
  
  // Load results list
  const loadResults = async () => {
    setIsLoading(true);
    try {
      const data = await listResults(resultType);
      setResults(data);
      
      // If no result is selected and we have results, select the first one
      if (!selectedResultId && data.length > 0) {
        setSelectedResultId(data[0].id);
        setSelectedResultIds([data[0].id]);
      }
    } catch (error) {
      console.error("Error loading results:", error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load a specific result
  const loadResult = async (id: string) => {
    try {
      const result = results.find(r => r.id === id);
      if (result) {
        if (result.type === 'optimization') {
          const data = await getOptimizationResult(id);
          setSelectedResult(data);
        } else if (result.type === 'benchmark') {
          const data = await getBenchmarkResult(id);
          setSelectedResult(data);
        }
      }
    } catch (error) {
      console.error(`Error loading result ${id}:`, error);
    }
  };
  
  // Toggle result selection for comparison
  const toggleResultSelection = (id: string) => {
    if (selectedResultIds.includes(id)) {
      setSelectedResultIds(selectedResultIds.filter(resultId => resultId !== id));
    } else {
      setSelectedResultIds([...selectedResultIds, id]);
    }
  };
  
  // Start comparison mode
  const startCompare = () => {
    if (selectedResultIds.length > 0) {
      setCompareMode(true);
      compareSelectedResults();
    }
  };
  
  // Compare selected results
  const compareSelectedResults = async () => {
    if (selectedResultIds.length < 2) return;
    
    setIsComparing(true);
    try {
      const data = await compareResults(selectedResultIds);
      setComparisonResults(data);
    } catch (error) {
      console.error("Error comparing results:", error);
    } finally {
      setIsComparing(false);
    }
  };
  
  // Exit comparison mode
  const exitCompareMode = () => {
    setCompareMode(false);
    setComparisonResults(null);
  };
  
  // Export results as CSV
  const exportResults = async () => {
    if (selectedResultId) {
      try {
        await exportResultAsCSV(selectedResultId);
      } catch (error) {
        console.error("Error exporting results:", error);
      }
    }
  };
  
  // Filter results by search term
  const filteredResults = results.filter(result => 
    result.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (result.type === 'optimization' && 
      (result as OptimizationResult).algorithmName?.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (result.type === 'benchmark' && 
      result.benchmarkName?.toLowerCase().includes(searchTerm.toLowerCase()))
  );
  
  // Prepare visualization data for our new components
  const preparePerformanceData = () => {
    if (!selectedResult) return [];
    
    // For benchmark results with multiple algorithms
    if (selectedResult.type === 'benchmark' && (selectedResult as BenchmarkResult).algorithmResults) {
      return Object.entries((selectedResult as BenchmarkResult).algorithmResults).map(([algoId, results]) => ({
        algorithm: algoId,
        algorithmName: results.algorithmName || algoId,
        metrics: {
          fitness: results.bestFitness || 0,
          executionTime: results.executionTime || 0,
          iterations: results.iterations || 0,
          accuracy: results.accuracy || 0
        }
      }));
    }
    
    // For single optimization result
    if (selectedResult.type === 'optimization') {
      const optResult = selectedResult as OptimizationResult;
      return [{
        algorithm: optResult.algorithmId || 'unknown',
        algorithmName: optResult.algorithmName || 'Algorithm',
        metrics: {
          fitness: optResult.bestFitness || 0,
          executionTime: optResult.executionTime || 0,
          iterations: optResult.iterations || 0,
          accuracy: optResult.accuracy || 0
        }
      }];
    }
    
    return [];
  };
  
  const prepareConvergenceData = () => {
    if (!selectedResult) return [];
    
    // For benchmark results with multiple algorithms
    if (selectedResult.type === 'benchmark' && (selectedResult as BenchmarkResult).algorithmResults) {
      return Object.entries((selectedResult as BenchmarkResult).algorithmResults).map(([algoId, results]) => ({
        algorithmId: algoId,
        algorithmName: results.algorithmName || algoId,
        data: results.convergence || []
      }));
    }
    
    // For single optimization result
    if (selectedResult.type === 'optimization') {
      const optResult = selectedResult as OptimizationResult;
      return [{
        algorithmId: optResult.algorithmId || 'unknown',
        algorithmName: optResult.algorithmName || 'Algorithm',
        data: optResult.convergence || []
      }];
    }
    
    return [];
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">
          {compareMode ? 'Compare Results' : 'Results Dashboard'}
        </h2>
        <div className="flex space-x-2">
          {compareMode ? (
            <Button variant="outline" onClick={exitCompareMode}>
              Exit Compare
            </Button>
          ) : (
            showCompare && (
              <Button 
                variant="outline" 
                onClick={startCompare}
                disabled={selectedResultIds.length < 2}
              >
                <GitCompare className="h-4 w-4 mr-2" />
                Compare ({selectedResultIds.length})
              </Button>
            )
          )}
          <Button 
            variant="outline" 
            onClick={exportResults}
            disabled={!selectedResultId}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Results</CardTitle>
                {showSearch && (
                  <div className="relative">
                    <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search results..."
                      className="pl-8 h-9"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : filteredResults.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p>No results found</p>
                  {searchTerm && <p className="text-sm mt-1">Try changing your search</p>}
                </div>
              ) : (
                <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                  {filteredResults.map((result) => (
                    <div
                      key={result.id}
                      className={`p-3 rounded border cursor-pointer transition-colors ${
                        selectedResultId === result.id 
                          ? 'bg-primary/10 border-primary' 
                          : 'hover:bg-muted border-border'
                      }`}
                      onClick={() => setSelectedResultId(result.id)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-medium">{result.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {result.type === 'optimization' 
                              ? `${(result as OptimizationResult).algorithmName || 'Algorithm'}`
                              : `${result.benchmarkName || 'Benchmark'}`}
                            {result.type === 'optimization' && (result as OptimizationResult).datasetName && 
                              ` • ${(result as OptimizationResult).datasetName}`}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(result.createdAt).toLocaleString()}
                          </p>
                        </div>
                        {compareMode && (
                          <div 
                            className="h-4 w-4 rounded border flex items-center justify-center"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleResultSelection(result.id);
                            }}
                          >
                            {selectedResultIds.includes(result.id) && (
                              <div className="h-2 w-2 bg-primary rounded-sm" />
                            )}
                          </div>
                        )}
                      </div>
                      <div className="flex mt-2">
                        <Badge variant="outline" className="text-xs">
                          {result.type}
                        </Badge>
                        {result.status && (
                          <Badge 
                            className="ml-2 text-xs"
                            variant={result.status === 'completed' ? 'default' : 'secondary'}
                          >
                            {result.status}
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        <div className="md:col-span-2">
          {compareMode && comparisonResults ? (
            // Comparison view
            <Card>
              <CardHeader>
                <CardTitle>Results Comparison</CardTitle>
                <CardDescription>
                  Comparing {selectedResultIds.length} results
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="bg-muted p-4 rounded-md">
                  <h4 className="font-medium mb-2">Overall Comparison</h4>
                  <p className="text-sm mb-1">
                    <span className="font-medium">Best Algorithm:</span> {comparisonResults.bestAlgorithm}
                  </p>
                  
                  <div className="space-y-1 mt-4">
                    <p className="text-sm font-medium">Rankings by Metric:</p>
                    {Object.entries(comparisonResults.rankings).map(([metric, rankings]) => (
                      <p key={metric} className="text-sm ml-4">
                        <span className="font-medium">{metric}:</span> {(rankings as string[]).join(' > ')}
                      </p>
                    ))}
                  </div>
                  
                  <div className="mt-4">
                    <p className="text-sm font-medium mb-1">Statistical Significance:</p>
                    {comparisonResults.statisticalTests.map((test, index) => (
                      <p key={index} className="text-sm ml-4">
                        {test.metric} ({test.test}): {test.significant ? 
                          <span className="text-green-600">Significant (p={test.pValue.toFixed(4)})</span> : 
                          <span className="text-muted-foreground">Not significant (p={test.pValue.toFixed(4)})</span>
                        }
                      </p>
                    ))}
                  </div>
                </div>
                
                {/* Add our new visualization components */}
                <ComparativeVisualization 
                  data={comparisonResults.performanceData || []}
                  title="Performance Comparison"
                  description="Comparing algorithm performance across key metrics"
                  highlightedAlgorithm={comparisonResults.bestAlgorithm}
                />
                
                <ConvergenceVisualization 
                  series={comparisonResults.convergenceData || []}
                  title="Convergence Comparison"
                  description="How algorithm performance improved over iterations"
                  highlightedAlgorithm={comparisonResults.bestAlgorithm}
                  showBands={true}
                />
              </CardContent>
            </Card>
          ) : selectedResult ? (
            // Single result view
            <Tabs defaultValue="summary">
              <TabsList className="mb-4">
                <TabsTrigger value="summary">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Summary
                </TabsTrigger>
                <TabsTrigger value="performance">
                  <BarChart4 className="h-4 w-4 mr-2" />
                  Performance
                </TabsTrigger>
                <TabsTrigger value="convergence">
                  <LineChart className="h-4 w-4 mr-2" />
                  Convergence
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="summary">
                <Card>
                  <CardHeader>
                    <CardTitle>{selectedResult.name}</CardTitle>
                    <CardDescription>
                      {selectedResult.type === 'optimization' ? 'Optimization Result' : 'Benchmark Result'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-muted p-3 rounded-md">
                        <p className="text-sm text-muted-foreground">Algorithm</p>
                        <p className="font-medium">
                          {selectedResult.type === 'optimization'
                            ? (selectedResult as OptimizationResult).algorithmName || 'Algorithm'
                            : 'Multiple Algorithms'}
                        </p>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <p className="text-sm text-muted-foreground">Dataset</p>
                        <p className="font-medium">
                          {selectedResult.type === 'optimization'
                            ? (selectedResult as OptimizationResult).datasetName || 'Dataset'
                            : (selectedResult as BenchmarkResult).datasetName || 'Dataset'}
                        </p>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <p className="text-sm text-muted-foreground">Best Fitness</p>
                        <p className="font-medium">
                          {selectedResult.type === 'optimization'
                            ? ((selectedResult as OptimizationResult).bestFitness?.toFixed(6) || 'N/A')
                            : 'Multiple Results'}
                        </p>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <p className="text-sm text-muted-foreground">Execution Time</p>
                        <p className="font-medium">
                          {selectedResult.type === 'optimization'
                            ? ((selectedResult as OptimizationResult).executionTime?.toFixed(2) || 'N/A') + 's'
                            : 'Multiple Results'}
                        </p>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div>
                      <h4 className="text-sm font-medium mb-2">Configuration</h4>
                      <div className="bg-muted p-3 rounded-md">
                        <pre className="text-xs overflow-auto max-h-40">
                          {JSON.stringify(
                            selectedResult.type === 'optimization'
                              ? (selectedResult as OptimizationResult).configuration || {}
                              : {}, 
                            null, 2
                          )}
                        </pre>
                      </div>
                    </div>
                    
                    {selectedResult.type === 'optimization' && (selectedResult as OptimizationResult).bestSolution && (
                      <>
                        <Separator />
                        <div>
                          <h4 className="text-sm font-medium mb-2">Best Solution</h4>
                          <div className="bg-muted p-3 rounded-md">
                            <pre className="text-xs overflow-auto max-h-40">
                              {JSON.stringify((selectedResult as OptimizationResult).bestSolution, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="performance">
                <ComparativeVisualization 
                  data={preparePerformanceData()}
                  title="Algorithm Performance"
                  description={`Performance metrics for ${
                    selectedResult.type === 'optimization'
                      ? (selectedResult as OptimizationResult).algorithmName || 'algorithm'
                      : 'multiple algorithms'
                  }`}
                />
              </TabsContent>
              
              <TabsContent value="convergence">
                <ConvergenceVisualization 
                  series={prepareConvergenceData()}
                  title="Convergence Plot"
                  description={`Convergence of ${
                    selectedResult.type === 'optimization'
                      ? (selectedResult as OptimizationResult).algorithmName || 'algorithm'
                      : 'algorithms'
                  } over iterations`}
                />
              </TabsContent>
            </Tabs>
          ) : (
            // No result selected
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <AlertCircle className="h-12 w-12 text-muted-foreground/60 mb-4" />
                <p className="text-muted-foreground">No result selected</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Select a result from the list to view details
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
} 