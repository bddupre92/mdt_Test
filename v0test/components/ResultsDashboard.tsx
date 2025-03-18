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
  Loader2 
} from "lucide-react"
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
import { ResultsPanel } from "./results-panel"

type Result = OptimizationResult | BenchmarkResult;

interface ResultsDashboardProps {
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
  // State for results list
  const [results, setResults] = useState<Result[]>([]);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(initialResultId || null);
  const [selectedResult, setSelectedResult] = useState<Result | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // State for comparison
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [selectedResultIds, setSelectedResultIds] = useState<string[]>([]);
  const [comparisonResults, setComparisonResults] = useState<ResultsComparison | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState<boolean>(false);
  
  // State for search filtering
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filteredResults, setFilteredResults] = useState<Result[]>([]);
  
  // Fetch results on component mount
  useEffect(() => {
    const fetchResults = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await listResults(resultType);
        setResults(data);
        setFilteredResults(data);
        
        // If we have results and initialResultId not provided, select the first one
        if (data.length > 0 && !initialResultId) {
          setSelectedResultId(data[0].id);
        }
      } catch (err) {
        setError('Failed to load results. Please try again later.');
        console.error('Error loading results:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchResults();
  }, [initialResultId, resultType]);
  
  // Filter results when search query changes
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredResults(results);
      return;
    }
    
    const query = searchQuery.toLowerCase();
    const filtered = results.filter(result => {
      // Search in ID
      if (result.id.toLowerCase().includes(query)) return true;
      
      // Search in algorithm name if available
      if ('algorithmId' in result && result.algorithmId.toLowerCase().includes(query)) return true;
      
      // Search in dataset name if available
      if ('datasetId' in result && result.datasetId?.toLowerCase().includes(query)) return true;
      
      // Search in algorithms array if it's a benchmark result
      if ('algorithms' in result && result.algorithms.some(alg => alg.toLowerCase().includes(query))) return true;
      
      return false;
    });
    
    setFilteredResults(filtered);
  }, [searchQuery, results]);
  
  // Fetch selected result when ID changes
  useEffect(() => {
    const fetchSelectedResult = async () => {
      if (!selectedResultId) {
        setSelectedResult(null);
        return;
      }
      
      setIsLoading(true);
      setError(null);
      
      try {
        // Try fetching as optimization result first
        try {
          const result = await getOptimizationResult(selectedResultId);
          setSelectedResult(result);
          return;
        } catch (err) {
          // If that fails, try fetching as benchmark result
          const result = await getBenchmarkResult(selectedResultId);
          setSelectedResult(result);
        }
      } catch (err) {
        setError(`Failed to load result ${selectedResultId}.`);
        console.error(`Error loading result ${selectedResultId}:`, err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchSelectedResult();
  }, [selectedResultId]);
  
  // Handle result selection
  const handleSelectResult = (resultId: string) => {
    if (compareMode) {
      // Toggle selection for comparison
      if (selectedResultIds.includes(resultId)) {
        setSelectedResultIds(selectedResultIds.filter(id => id !== resultId));
      } else {
        setSelectedResultIds([...selectedResultIds, resultId]);
      }
    } else {
      // Regular selection
      setSelectedResultId(resultId);
    }
  };
  
  // Handle comparing results
  const handleCompareResults = async () => {
    if (selectedResultIds.length < 2) {
      setError('Please select at least two results to compare.');
      return;
    }
    
    setComparisonLoading(true);
    
    try {
      const comparisonData = await compareResults(selectedResultIds);
      setComparisonResults(comparisonData);
    } catch (err) {
      setError('Failed to compare results. Please try again.');
      console.error('Error comparing results:', err);
    } finally {
      setComparisonLoading(false);
    }
  };
  
  // Handle exporting result as CSV
  const handleExportCSV = async () => {
    if (!selectedResultId) return;
    
    try {
      const csvData = await exportResultAsCSV(selectedResultId);
      
      // Create a blob and download
      const blob = new Blob([csvData], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `result-${selectedResultId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to export result as CSV.');
      console.error('Error exporting result:', err);
    }
  };
  
  // Toggle compare mode
  const toggleCompareMode = () => {
    setCompareMode(!compareMode);
    setSelectedResultIds([]);
    setComparisonResults(null);
  };
  
  // Render loading state
  if (isLoading && !selectedResult && !results.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
        <p className="text-muted-foreground">Loading results...</p>
      </div>
    );
  }
  
  // Render error state
  if (error && !selectedResult && !results.length) {
    return (
      <div className="border border-red-200 bg-red-50 p-6 rounded-md text-center">
        <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-4" />
        <p className="text-red-600 mb-4">{error}</p>
        <Button 
          variant="outline" 
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </div>
    );
  }
  
  // Render empty state
  if (!isLoading && results.length === 0) {
    return (
      <div className="border border-dashed rounded-lg p-12 text-center">
        <h3 className="font-medium text-lg mb-2">No Results Available</h3>
        <p className="text-muted-foreground mb-6">
          Run an optimization or benchmark to see results here.
        </p>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Header with search and actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Results Dashboard</h2>
        
        <div className="flex space-x-2">
          {showCompare && (
            <Button 
              variant={compareMode ? "default" : "outline"} 
              onClick={toggleCompareMode}
            >
              {compareMode ? "Exit Compare" : "Compare Results"}
            </Button>
          )}
        </div>
      </div>
      
      {/* Search bar */}
      {showSearch && (
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search results..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      )}
      
      {/* Main content grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Results list */}
        <div className="md:col-span-1 space-y-4">
          <h3 className="font-medium text-lg">Available Results</h3>
          
          {compareMode && (
            <div className="bg-muted p-3 rounded-md">
              <p className="text-sm mb-2">Select results to compare</p>
              <div className="flex justify-between">
                <p className="text-sm text-muted-foreground">
                  Selected: {selectedResultIds.length}
                </p>
                <Button 
                  size="sm" 
                  onClick={handleCompareResults}
                  disabled={selectedResultIds.length < 2 || comparisonLoading}
                >
                  {comparisonLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : null}
                  Compare
                </Button>
              </div>
            </div>
          )}
          
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {filteredResults.map(result => {
              const isSelected = compareMode 
                ? selectedResultIds.includes(result.id)
                : selectedResultId === result.id;
              
              // Determine result type and show relevant info
              const isOptimizationResult = 'algorithmId' in result;
              
              return (
                <div
                  key={result.id}
                  className={`border rounded-md p-3 cursor-pointer transition-colors ${
                    isSelected 
                      ? 'border-primary bg-primary/5' 
                      : 'hover:border-primary/50'
                  }`}
                  onClick={() => handleSelectResult(result.id)}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-medium truncate" style={{ maxWidth: '180px' }}>
                          {isOptimizationResult 
                            ? `${(result as OptimizationResult).algorithmId}`
                            : `Benchmark (${(result as BenchmarkResult).algorithms.length} algorithms)`}
                        </span>
                        <Badge variant="outline" className={
                          result.status === 'completed'
                            ? 'bg-green-50 text-green-600 border-green-200'
                            : result.status === 'error'
                            ? 'bg-red-50 text-red-600 border-red-200'
                            : 'bg-yellow-50 text-yellow-600 border-yellow-200'
                        }>
                          {result.status}
                        </Badge>
                      </div>
                      
                      <p className="text-xs text-muted-foreground mt-1">
                        ID: {result.id.substring(0, 8)}...
                      </p>
                      
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(result.createdAt).toLocaleString()}
                      </p>
                    </div>
                    
                    {compareMode && isSelected && (
                      <div className="bg-primary text-primary-foreground rounded-full p-1">
                        <ThumbsUp className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Results details */}
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
                    {comparisonResults.statisticalTests.map((test, index: number) => (
                      <p key={index} className="text-sm ml-4">
                        {test.metric} ({test.test}): {test.significant ? 
                          <span className="text-green-600">Significant (p={test.pValue.toFixed(4)})</span> : 
                          <span className="text-muted-foreground">Not significant (p={test.pValue.toFixed(4)})</span>
                        }
                      </p>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-3">Recommendations</h4>
                  <ul className="space-y-2">
                    {comparisonResults.recommendations.map((rec: string, index: number) => (
                      <li key={index} className="text-sm flex">
                        <span className="mr-2">•</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          ) : !compareMode && selectedResult ? (
            // Single result view
            <ResultsPanel 
              results={selectedResult} 
              onExport={handleExportCSV} 
              onShare={() => {}}
            />
          ) : (
            // No selection or loading state
            <div className="border rounded-lg flex items-center justify-center h-96">
              <div className="text-center p-6">
                {isLoading ? (
                  <>
                    <Loader2 className="h-10 w-10 animate-spin mx-auto mb-4 text-primary" />
                    <p>Loading result details...</p>
                  </>
                ) : compareMode ? (
                  <>
                    <p className="text-muted-foreground mb-2">Select two or more results to compare</p>
                    <Button 
                      size="sm" 
                      onClick={handleCompareResults}
                      disabled={selectedResultIds.length < 2}
                    >
                      Compare Selected
                    </Button>
                  </>
                ) : (
                  <p className="text-muted-foreground">Select a result to view details</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 