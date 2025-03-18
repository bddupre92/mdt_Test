"use client"

import React, { useState, useMemo } from "react"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart3, LineChart as LineChartIcon, GitCompare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { Sliders, BarChart2, ArrowDownUp } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { cn } from "@/lib/utils"

// Define types
export type AlgorithmPerformance = {
  algorithm: string;
  algorithmName?: string;
  metrics: {
    [key: string]: number;
  };
  details?: Record<string, any>;
}

export type ComparisonMetric = {
  key: string;
  name: string;
  description: string;
  unit?: string;
  higherIsBetter: boolean;
  formatter?: (value: number) => string;
}

export interface ComparativeVisualizationProps {
  data: AlgorithmPerformance[];
  metrics?: ComparisonMetric[];
  title?: string;
  description?: string;
  highlightedAlgorithm?: string;
  onMetricSelect?: (metric: string) => void;
  isLoading?: boolean;
}

const defaultMetrics: ComparisonMetric[] = [
  {
    key: "fitness",
    name: "Fitness",
    description: "Solution quality (lower is better)",
    higherIsBetter: false,
    formatter: (v) => v.toFixed(6),
  },
  {
    key: "executionTime",
    name: "Execution Time",
    description: "Time taken to complete (seconds)",
    unit: "s",
    higherIsBetter: false,
    formatter: (v) => v.toFixed(2),
  },
  {
    key: "iterations",
    name: "Iterations",
    description: "Number of iterations performed",
    higherIsBetter: false,
    formatter: (v) => v.toLocaleString(),
  },
  {
    key: "accuracy",
    name: "Accuracy",
    description: "Prediction accuracy percentage",
    unit: "%",
    higherIsBetter: true,
    formatter: (v) => v.toFixed(2),
  },
]

// Helper function to get the best algorithm based on a metric
const getBestAlgorithm = (data: AlgorithmPerformance[], metric: string, isLowerBetter = false) => {
  if (!data.length) return null;
  
  return data.reduce((best, current) => {
    const currentValue = current.metrics[metric] || 0;
    const bestValue = best.metrics[metric] || 0;
    
    if (isLowerBetter) {
      return currentValue < bestValue ? current : best;
    }
    return currentValue > bestValue ? current : best;
  }, data[0]);
};

// Helper function to normalize metric values
const normalizeValues = (data: AlgorithmPerformance[], metric: string, isLowerBetter = false) => {
  if (!data.length) return [];
  
  // Find min and max values
  const values = data.map(item => item.metrics[metric] || 0);
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  // If all values are the same, return equal normalized values
  if (min === max) {
    return data.map(item => ({
      ...item,
      normalizedValue: 1,
      percentOfBest: 100
    }));
  }
  
  return data.map(item => {
    const value = item.metrics[metric] || 0;
    // Normalize to 0-1 range
    let normalizedValue = isLowerBetter
      ? (max - value) / (max - min)
      : (value - min) / (max - min);
    
    // Calculate percent of best
    const best = isLowerBetter ? min : max;
    let percentOfBest = isLowerBetter
      ? (best / value) * 100
      : (value / best) * 100;
    
    // Cap at 100%
    percentOfBest = Math.min(percentOfBest, 100);
    
    return {
      ...item,
      normalizedValue,
      percentOfBest,
      rawValue: value
    };
  });
};

export function ComparativeVisualization({
  data = [],
  metrics = defaultMetrics,
  title = "Algorithm Comparison",
  description = "Side-by-side comparison of algorithm performance across key metrics",
  highlightedAlgorithm,
  onMetricSelect,
  isLoading = false,
}: ComparativeVisualizationProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>("fitness")
  const [normalizeValues, setNormalizeValues] = useState<boolean>(true)
  const [viewMode, setViewMode] = useState<string>("bars")
  const [sortBy, setSortBy] = useState<string>("performance")
  
  // Get current metric definition
  const currentMetric = useMemo(() => {
    return metrics.find(m => m.key === selectedMetric) || metrics[0]
  }, [metrics, selectedMetric])
  
  // Normalize and sort data
  let processedData = normalizeValues(data, selectedMetric, currentMetric.higherIsBetter)
  
  // Sort data based on user selection
  if (sortBy === "performance") {
    processedData = processedData.sort((a, b) => {
      return currentMetric.higherIsBetter
        ? (a.rawValue || 0) - (b.rawValue || 0)
        : (b.rawValue || 0) - (a.rawValue || 0)
    })
  } else if (sortBy === "alphabetical") {
    processedData = processedData.sort((a, b) => {
      return (a.algorithmName || a.algorithm).localeCompare(b.algorithmName || b.algorithm)
    })
  }
  
  // Find best algorithm
  const bestAlgorithm = getBestAlgorithm(data, selectedMetric, currentMetric.higherIsBetter)
  
  // Prepare data for bar chart
  const chartData = processedData.map(item => ({
    name: item.algorithmName || item.algorithm,
    value: item.rawValue || 0,
    normalizedValue: item.normalizedValue || 0,
    isBest: bestAlgorithm && bestAlgorithm.algorithm === item.algorithm
  }))
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        </CardContent>
      </Card>
    )
  }
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className="flex space-x-2">
            <ToggleGroup type="single" value={viewMode} onValueChange={(value) => value && setViewMode(value)}>
              <ToggleGroupItem value="bars" aria-label="View as bars">
                <BarChart2 className="h-4 w-4" />
              </ToggleGroupItem>
              <ToggleGroupItem value="differences" aria-label="View as differences">
                <GitCompare className="h-4 w-4" />
              </ToggleGroupItem>
            </ToggleGroup>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <Sliders className="h-4 w-4 mr-2" /> Metric
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                {metrics.map(metric => (
                  <DropdownMenuItem 
                    key={metric.key}
                    onClick={() => setSelectedMetric(metric.key)}
                    className={cn(selectedMetric === metric.key && "font-bold")}
                  >
                    {metric.name}
                    {metric.higherIsBetter ? " (Lower is better)" : " (Higher is better)"}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <ArrowDownUp className="h-4 w-4 mr-2" /> Sort
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem 
                  onClick={() => setSortBy("performance")}
                  className={cn(sortBy === "performance" && "font-bold")}
                >
                  By Performance
                </DropdownMenuItem>
                <DropdownMenuItem 
                  onClick={() => setSortBy("alphabetical")}
                  className={cn(sortBy === "alphabetical" && "font-bold")}
                >
                  Alphabetically
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="normalized" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="normalized">Normalized View</TabsTrigger>
            <TabsTrigger value="barchart">Bar Chart</TabsTrigger>
            {viewMode === "differences" && (
              <TabsTrigger value="differences">Differences</TabsTrigger>
            )}
          </TabsList>
          
          <TabsContent value="normalized" className="space-y-4">
            {processedData.map((item, index) => {
              const isBest = bestAlgorithm && bestAlgorithm.algorithm === item.algorithm
              
              return (
                <div key={item.algorithm} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="font-medium text-sm w-48 truncate">
                        {item.algorithmName || item.algorithm}
                      </span>
                      {isBest && (
                        <Badge variant="secondary" className="ml-2">Best</Badge>
                      )}
                    </div>
                    <span className="text-sm">
                      {item.higherIsBetter
                        ? `${(item.rawValue || 0).toFixed(4)} (${item.percentOfBest?.toFixed(1)}%)`
                        : `${(item.rawValue || 0).toFixed(4)} (${item.percentOfBest?.toFixed(1)}%)`}
                    </span>
                  </div>
                  <Progress 
                    value={item.normalizedValue ? item.normalizedValue * 100 : 0} 
                    className={cn(
                      "h-2",
                      isBest ? "bg-blue-100" : "bg-gray-100",
                    )}
                    indicatorClassName={isBest ? "bg-blue-500" : undefined}
                  />
                </div>
              )
            })}
          </TabsContent>
          
          <TabsContent value="barchart">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45} 
                    textAnchor="end" 
                    height={70}
                  />
                  <YAxis
                    label={{
                      value: currentMetric.name,
                      angle: -90,
                      position: 'insideLeft',
                      style: { textAnchor: 'middle' }
                    }}
                  />
                  <Tooltip
                    formatter={(value) => [
                      `${Number(value).toFixed(4)}`,
                      currentMetric.name
                    ]}
                  />
                  <Bar dataKey="value">
                    {chartData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.isBest ? '#3b82f6' : '#94a3b8'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          {viewMode === "differences" && (
            <TabsContent value="differences" className="space-y-4">
              {processedData.length > 0 && (
                <div className="space-y-6">
                  <div className="font-medium">Performance relative to best algorithm</div>
                  {processedData.map((item, index) => {
                    const isBest = bestAlgorithm && bestAlgorithm.algorithm === item.algorithm
                    const bestValue = bestAlgorithm ? (bestAlgorithm.metrics[selectedMetric] || 0) : 0
                    const currentValue = item.metrics[selectedMetric] || 0
                    
                    let difference = 0
                    let percentDifference = 0
                    
                    if (currentMetric.higherIsBetter) {
                      difference = currentValue - bestValue
                      percentDifference = bestValue !== 0 
                        ? ((currentValue - bestValue) / bestValue) * 100 
                        : 0
                    } else {
                      difference = bestValue - currentValue
                      percentDifference = bestValue !== 0 
                        ? ((bestValue - currentValue) / bestValue) * 100 
                        : 0
                    }
                    
                    // Skip the best algorithm (difference is 0)
                    if (isBest) return null
                    
                    return (
                      <div key={item.algorithm} className="bg-muted/50 p-3 rounded-md">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-sm">
                            {item.algorithmName || item.algorithm}
                          </span>
                          <span className="text-sm text-red-500">
                            {currentMetric.higherIsBetter
                              ? `+${difference.toFixed(4)} (+${percentDifference.toFixed(1)}%)`
                              : `-${difference.toFixed(4)} (-${percentDifference.toFixed(1)}%)`}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>
                            {item.algorithmName || item.algorithm}: {currentValue.toFixed(4)}
                          </span>
                          <span>vs</span>
                          <span>
                            {bestAlgorithm?.algorithmName || bestAlgorithm?.algorithm}: {bestValue.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    )
                  }).filter(Boolean)}
                </div>
              )}
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
    </Card>
  )
} 