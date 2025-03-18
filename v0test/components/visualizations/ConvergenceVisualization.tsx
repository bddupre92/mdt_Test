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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine,
  Area
} from "recharts"
import { ZoomIn, ZoomOut, RefreshCw, AlertCircle, Eye } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import { cn } from "@/lib/utils"

// Define types
export type ConvergenceSeries = {
  algorithmId: string;
  algorithmName: string;
  color?: string;
  data: {
    iteration: number;
    fitness: number;
    upperBound?: number;
    lowerBound?: number;
  }[];
}

export interface ConvergenceVisualizationProps {
  series: ConvergenceSeries[];
  title?: string;
  description?: string;
  showAxisLabels?: boolean;
  showLegend?: boolean;
  showControls?: boolean;
}

// Default colors for different algorithms
const ALGORITHM_COLORS = [
  "#1f77b4", // Blue
  "#ff7f0e", // Orange
  "#2ca02c", // Green
  "#d62728", // Red
  "#9467bd", // Purple
  "#8c564b", // Brown
  "#e377c2", // Pink
  "#7f7f7f", // Gray
  "#bcbd22", // Olive
  "#17becf", // Teal
]

export function ConvergenceVisualization({
  series = [],
  title = "Convergence Plot",
  description = "Algorithm performance over iterations",
  showAxisLabels = true,
  showLegend = true,
  showControls = true
}: ConvergenceVisualizationProps) {
  const [zoomLevel, setZoomLevel] = useState<number>(1)
  const [logarithmicScale, setLogarithmicScale] = useState<boolean>(false)
  const [showConfidenceBands, setShowConfidenceBands] = useState<boolean>(true)
  const [visibleAlgorithms, setVisibleAlgorithms] = useState<Record<string, boolean>>(
    // Initialize all algorithms as visible
    series.reduce((acc, s) => ({ ...acc, [s.algorithmId]: true }), {})
  )
  
  // Calculate the visible range of iterations based on zoom level
  const visibleRange = useMemo(() => {
    if (!series.length) return { min: 0, max: 100 }
    
    // Find the max number of iterations across all algorithms
    const maxIterations = Math.max(
      ...series.map(s => {
        const lastPoint = s.data[s.data.length - 1]
        return lastPoint ? lastPoint.iteration : 0
      })
    )
    
    // Calculate the range to display based on zoom level
    // Higher zoom = smaller range (focus on later iterations)
    const range = maxIterations / zoomLevel
    const min = Math.max(0, maxIterations - range)
    
    return { min, max: maxIterations }
  }, [series, zoomLevel])
  
  // Prepare data for chart
  const chartData = useMemo(() => {
    if (!series.length) return []
    
    // Collect all unique iteration points
    const iterations = new Set<number>()
    series.forEach(s => {
      s.data.forEach(point => {
        iterations.add(point.iteration)
      })
    })
    
    // Convert to array and sort
    const sortedIterations = Array.from(iterations).sort((a, b) => a - b)
    
    // Filter by visible range
    const filteredIterations = sortedIterations.filter(
      i => i >= visibleRange.min && i <= visibleRange.max
    )
    
    // Create data points for each iteration
    return filteredIterations.map(iteration => {
      const dataPoint: Record<string, any> = { iteration }
      
      // Add data for each algorithm
      series.forEach(s => {
        // Skip if algorithm is not visible
        if (!visibleAlgorithms[s.algorithmId]) return
        
        // Find the closest data point at or before this iteration
        const point = s.data.find(p => p.iteration === iteration)
        
        if (point) {
          // Use logarithmic scale if enabled
          const fitnessValue = logarithmicScale
            ? point.fitness > 0 ? Math.log10(point.fitness) : 0
            : point.fitness
          
          dataPoint[`${s.algorithmId}_fitness`] = fitnessValue
          
          // Add confidence bands if available
          if (showConfidenceBands && point.upperBound !== undefined) {
            dataPoint[`${s.algorithmId}_upper`] = logarithmicScale
              ? point.upperBound > 0 ? Math.log10(point.upperBound) : 0
              : point.upperBound
          }
          
          if (showConfidenceBands && point.lowerBound !== undefined) {
            dataPoint[`${s.algorithmId}_lower`] = logarithmicScale
              ? point.lowerBound > 0 ? Math.log10(point.lowerBound) : 0
              : point.lowerBound
          }
        }
      })
      
      return dataPoint
    })
  }, [series, visibleRange, logarithmicScale, showConfidenceBands, visibleAlgorithms])
  
  // Handle toggling visibility of algorithms
  const toggleAlgorithmVisibility = (algorithmId: string) => {
    setVisibleAlgorithms(prev => ({
      ...prev,
      [algorithmId]: !prev[algorithmId]
    }))
  }
  
  // Handle zoom controls
  const zoomIn = () => setZoomLevel(prev => Math.min(prev * 2, 16))
  const zoomOut = () => setZoomLevel(prev => Math.max(prev / 2, 1))
  const resetZoom = () => setZoomLevel(1)
  
  // Format the Y-axis label based on scale
  const formatYAxis = (value: number) => {
    if (logarithmicScale) {
      return `10^${value.toFixed(1)}`
    }
    return value.toFixed(2)
  }
  
  if (series.length === 0) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </CardHeader>
        <CardContent className="h-80">
          <Skeleton className="h-full w-full" />
        </CardContent>
      </Card>
    )
  }
  
  // No data to display
  if (series.length === 0 || chartData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="py-12 text-center text-muted-foreground flex flex-col items-center justify-center">
            <AlertCircle className="h-12 w-12 mb-4 text-muted-foreground/60" />
            <p>No convergence data available.</p>
            <p className="text-sm mt-1">Run algorithms to see convergence plots.</p>
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
          {showControls && (
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={zoomIn}
                title="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={zoomOut}
                title="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={resetZoom}
                title="Reset zoom"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col space-y-4">
          {showControls && (
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center space-x-2">
                <Switch
                  id="log-scale"
                  checked={logarithmicScale}
                  onCheckedChange={setLogarithmicScale}
                />
                <Label htmlFor="log-scale">Log scale</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="confidence-bands"
                  checked={showConfidenceBands}
                  onCheckedChange={setShowConfidenceBands}
                />
                <Label htmlFor="confidence-bands">Confidence bands</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // Toggle all to visible
                    const allVisible = Object.values(visibleAlgorithms).some(v => !v)
                    const newState = series.reduce(
                      (acc, s) => ({ ...acc, [s.algorithmId]: allVisible }),
                      {}
                    )
                    setVisibleAlgorithms(newState)
                  }}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  {Object.values(visibleAlgorithms).every(v => v)
                    ? "Hide all"
                    : "Show all"}
                </Button>
              </div>
            </div>
          )}
          
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis
                  dataKey="iteration"
                  type="number"
                  label={
                    showAxisLabels
                      ? {
                          value: "Iterations",
                          position: "bottom",
                          offset: 10,
                        }
                      : undefined
                  }
                  domain={[visibleRange.min, visibleRange.max]}
                />
                <YAxis
                  label={
                    showAxisLabels
                      ? {
                          value: logarithmicScale ? "Fitness (log)" : "Fitness",
                          angle: -90,
                          position: "insideLeft",
                        }
                      : undefined
                  }
                  tickFormatter={formatYAxis}
                />
                <Tooltip
                  formatter={(value: number, name: string) => {
                    // Extract algorithm ID from the name
                    const [algorithmId, type] = name.split("_")
                    const algorithm = series.find(s => s.algorithmId === algorithmId)
                    const algorithmName = algorithm?.algorithmName || algorithmId
                    
                    // Format value based on type
                    if (type === "fitness") {
                      return [
                        logarithmicScale ? `10^${value.toFixed(4)}` : value.toFixed(4),
                        `${algorithmName} Fitness`
                      ]
                    }
                    if (type === "upper") {
                      return [
                        logarithmicScale ? `10^${value.toFixed(4)}` : value.toFixed(4),
                        `${algorithmName} Upper Bound`
                      ]
                    }
                    if (type === "lower") {
                      return [
                        logarithmicScale ? `10^${value.toFixed(4)}` : value.toFixed(4),
                        `${algorithmName} Lower Bound`
                      ]
                    }
                    return [value, name]
                  }}
                  labelFormatter={(iteration) => `Iteration: ${iteration}`}
                />
                {showLegend && <Legend />}
                
                {/* Render series */}
                {series.map((s, index) => {
                  // Skip if algorithm is not visible
                  if (!visibleAlgorithms[s.algorithmId]) return null
                  
                  const color = s.color || ALGORITHM_COLORS[index % ALGORITHM_COLORS.length]
                  
                  return (
                    <React.Fragment key={s.algorithmId}>
                      {/* Confidence bands */}
                      {showConfidenceBands && (
                        <Area
                          type="monotone"
                          dataKey={`${s.algorithmId}_upper`}
                          stroke="none"
                          fill={color}
                          fillOpacity={0.1}
                          activeDot={false}
                          isAnimationActive={false}
                        />
                      )}
                      {showConfidenceBands && (
                        <Area
                          type="monotone"
                          dataKey={`${s.algorithmId}_lower`}
                          stroke="none"
                          fill={color}
                          fillOpacity={0.1}
                          activeDot={false}
                          isAnimationActive={false}
                        />
                      )}
                      
                      {/* Main line */}
                      <Line
                        type="monotone"
                        dataKey={`${s.algorithmId}_fitness`}
                        name={s.algorithmId}
                        stroke={color}
                        activeDot={{ r: 6 }}
                        isAnimationActive={false}
                        dot={false}
                        strokeWidth={2}
                      />
                    </React.Fragment>
                  )
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Algorithm selector */}
          {showControls && series.length > 1 && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2">
              {series.map((s, index) => {
                const color = s.color || ALGORITHM_COLORS[index % ALGORITHM_COLORS.length]
                const isVisible = visibleAlgorithms[s.algorithmId]
                
                return (
                  <div
                    key={s.algorithmId}
                    className={cn(
                      "flex items-center p-2 rounded-md cursor-pointer",
                      isVisible ? "bg-muted/50" : "bg-transparent"
                    )}
                    onClick={() => toggleAlgorithmVisibility(s.algorithmId)}
                  >
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: isVisible ? color : "#ccc" }}
                    />
                    <Checkbox
                      checked={isVisible}
                      onCheckedChange={() => toggleAlgorithmVisibility(s.algorithmId)}
                      className="mr-2"
                    />
                    <span className={cn("text-sm", !isVisible && "text-muted-foreground")}>
                      {s.algorithmName}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
} 