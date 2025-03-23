"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Heatmap } from "./heatmap"

interface TuningResult {
  id: number
  params: Record<string, number | string | boolean>
  score: number
  time: number
  rank: number
}

interface HyperparameterTuningProps {
  tuningResults: TuningResult[]
  parameterRanges: Record<string, any[]>
  bestParams: Record<string, any>
  metricName: string
  modelType: string
}

export function HyperparameterTuning({
  tuningResults = [],
  parameterRanges = {},
  bestParams = {},
  metricName = "Score",
  modelType = "Unknown",
}: HyperparameterTuningProps) {
  const [visualizationType, setVisualizationType] = useState<string>("scatter")
  const [xAxis, setXAxis] = useState<string>("")
  const [yAxis, setYAxis] = useState<string>("")
  const [colorAxis, setColorAxis] = useState<string>("score")

  // Initialize x and y axis parameters based on available numeric parameters
  useEffect(() => {
    if (!parameterRanges || Object.keys(parameterRanges).length === 0) return

    const numericParams = Object.keys(parameterRanges).filter(
      (param) =>
        Array.isArray(parameterRanges[param]) &&
        parameterRanges[param].length > 0 &&
        typeof parameterRanges[param][0] === "number",
    )

    if (numericParams.length >= 2) {
      setXAxis(numericParams[0])
      setYAxis(numericParams[1])
    } else if (numericParams.length === 1) {
      setXAxis(numericParams[0])
      // Try to find any non-numeric parameter for y-axis
      const otherParams = Object.keys(parameterRanges).filter(
        (param) =>
          param !== numericParams[0] && Array.isArray(parameterRanges[param]) && parameterRanges[param].length > 0,
      )
      if (otherParams.length > 0) {
        setYAxis(otherParams[0])
      }
    }
  }, [parameterRanges])

  // Process data for visualization
  const processData = () => {
    if (!xAxis || !yAxis || !tuningResults || tuningResults.length === 0) return []

    return tuningResults.map((result) => ({
      x: result.params[xAxis],
      y: result.params[yAxis],
      z: colorAxis === "score" ? result.score : result.time,
      id: result.id,
      isBest: result.rank === 1,
      ...result,
    }))
  }

  const data = processData()

  // Format parameter value for display
  const formatParamValue = (value: any) => {
    if (value === undefined || value === null) return "N/A"
    if (typeof value === "boolean") return value ? "True" : "False"
    if (typeof value === "number") {
      return value > 0.01 ? value.toFixed(2) : value.toExponential(2)
    }
    return String(value)
  }

  // Generate heatmap data
  const generateHeatmapData = () => {
    if (!xAxis || !yAxis || !data || data.length === 0) {
      return { data: [], xLabels: [], yLabels: [] }
    }

    // Get unique x and y values
    const xValues = Array.from(new Set(data.map((d) => d.x))).sort((a: any, b: any) => {
      if (typeof a === "number" && typeof b === "number") return a - b
      return String(a).localeCompare(String(b))
    })

    const yValues = Array.from(new Set(data.map((d) => d.y))).sort((a: any, b: any) => {
      if (typeof a === "number" && typeof b === "number") return a - b
      return String(a).localeCompare(String(b))
    })

    // Create grid
    const grid: any[] = []

    for (let yIdx = 0; yIdx < yValues.length; yIdx++) {
      const row = []
      const y = yValues[yIdx]

      for (let xIdx = 0; xIdx < xValues.length; xIdx++) {
        const x = xValues[xIdx]

        // Find matching data point
        const point = data.find((d) => d.x === x && d.y === y)

        row.push({
          x,
          y,
          value: point ? point.z : null,
          isBest: point?.isBest || false,
        })
      }

      grid.push(row)
    }

    return {
      data: grid,
      xLabels: xValues,
      yLabels: yValues,
    }
  }

  const heatmapData = generateHeatmapData()

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Hyperparameter Tuning: {modelType}</CardTitle>
        <CardDescription>Visualization of hyperparameter tuning results</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-2">
            <Label htmlFor="visualization-type">Visualization Type</Label>
            <Select value={visualizationType} onValueChange={setVisualizationType}>
              <SelectTrigger id="visualization-type" className="w-[180px]">
                <SelectValue placeholder="Scatter plot" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="scatter">Scatter plot</SelectItem>
                <SelectItem value="heatmap">Heatmap</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="x-axis">X Axis</Label>
            <Select value={xAxis} onValueChange={setXAxis}>
              <SelectTrigger id="x-axis" className="w-[150px]">
                <SelectValue placeholder="Select parameter" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(parameterRanges).map((param) => (
                  <SelectItem key={param} value={param}>
                    {param}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="y-axis">Y Axis</Label>
            <Select value={yAxis} onValueChange={setYAxis}>
              <SelectTrigger id="y-axis" className="w-[150px]">
                <SelectValue placeholder="Select parameter" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(parameterRanges).map((param) => (
                  <SelectItem key={param} value={param}>
                    {param}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="color-axis">Color By</Label>
            <Select value={colorAxis} onValueChange={setColorAxis}>
              <SelectTrigger id="color-axis" className="w-[150px]">
                <SelectValue placeholder="Score" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="score">{metricName}</SelectItem>
                <SelectItem value="time">Execution Time</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="h-[400px]">
          {data.length > 0 ? (
            <>
              {visualizationType === "scatter" && xAxis && yAxis && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart
                    margin={{
                      top: 20,
                      right: 20,
                      bottom: 20,
                      left: 20,
                    }}
                  >
                    <CartesianGrid />
                    <XAxis
                      type="category"
                      dataKey="x"
                      name={xAxis}
                      label={{ value: xAxis, position: "insideBottom", offset: -5 }}
                    />
                    <YAxis
                      type="category"
                      dataKey="y"
                      name={yAxis}
                      label={{ value: yAxis, angle: -90, position: "insideLeft" }}
                    />
                    <ZAxis
                      type="number"
                      dataKey="z"
                      range={[20, 500]}
                      name={colorAxis === "score" ? metricName : "Time (s)"}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      formatter={(value: any, name: string) => {
                        if (name === "x") return [formatParamValue(value), xAxis]
                        if (name === "y") return [formatParamValue(value), yAxis]
                        if (name === "z")
                          return [
                            typeof value === "number" ? value.toFixed(4) : value,
                            colorAxis === "score" ? metricName : "Time (s)",
                          ]
                        return [value, name]
                      }}
                    />
                    <Legend />
                    <Scatter
                      name={colorAxis === "score" ? metricName : "Time (s)"}
                      data={data}
                      fill="#8884d8"
                      shape={(props: any) => {
                        const { cx, cy, fill } = props
                        const isBest = props.payload?.isBest
                        return (
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isBest ? 8 : 6}
                            fill={isBest ? "#ff7300" : fill}
                            stroke={isBest ? "#000" : "none"}
                            strokeWidth={isBest ? 1 : 0}
                          />
                        )
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              )}

              {visualizationType === "heatmap" && xAxis && yAxis && (
                <Heatmap
                  data={heatmapData.data}
                  xLabels={heatmapData.xLabels}
                  yLabels={heatmapData.yLabels}
                  xLabel={xAxis}
                  yLabel={yAxis}
                  valueLabel={colorAxis === "score" ? metricName : "Time (s)"}
                />
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full bg-muted rounded-md">
              <p className="text-muted-foreground">No hyperparameter tuning data available</p>
            </div>
          )}
        </div>

        <div className="mt-6 space-y-4">
          <h3 className="text-sm font-medium">Best Hyperparameters</h3>
          {Object.keys(bestParams).length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {Object.entries(bestParams).map(([param, value]) => (
                <div key={param} className="bg-muted rounded-lg p-3">
                  <p className="text-xs text-muted-foreground truncate">{param}</p>
                  <p className="text-sm font-semibold truncate">{formatParamValue(value)}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-muted rounded-lg p-3">
              <p className="text-sm text-muted-foreground">No best parameters available</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

