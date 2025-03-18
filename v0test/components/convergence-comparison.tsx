"use client"

import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { useState } from "react"

interface ConvergenceData {
  iteration: number
  [key: string]: number
}

interface ConvergenceComparisonProps {
  algorithmResults: {
    [key: string]: {
      name: string
      convergence: Array<{ iteration: number; fitness: number }>
    }
  }
  benchmarkName: string
}

export function ConvergenceComparison({
  algorithmResults = {},
  benchmarkName = "Unknown",
}: ConvergenceComparisonProps) {
  const [useLogScale, setUseLogScale] = useState(false)
  const [normalizeCurves, setNormalizeCurves] = useState(true)
  const [maxIterations, setMaxIterations] = useState<string>("all")

  // Process data for the chart
  const processData = (): ConvergenceData[] => {
    if (!algorithmResults || Object.keys(algorithmResults).length === 0) {
      return [{ iteration: 0 }]
    }

    // Get max iterations across all algorithms
    const maxIter = Math.max(
      ...Object.values(algorithmResults).map((result) =>
        result.convergence && result.convergence.length > 0
          ? result.convergence[result.convergence.length - 1].iteration
          : 0,
      ),
    )

    // Get the number of iterations to display
    const iterationsToDisplay = maxIterations === "all" ? maxIter : Math.min(Number.parseInt(maxIterations), maxIter)

    // Create an array of iteration points
    const data: ConvergenceData[] = Array.from({ length: iterationsToDisplay }, (_, i) => ({
      iteration: i + 1,
    }))

    // For each algorithm, add its fitness at each iteration
    Object.entries(algorithmResults).forEach(([key, result]) => {
      if (!result.convergence || result.convergence.length === 0) return

      // Find baseline (first value) for normalization
      const baseline = result.convergence[0]?.fitness || 1

      // Add fitness values for each iteration
      result.convergence.forEach((point) => {
        if (point.iteration <= iterationsToDisplay) {
          const value = normalizeCurves ? point.fitness / baseline : point.fitness

          data[point.iteration - 1][key] = useLogScale ? Math.log10(Math.max(0.000001, value)) : value
        }
      })
    })

    return data
  }

  const chartData = processData()

  // Generate configuration for each algorithm
  const generateChartConfig = () => {
    const config: any = {}

    Object.entries(algorithmResults).forEach(([key, result], index) => {
      config[key] = {
        label: result.name || `Algorithm ${index + 1}`,
        color: `hsl(var(--chart-${(index % 9) + 1}))`,
      }
    })

    return config
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Convergence Comparison: {benchmarkName}</CardTitle>
        <CardDescription>Compare convergence behavior across optimization algorithms</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-2">
            <Label htmlFor="max-iterations">Max Iterations</Label>
            <Select value={maxIterations} onValueChange={setMaxIterations}>
              <SelectTrigger id="max-iterations" className="w-[150px]">
                <SelectValue placeholder="All iterations" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All iterations</SelectItem>
                <SelectItem value="10">10 iterations</SelectItem>
                <SelectItem value="25">25 iterations</SelectItem>
                <SelectItem value="50">50 iterations</SelectItem>
                <SelectItem value="100">100 iterations</SelectItem>
                <SelectItem value="250">250 iterations</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center space-x-2">
            <Switch id="log-scale" checked={useLogScale} onCheckedChange={setUseLogScale} />
            <Label htmlFor="log-scale">Log scale</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch id="normalize" checked={normalizeCurves} onCheckedChange={setNormalizeCurves} />
            <Label htmlFor="normalize">Normalize curves</Label>
          </div>
        </div>

        <div className="h-[400px]">
          {Object.keys(algorithmResults).length > 0 ? (
            <ChartContainer config={generateChartConfig()} className="h-full w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="iteration"
                    label={{
                      value: "Iteration",
                      position: "insideBottomRight",
                      offset: -10,
                    }}
                  />
                  <YAxis
                    label={{
                      value: useLogScale ? "Log(Fitness)" : normalizeCurves ? "Normalized Fitness" : "Fitness",
                      angle: -90,
                      position: "insideLeft",
                      offset: 10,
                    }}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />

                  {Object.keys(algorithmResults).map((key, index) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={`var(--color-${key})`}
                      activeDot={{ r: 8 }}
                      strokeWidth={2}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          ) : (
            <div className="flex items-center justify-center h-full bg-muted rounded-md">
              <p className="text-muted-foreground">No convergence data available</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

