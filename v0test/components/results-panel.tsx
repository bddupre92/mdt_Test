"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { LineChart, BarChart } from "@/components/charts"
import { Download, Share2 } from "lucide-react"

interface ResultsPanelProps {
  results: any
  onExport?: () => void
  onShare?: () => void
}

export function ResultsPanel({ results, onExport, onShare }: ResultsPanelProps) {
  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center h-[300px] text-center">
        <p className="text-muted-foreground">Execute a model to see results</p>
      </div>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Results</CardTitle>
        <CardDescription>Model execution results and metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="metrics">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="charts">Charts</TabsTrigger>
            <TabsTrigger value="data">Data</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(results.metrics || {}).map(([key, value]) => (
                <div key={key} className="bg-muted rounded-lg p-3">
                  <p className="text-xs text-muted-foreground capitalize">{key.replace(/([A-Z])/g, " $1")}</p>
                  <p className="text-lg font-semibold">{typeof value === "number" ? value.toFixed(4) : value}</p>
                </div>
              ))}
            </div>

            <div className="mt-4">
              <p className="text-sm font-medium mb-1">Execution Time</p>
              <p>{results.executionTime?.toFixed(2) || "0.00"} seconds</p>
            </div>
          </TabsContent>

          <TabsContent value="charts">
            <div className="space-y-4">
              {results.convergence && (
                <div>
                  <p className="text-sm font-medium mb-2">Convergence</p>
                  <div className="h-[200px] bg-muted rounded-md">
                    <LineChart
                      data={results.convergence.map((point: any) => ({
                        x: point.iteration,
                        y: point.fitness,
                      }))}
                      xLabel="Iteration"
                      yLabel="Fitness"
                    />
                  </div>
                </div>
              )}

              {results.featureImportance && (
                <div>
                  <p className="text-sm font-medium mb-2">Feature Importance</p>
                  <div className="h-[200px] bg-muted rounded-md">
                    <BarChart
                      data={Object.entries(results.featureImportance).map(([feature, value]) => ({
                        x: feature,
                        y: value as number,
                      }))}
                      xLabel="Feature"
                      yLabel="Importance"
                    />
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="data">
            <div className="h-[300px] overflow-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-muted">
                    <th className="p-2 text-left text-sm font-medium">Index</th>
                    <th className="p-2 text-left text-sm font-medium">Actual</th>
                    <th className="p-2 text-left text-sm font-medium">Predicted</th>
                    <th className="p-2 text-left text-sm font-medium">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {results.predictions &&
                    results.predictions.slice(0, 10).map((pred: number, i: number) => {
                      const actual = results.actual?.[i] || 0
                      const error = actual - pred
                      return (
                        <tr key={i} className="border-b border-muted">
                          <td className="p-2 text-sm">{i}</td>
                          <td className="p-2 text-sm">{actual.toFixed(4)}</td>
                          <td className="p-2 text-sm">{pred.toFixed(4)}</td>
                          <td className="p-2 text-sm">{error.toFixed(4)}</td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onExport}>
          <Download className="mr-2 h-4 w-4" />
          Export
        </Button>
        <Button variant="outline" onClick={onShare}>
          <Share2 className="mr-2 h-4 w-4" />
          Share
        </Button>
      </CardFooter>
    </Card>
  )
}

