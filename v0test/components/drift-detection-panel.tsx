"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { LineChart } from "@/components/charts"
import { AlertTriangle, CheckCircle2 } from "lucide-react"

interface DriftDetectionPanelProps {
  driftData: {
    detected: boolean
    score: number
    threshold: number
    metrics: {
      ksStatistic: number
      pValue: number
      meanShift: number
      varianceRatio: number
    }
    history: Array<{ time: number; score: number }>
  }
}

export function DriftDetectionPanel({ driftData }: DriftDetectionPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Drift Detection</CardTitle>
        <CardDescription>Monitor and detect changes in data distribution</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium">Drift Status</h3>
              <p className="text-sm text-muted-foreground">
                {driftData.detected ? "Significant drift detected" : "No significant drift detected"}
              </p>
            </div>
            <Badge
              variant="outline"
              className={driftData.detected ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}
            >
              {driftData.detected ? (
                <AlertTriangle className="mr-1 h-3 w-3" />
              ) : (
                <CheckCircle2 className="mr-1 h-3 w-3" />
              )}
              {driftData.detected ? "Drift Detected" : "Stable"}
            </Badge>
          </div>

          <Separator />

          <div>
            <h3 className="text-sm font-medium mb-2">Drift Metrics</h3>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-muted rounded-lg p-2">
                <p className="text-xs text-muted-foreground">KS Statistic</p>
                <p className="font-medium">{driftData.metrics.ksStatistic.toFixed(3)}</p>
              </div>
              <div className="bg-muted rounded-lg p-2">
                <p className="text-xs text-muted-foreground">P-Value</p>
                <p className="font-medium">{driftData.metrics.pValue.toFixed(3)}</p>
              </div>
              <div className="bg-muted rounded-lg p-2">
                <p className="text-xs text-muted-foreground">Mean Shift</p>
                <p className="font-medium">{driftData.metrics.meanShift.toFixed(3)}</p>
              </div>
              <div className="bg-muted rounded-lg p-2">
                <p className="text-xs text-muted-foreground">Variance Ratio</p>
                <p className="font-medium">{driftData.metrics.varianceRatio.toFixed(3)}</p>
              </div>
            </div>
          </div>

          <div className="h-[150px] bg-muted rounded-md">
            <LineChart
              data={driftData.history.map((point) => ({
                x: point.time,
                y: point.score,
              }))}
              xLabel="Time Window"
              yLabel="Drift Score"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

