"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { useState } from "react"

interface ConfusionMatrixProps {
  matrix: number[][]
  labels: string[]
  modelName: string
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1: number
  }
}

export function ConfusionMatrix({
  matrix = [
    [0, 0],
    [0, 0],
  ],
  labels = ["Negative", "Positive"],
  modelName = "Unknown",
  metrics,
}: ConfusionMatrixProps) {
  const [normalized, setNormalized] = useState<boolean>(false)
  const [colorMode, setColorMode] = useState<string>("default")

  // Calculate totals for normalization
  const rowTotals = matrix.map((row) => row.reduce((sum, val) => sum + val, 0))
  const colTotals = matrix[0].map((_, colIndex) => matrix.reduce((sum, row) => sum + row[colIndex], 0))
  const total = rowTotals.reduce((sum, val) => sum + val, 0)

  // Get cell value based on normalization setting
  const getCellValue = (rowIdx: number, colIdx: number) => {
    const value = matrix[rowIdx][colIdx]

    if (!normalized) return value

    // Normalize by row (recall perspective)
    if (rowTotals[rowIdx] === 0) return 0
    return value / rowTotals[rowIdx]
  }

  // Get cell background color based on value and color mode
  const getCellBackground = (rowIdx: number, colIdx: number) => {
    const value = getCellValue(rowIdx, colIdx)
    const normalizedValue = normalized ? value : value / Math.max(...matrix.flat(), 1)

    // Green for correct predictions (diagonal), red for incorrect
    if (colorMode === "correctness") {
      return rowIdx === colIdx
        ? `rgba(0, 255, 0, ${normalizedValue * 0.8})` // Green for correct
        : `rgba(255, 0, 0, ${normalizedValue * 0.8})` // Red for incorrect
    }

    // Default blue gradient
    return `rgba(0, 0, 255, ${normalizedValue * 0.8})`
  }

  // Format cell value for display
  const formatCellValue = (rowIdx: number, colIdx: number) => {
    const value = getCellValue(rowIdx, colIdx)

    if (normalized) {
      // Display as percentage
      return `${(value * 100).toFixed(1)}%`
    }

    // Display as integer
    return value
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Confusion Matrix: {modelName}</CardTitle>
        <CardDescription>Evaluation of classification performance</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          <div className="flex items-center space-x-2">
            <Switch id="normalize" checked={normalized} onCheckedChange={setNormalized} />
            <Label htmlFor="normalize">Normalize</Label>
          </div>

          <div className="space-y-2">
            <Label htmlFor="color-mode">Color Mode</Label>
            <Select value={colorMode} onValueChange={setColorMode}>
              <SelectTrigger id="color-mode" className="w-[150px]">
                <SelectValue placeholder="Default" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Default</SelectItem>
                <SelectItem value="correctness">Correctness</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Confusion Matrix Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="border px-4 py-2 bg-muted font-medium text-sm">Actual ↓ / Predicted →</th>
                {labels.map((label, i) => (
                  <th key={i} className="border px-4 py-2 bg-muted font-medium text-sm">
                    {label}
                  </th>
                ))}
                <th className="border px-4 py-2 bg-muted font-medium text-sm">Total</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td className="border px-4 py-2 bg-muted font-medium text-sm">{labels[rowIdx]}</td>
                  {row.map((cell, colIdx) => (
                    <td
                      key={colIdx}
                      className="border px-4 py-2 text-center relative"
                      style={{ backgroundColor: getCellBackground(rowIdx, colIdx) }}
                    >
                      <span className={`font-medium ${rowIdx === colIdx ? "text-white" : ""}`}>
                        {formatCellValue(rowIdx, colIdx)}
                      </span>
                    </td>
                  ))}
                  <td className="border px-4 py-2 text-center bg-muted font-medium">{rowTotals[rowIdx]}</td>
                </tr>
              ))}
              <tr>
                <td className="border px-4 py-2 bg-muted font-medium text-sm">Total</td>
                {colTotals.map((tot, i) => (
                  <td key={i} className="border px-4 py-2 text-center bg-muted font-medium">
                    {tot}
                  </td>
                ))}
                <td className="border px-4 py-2 text-center bg-muted font-medium">{total}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Metrics display */}
        {metrics && (
          <div className="mt-6">
            <h3 className="text-sm font-medium mb-2">Performance Metrics</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Accuracy</p>
                <p className="text-lg font-semibold">{metrics.accuracy.toFixed(4)}</p>
              </div>
              <div className="bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Precision</p>
                <p className="text-lg font-semibold">{metrics.precision.toFixed(4)}</p>
              </div>
              <div className="bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Recall</p>
                <p className="text-lg font-semibold">{metrics.recall.toFixed(4)}</p>
              </div>
              <div className="bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground">F1 Score</p>
                <p className="text-lg font-semibold">{metrics.f1.toFixed(4)}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

