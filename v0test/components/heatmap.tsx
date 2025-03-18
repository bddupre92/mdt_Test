"use client"

import { ResponsiveContainer } from "recharts"
import { useRef, useEffect, useMemo } from "react"

interface HeatmapProps {
  data: Array<
    Array<{
      x: number | string
      y: number | string
      value: number | null
      isBest?: boolean
    }>
  >
  xLabels: Array<number | string>
  yLabels: Array<number | string>
  xLabel: string
  yLabel: string
  valueLabel: string
}

export function Heatmap({
  data = [],
  xLabels = [],
  yLabels = [],
  xLabel = "",
  yLabel = "",
  valueLabel = "",
}: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Find min and max values for color scaling
  const { minValue, maxValue } = useMemo(() => {
    const values = data
      .flat()
      .map((d) => d?.value)
      .filter((v) => v !== null && v !== undefined) as number[]
    if (values.length === 0) return { minValue: 0, maxValue: 1 }
    return {
      minValue: Math.min(...values),
      maxValue: Math.max(...values),
    }
  }, [data])

  // Function to map value to color
  const getColor = (value: number | null) => {
    if (value === null || value === undefined) return "rgba(200, 200, 200, 0.5)" // gray for missing values

    // Normalize between 0 and 1
    const normalizedValue = (value - minValue) / Math.max(maxValue - minValue, 0.001)

    // Blue to red color gradient
    const r = Math.floor(normalizedValue * 255)
    const g = 0
    const b = Math.floor(255 - normalizedValue * 255)

    return `rgb(${r}, ${g}, ${b})`
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // If no data, draw empty state
    if (data.length === 0 || xLabels.length === 0 || yLabels.length === 0) {
      ctx.fillStyle = "rgba(200, 200, 200, 0.5)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = "black"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("No data available", canvas.width / 2, canvas.height / 2)
      return
    }

    // Calculate dimensions
    const cellWidth = canvas.width / (xLabels.length + 1)
    const cellHeight = canvas.height / (yLabels.length + 1)

    // Draw cells
    data.forEach((row, rowIdx) => {
      row.forEach((cell, colIdx) => {
        if (!cell) return

        const x = (colIdx + 1) * cellWidth
        const y = (rowIdx + 1) * cellHeight

        // Draw rectangle
        ctx.fillStyle = getColor(cell.value)
        ctx.fillRect(x, y, cellWidth, cellHeight)

        // Draw border
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"
        ctx.strokeRect(x, y, cellWidth, cellHeight)

        // Highlight best parameter combination
        if (cell.isBest) {
          ctx.strokeStyle = "black"
          ctx.lineWidth = 2
          ctx.strokeRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4)
          ctx.lineWidth = 1
        }

        // Draw value
        if (cell.value !== null && cell.value !== undefined) {
          ctx.fillStyle = "white"
          ctx.font = "10px Arial"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(cell.value.toFixed(3), x + cellWidth / 2, y + cellHeight / 2)
        }
      })
    })

    // Draw x-axis labels
    ctx.fillStyle = "black"
    ctx.font = "10px Arial"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    xLabels.forEach((label, idx) => {
      const x = (idx + 1.5) * cellWidth
      ctx.fillText(String(label), x, (yLabels.length + 1) * cellHeight + 5)
    })

    // Draw y-axis labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    yLabels.forEach((label, idx) => {
      const y = (idx + 1.5) * cellHeight
      ctx.fillText(String(label), cellWidth - 5, y)
    })

    // Draw x-axis title
    ctx.font = "12px Arial"
    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    ctx.fillText(xLabel, canvas.width / 2, canvas.height - 5)

    // Draw y-axis title
    ctx.save()
    ctx.translate(10, canvas.height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText(yLabel, 0, 0)
    ctx.restore()
  }, [data, xLabels, yLabels, xLabel, yLabel, minValue, maxValue])

  return (
    <ResponsiveContainer width="100%" height="100%">
      <div className="w-full h-full flex items-center justify-center">
        <canvas ref={canvasRef} width={600} height={400} className="max-w-full max-h-full" />
      </div>
    </ResponsiveContainer>
  )
}

