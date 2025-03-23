"use client"

import { useEffect, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Loader2 } from "lucide-react"

interface BarChartSeries {
  name: string
  data: number[]
  color?: string
}

interface BarChartProps {
  series: BarChartSeries[]
  categories: string[]
  title?: string
  xAxisTitle?: string
  yAxisTitle?: string
  height?: number
  width?: number
  isLoading?: boolean
  showLegend?: boolean
  showGrid?: boolean
  theme?: "light" | "dark"
  colors?: string[]
  horizontal?: boolean
  stacked?: boolean
}

export function BarChart({
  series,
  categories,
  title,
  xAxisTitle = "",
  yAxisTitle = "",
  height = 300,
  width,
  isLoading = false,
  showLegend = true,
  showGrid = true,
  theme = "light",
  colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"],
  horizontal = false,
  stacked = false
}: BarChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [chartLoaded, setChartLoaded] = useState(false)
  const [apexCharts, setApexCharts] = useState<any>(null)
  const [chart, setChart] = useState<any>(null)

  // Dynamic import ApexCharts on client side only
  useEffect(() => {
    import("apexcharts").then(mod => {
      setApexCharts(mod.default)
      setChartLoaded(true)
    })
  }, [])

  // Initialize and update chart when data changes
  useEffect(() => {
    // Wait for ApexCharts to load and have a reference to the DOM element
    if (!chartLoaded || !chartRef.current || !apexCharts) return

    // Convert series data to ApexCharts format
    const formattedSeries = series.map((s, i) => ({
      name: s.name,
      data: s.data,
      color: s.color || colors[i % colors.length]
    }))

    // Chart options
    const options = {
      chart: {
        type: horizontal ? "bar" : "column",
        height,
        width: width || "100%",
        stacked: stacked,
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: false,
            zoom: false,
            zoomin: false,
            zoomout: false,
            pan: false,
            reset: true
          }
        },
        animations: {
          enabled: true,
          speed: 300,
          animateGradually: {
            enabled: true,
            delay: 150
          }
        },
        background: "transparent",
        fontFamily: "inherit"
      },
      colors: formattedSeries.map(s => s.color),
      series: formattedSeries,
      xaxis: {
        categories: categories,
        title: {
          text: xAxisTitle,
          style: {
            fontSize: "12px",
            fontWeight: 600
          }
        },
        labels: {
          style: {
            fontSize: "10px"
          },
          rotate: -45,
          rotateAlways: false,
          hideOverlappingLabels: true
        },
        axisBorder: {
          show: true
        },
        axisTicks: {
          show: true
        }
      },
      yaxis: {
        title: {
          text: yAxisTitle,
          style: {
            fontSize: "12px",
            fontWeight: 600
          }
        },
        labels: {
          style: {
            fontSize: "10px"
          },
          formatter: (value: number) => {
            // Format numbers to avoid scientific notation
            return value.toLocaleString(undefined, {
              maximumFractionDigits: 6
            })
          }
        }
      },
      grid: {
        show: showGrid,
        borderColor: "#f1f5f9", // Slate-100
        strokeDashArray: 4,
        position: "back"
      },
      legend: {
        show: showLegend,
        position: "top",
        horizontalAlign: "right",
        fontSize: "12px",
        markers: {
          width: 12,
          height: 12,
          radius: 2
        }
      },
      plotOptions: {
        bar: {
          horizontal: horizontal,
          borderRadius: 2,
          columnWidth: '70%',
          distributed: series.length === 1, // If only one series, distribute the colors
          dataLabels: {
            position: 'top'
          }
        }
      },
      dataLabels: {
        enabled: false
      },
      tooltip: {
        enabled: true,
        shared: true,
        intersect: false,
        style: {
          fontSize: "12px"
        },
        y: {
          formatter: (value: number) => value.toFixed(6)
        }
      },
      theme: {
        mode: theme
      },
      title: title
        ? {
            text: title,
            align: "left",
            style: {
              fontSize: "14px",
              fontWeight: 600
            }
          }
        : undefined
    }

    // Initialize chart if it doesn't exist, update it otherwise
    if (chart) {
      chart.updateOptions(options)
    } else {
      const newChart = new apexCharts(chartRef.current, options)
      newChart.render()
      setChart(newChart)
    }

    // Cleanup function
    return () => {
      if (chart) {
        chart.destroy()
        setChart(null)
      }
    }
  }, [
    chartLoaded,
    apexCharts,
    series,
    categories,
    title,
    xAxisTitle,
    yAxisTitle,
    height,
    width,
    showLegend,
    showGrid,
    theme,
    colors,
    horizontal,
    stacked,
    chart
  ])

  if (isLoading) {
    return (
      <Card className="w-full flex items-center justify-center" style={{ height }}>
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2 text-primary" />
          <p className="text-sm text-muted-foreground">Loading chart data...</p>
        </div>
      </Card>
    )
  }

  if (series.length === 0 || series.every(s => s.data.length === 0)) {
    return (
      <Card className="w-full flex items-center justify-center" style={{ height }}>
        <div className="text-center p-6">
          <p className="text-sm text-muted-foreground">No data available</p>
        </div>
      </Card>
    )
  }

  return (
    <div className="w-full">
      <div ref={chartRef} className="w-full"></div>
    </div>
  )
} 