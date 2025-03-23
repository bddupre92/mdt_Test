"use client"

import { useState, useEffect, useMemo } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartOptions
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Button } from "../ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select"
import { Slider } from "../ui/slider"
import { getPatientHRV } from '../../lib/api/physiological'
import 'chartjs-adapter-date-fns'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  TimeScale,
  Title,
  Tooltip,
  Legend
)

interface HRVVisualizationProps {
  patientId: string
  data?: {
    rr_intervals: number[]
    timestamps: string[]
    sdnn?: number
    rmssd?: number
    lf_hf_ratio?: number
  }
  title?: string
  height?: string
  showControls?: boolean
  isMock?: boolean
}

export default function HRVVisualization({
  patientId,
  data: propData,
  title = "Heart Rate Variability",
  height = "400px",
  showControls = true,
  isMock = false
}: HRVVisualizationProps) {
  const [data, setData] = useState(propData)
  const [isLoading, setIsLoading] = useState(!propData)
  const [error, setError] = useState<string | null>(null)
  const [timeWindow, setTimeWindow] = useState<string>("5min")
  const [viewMode, setViewMode] = useState<string>("rr")
  const [smoothingFactor, setSmoothingFactor] = useState<number>(5)
  
  // Generate mock data for demonstration
  const generateMockData = () => {
    const now = new Date()
    const timestamps: string[] = []
    const rr_intervals: number[] = []
    
    // Base RR interval around 800ms (75 bpm) with natural variability
    let lastRR = 800 + (Math.random() * 100 - 50)
    
    // Generate 5 minutes of RR intervals (assuming about 75 bpm)
    for (let i = 0; i < 375; i++) {
      // Add some variability with occasional "stress" periods
      const stressPeriod = i > 100 && i < 150 || i > 250 && i < 300
      
      // More variability during normal periods, less during "stress"
      const variability = stressPeriod ? 30 : 80
      
      // Adjust mean RR interval - shorter during "stress" periods
      const meanRR = stressPeriod ? 700 : 850
      
      // Generate next RR interval with some auto-correlation to previous
      lastRR = lastRR * 0.7 + 0.3 * (meanRR + (Math.random() * variability - variability/2))
      
      // Ensure reasonable bounds
      lastRR = Math.max(600, Math.min(1100, lastRR))
      
      rr_intervals.push(lastRR)
      
      // Calculate timestamp based on cumulative RR intervals
      const timestamp = new Date(now.getTime() - (375 - i) * lastRR)
      timestamps.push(timestamp.toISOString())
    }
    
    // Calculate some HRV metrics
    const sdnn = calculateSDNN(rr_intervals)
    const rmssd = calculateRMSSD(rr_intervals)
    const lf_hf_ratio = 2.1 + Math.random() * 0.8 // Mock value
    
    return {
      rr_intervals,
      timestamps,
      sdnn,
      rmssd,
      lf_hf_ratio
    }
  }
  
  // Calculate SDNN (Standard Deviation of NN intervals)
  const calculateSDNN = (rrIntervals: number[]): number => {
    const mean = rrIntervals.reduce((sum, val) => sum + val, 0) / rrIntervals.length
    const squaredDiffs = rrIntervals.map(val => Math.pow(val - mean, 2))
    return Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0) / rrIntervals.length)
  }
  
  // Calculate RMSSD (Root Mean Square of Successive Differences)
  const calculateRMSSD = (rrIntervals: number[]): number => {
    let successiveDiffSquareSum = 0
    for (let i = 1; i < rrIntervals.length; i++) {
      const diff = rrIntervals[i] - rrIntervals[i-1]
      successiveDiffSquareSum += diff * diff
    }
    return Math.sqrt(successiveDiffSquareSum / (rrIntervals.length - 1))
  }
  
  // Apply smoothing to data series
  const applySmoothing = (data: number[], windowSize: number): number[] => {
    if (windowSize <= 1) return data
    
    const smoothed = []
    for (let i = 0; i < data.length; i++) {
      let sum = 0
      let count = 0
      
      for (let j = Math.max(0, i - Math.floor(windowSize/2)); 
           j <= Math.min(data.length - 1, i + Math.floor(windowSize/2)); j++) {
        sum += data[j]
        count++
      }
      
      smoothed.push(sum / count)
    }
    
    return smoothed
  }
  
  // Fetch HRV data from API
  useEffect(() => {
    if (propData) {
      setData(propData)
      return
    }
    
    const fetchData = async () => {
      setIsLoading(true)
      setError(null)
      
      try {
        if (isMock) {
          // Use mock data for demonstrations
          setTimeout(() => {
            const mockData = generateMockData()
            setData(mockData)
            setIsLoading(false)
          }, 1000)
        } else {
          // Get real data from the API
          const timeWindowMs = timeWindow === "5min" ? 5 * 60 * 1000 : 
                               timeWindow === "1hr" ? 60 * 60 * 1000 : 
                               24 * 60 * 60 * 1000
          
          const response = await getPatientHRV(patientId, timeWindowMs)
          setData(response)
          setIsLoading(false)
        }
      } catch (err) {
        console.error("Error fetching HRV data:", err)
        setError("Failed to load HRV data")
        setIsLoading(false)
      }
    }
    
    fetchData()
  }, [patientId, propData, timeWindow, isMock])
  
  // Prepare chart data
  const chartData = useMemo(() => {
    if (!data) return null
    
    let plotData: number[]
    let labels: string[]
    
    if (viewMode === "rr") {
      // Use RR intervals
      plotData = [...data.rr_intervals]
      labels = [...data.timestamps]
    } else {
      // Calculate instantaneous HR from RR intervals (60000 / RR in ms = HR in bpm)
      plotData = data.rr_intervals.map(rr => 60000 / rr)
      labels = [...data.timestamps]
    }
    
    // Apply smoothing
    const smoothedData = applySmoothing(plotData, smoothingFactor)
    
    return {
      labels,
      datasets: [
        {
          label: viewMode === "rr" ? 'RR Intervals (ms)' : 'Heart Rate (bpm)',
          data: smoothedData,
          borderColor: viewMode === "rr" ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)',
          backgroundColor: viewMode === "rr" ? 'rgba(75, 192, 192, 0.2)' : 'rgba(255, 99, 132, 0.2)',
          borderWidth: 1.5,
          pointRadius: smoothingFactor > 0 ? 0 : 1,
          tension: 0.2
        }
      ]
    }
  }, [data, viewMode, smoothingFactor])
  
  // Chart options
  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: timeWindow === "5min" ? 'minute' : timeWindow === "1hr" ? 'minute' : 'hour',
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm'
          }
        },
        title: {
          display: true,
          text: 'Time'
        }
      },
      y: {
        title: {
          display: true,
          text: viewMode === "rr" ? 'RR Interval (ms)' : 'Heart Rate (bpm)'
        },
        suggestedMin: viewMode === "rr" ? 600 : 50,
        suggestedMax: viewMode === "rr" ? 1100 : 100
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          title: function(tooltipItems: any[]) {
            return new Date(tooltipItems[0].parsed.x).toLocaleTimeString()
          }
        }
      }
    }
  }
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>Loading HRV data...</CardDescription>
        </CardHeader>
        <CardContent style={{ height }}>
          <div className="h-full w-full flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription className="text-red-500">{error}</CardDescription>
        </CardHeader>
        <CardContent style={{ height }}>
          <div className="h-full w-full flex flex-col items-center justify-center">
            <p className="text-muted-foreground mb-4">Unable to load HRV data</p>
            <Button onClick={() => window.location.reload()}>Retry</Button>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  return (
    <Card>
      <CardHeader className="space-y-0 pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>
              Analysis of heart beat timing variability
            </CardDescription>
          </div>
          
          {showControls && (
            <div className="flex space-x-2">
              <Select value={viewMode} onValueChange={setViewMode}>
                <SelectTrigger className="w-[120px] h-8">
                  <SelectValue placeholder="View" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rr">RR Intervals</SelectItem>
                  <SelectItem value="hr">Heart Rate</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={timeWindow} onValueChange={setTimeWindow}>
                <SelectTrigger className="w-[100px] h-8">
                  <SelectValue placeholder="Time" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="5min">5 min</SelectItem>
                  <SelectItem value="1hr">1 hour</SelectItem>
                  <SelectItem value="24hr">24 hours</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        <div style={{ height }}>
          {chartData ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="h-full w-full flex items-center justify-center">
              <p className="text-muted-foreground">No data available</p>
            </div>
          )}
        </div>
        
        {showControls && data && (
          <div className="mt-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Smoothing</span>
              <span className="text-sm text-muted-foreground">{smoothingFactor}</span>
            </div>
            <Slider
              value={[smoothingFactor]}
              min={0}
              max={20}
              step={1}
              onValueChange={(values) => setSmoothingFactor(values[0])}
              className="my-2"
            />
            
            <div className="grid grid-cols-3 gap-4 mt-6 text-center">
              <div>
                <div className="text-sm font-medium">SDNN</div>
                <div className="text-xl font-bold">{data.sdnn ? Math.round(data.sdnn) : "N/A"}</div>
                <div className="text-xs text-muted-foreground">ms</div>
              </div>
              <div>
                <div className="text-sm font-medium">RMSSD</div>
                <div className="text-xl font-bold">{data.rmssd ? Math.round(data.rmssd) : "N/A"}</div>
                <div className="text-xs text-muted-foreground">ms</div>
              </div>
              <div>
                <div className="text-sm font-medium">LF/HF</div>
                <div className="text-xl font-bold">{data.lf_hf_ratio ? data.lf_hf_ratio.toFixed(2) : "N/A"}</div>
                <div className="text-xs text-muted-foreground">ratio</div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 