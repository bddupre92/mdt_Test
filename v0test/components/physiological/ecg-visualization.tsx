"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Button } from "../ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select"
import { Badge } from "../ui/badge"
import { ChartContainer } from "../ui/chart"
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface ECGVisualizationProps {
  patientId?: string
  data?: number[]
  sampling_rate?: number
  timestamp?: string
  showControls?: boolean
  height?: string
  title?: string
  isMock?: boolean
}

export default function ECGVisualization({
  patientId,
  data,
  sampling_rate = 250,
  timestamp,
  showControls = true,
  height = "300px",
  title = "ECG Visualization",
  isMock = false
}: ECGVisualizationProps) {
  // State for the visualization
  const [ecgData, setEcgData] = useState<number[]>(data || [])
  const [timeWindow, setTimeWindow] = useState<string>("10s")
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Generate mock data if no data is provided and isMock is true
  useEffect(() => {
    if (isMock && (!data || data.length === 0)) {
      generateMockData()
    } else if (data && data.length > 0) {
      setEcgData(data)
    }
  }, [data, isMock])

  // Generate realistic-looking ECG data for demo purposes
  const generateMockData = () => {
    setIsLoading(true)
    
    // Basic ECG waveform pattern (P, Q, R, S, T waves)
    const basePattern = [
      0.1, 0.2, 0.3, 0.2, 0.1, 0, -0.1, -0.2, 0.1, 1.2, 2.5, 1.0, -0.5, -1.0, -0.5, 0, 0.2, 0.4, 0.3, 0.2, 0.1, 0
    ]
    
    // Generate enough cycles to fill the time window (default 10s at 250Hz = 2500 samples)
    const cycleDuration = basePattern.length
    const numSamples = parseInt(timeWindow.replace('s', '')) * sampling_rate
    const numCycles = Math.ceil(numSamples / cycleDuration)
    
    const mockData: number[] = []
    
    // Create multiple cycles with small variations
    for (let i = 0; i < numCycles; i++) {
      // Add some variability to the heart rate
      const variability = 1 + (Math.random() * 0.1 - 0.05) // ±5% variability
      
      // Add base pattern with variability
      for (let j = 0; j < basePattern.length; j++) {
        mockData.push(basePattern[j] * variability + (Math.random() * 0.05 - 0.025)) // Add small noise
      }
    }
    
    // Trim to exact length needed
    const trimmedData = mockData.slice(0, numSamples)
    setEcgData(trimmedData)
    setIsLoading(false)
  }

  // Change the time window for visualization
  const handleTimeWindowChange = (value: string) => {
    setTimeWindow(value)
    if (isMock) {
      generateMockData()
    }
  }

  // Refresh the data
  const handleRefresh = () => {
    if (isMock) {
      generateMockData()
    } else {
      // In a real implementation, this would fetch fresh data from the API
      setIsLoading(true)
      setTimeout(() => {
        setIsLoading(false)
      }, 500)
    }
  }

  // Prepare data for Chart.js
  const chartData = {
    labels: ecgData.map((_, index) => index),
    datasets: [
      {
        label: 'ECG',
        data: ecgData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderWidth: 1.5,
        pointRadius: 0, // Don't show points for ECG data - too many points
        tension: 0.1, // Slight curve for smoother appearance
      },
    ],
  }

  // Chart options
  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Sample Number'
        },
        ticks: {
          maxTicksLimit: 10,
          callback: function(value: number | string) {
            return `${value}`
          }
        }
      },
      y: {
        title: {
          display: true,
          text: 'Amplitude (mV)'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    animation: {
      duration: 0 // Disable animation for better performance with large datasets
    }
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>
              {timestamp ? (
                <>Recorded at {new Date(timestamp).toLocaleString()}</>
              ) : (
                <>Real-time ECG monitoring</>
              )}
            </CardDescription>
          </div>
          
          {patientId && (
            <Badge variant="secondary">
              Patient ID: {patientId}
            </Badge>
          )}
        </div>
      </CardHeader>
      
      {showControls && (
        <div className="px-6 pb-2 flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">Time Window:</span>
            <Select 
              value={timeWindow} 
              onValueChange={handleTimeWindowChange}
            >
              <SelectTrigger className="w-24">
                <SelectValue placeholder="10s" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5s">5s</SelectItem>
                <SelectItem value="10s">10s</SelectItem>
                <SelectItem value="30s">30s</SelectItem>
                <SelectItem value="60s">60s</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Button 
            size="sm" 
            variant="outline" 
            onClick={handleRefresh}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </Button>
        </div>
      )}
      
      <CardContent>
        <div style={{ height }}>
          {ecgData.length > 0 ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <p className="text-muted-foreground">
                {error || 'No ECG data available'}
              </p>
            </div>
          )}
        </div>
        
        {ecgData.length > 0 && (
          <div className="mt-4 flex items-center text-sm text-muted-foreground">
            <span>Sampling Rate: {sampling_rate} Hz</span>
            <span className="mx-2">•</span>
            <span>Duration: {(ecgData.length / sampling_rate).toFixed(1)}s</span>
            <span className="mx-2">•</span>
            <span>Samples: {ecgData.length}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 