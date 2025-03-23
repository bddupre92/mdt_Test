"use client"

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs"
import { Button } from "../ui/button"
import { Badge } from "../ui/badge"
import { Progress } from "../ui/progress"
import { Separator } from "../ui/separator"
import { getPatientDataSummary, getPatientTriggers, PatientDataSummary, PatientTrigger } from '../../lib/api/prediction'
import ECGVisualization from './ecg-visualization'
import HRVVisualization from './hrv-visualization'
import { Clock, Activity, Thermometer, BarChart2, AlertCircle } from 'lucide-react'

interface HeartRateData {
  current: number
  min: number
  max: number
  average: number
  timestamp: string
}

interface PatientDashboardProps {
  patientId?: string
}

export default function PatientDashboard({ patientId: propPatientId }: PatientDashboardProps) {
  const searchParams = useSearchParams()
  const patientId = propPatientId || searchParams.get('id') || 'demo-patient'
  
  const [activePeriod, setActivePeriod] = useState<string>('day')
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [dataSummary, setDataSummary] = useState<PatientDataSummary | null>(null)
  const [triggers, setTriggers] = useState<PatientTrigger[]>([])
  const [heartRateData, setHeartRateData] = useState<HeartRateData>({
    current: 72,
    min: 58,
    max: 115,
    average: 68,
    timestamp: new Date().toISOString()
  })
  
  // Fetch patient data on component mount
  useEffect(() => {
    if (!patientId) return
    
    const fetchData = async () => {
      setIsLoading(true)
      try {
        // In a real implementation, this would fetch actual patient data
        // For demo, we use mock data if the patient is 'demo-patient'
        if (patientId === 'demo-patient') {
          // Mock data for demo
          setTimeout(() => {
            setDataSummary({
              patient_id: 'demo-patient',
              physiological_count: 12500,
              environmental_count: 150,
              behavioral_count: 35,
              data_period: {
                start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
                end: new Date().toISOString()
              }
            })
            
            setTriggers([
              {
                trigger: 'Sleep disruption',
                confidence: 0.92,
                occurrences: 5,
                last_detected: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
              },
              {
                trigger: 'Stress',
                confidence: 0.85,
                occurrences: 8,
                last_detected: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString()
              },
              {
                trigger: 'Weather changes',
                confidence: 0.78,
                occurrences: 3,
                last_detected: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString()
              },
              {
                trigger: 'Skipped meals',
                confidence: 0.65,
                occurrences: 4,
                last_detected: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
              }
            ])
            
            setIsLoading(false)
          }, 1000)
        } else {
          // Fetch real data from API
          const [summaryData, triggerData] = await Promise.all([
            getPatientDataSummary(patientId),
            getPatientTriggers(patientId)
          ])
          
          setDataSummary(summaryData)
          setTriggers(triggerData)
          setIsLoading(false)
        }
      } catch (err) {
        console.error('Error fetching patient data:', err)
        setError('Failed to load patient data')
        setIsLoading(false)
      }
    }
    
    fetchData()
  }, [patientId])
  
  // Calculate risk score based on triggers and other factors
  const calculateRiskScore = (): number => {
    if (!triggers.length) return 0.15 // Base risk
    
    // Simple weighted calculation based on trigger confidence and recency
    let score = 0
    const now = new Date().getTime()
    
    triggers.forEach(trigger => {
      const lastDetectedDate = new Date(trigger.last_detected).getTime()
      const daysSinceDetection = (now - lastDetectedDate) / (1000 * 60 * 60 * 24)
      
      // Higher weight for recent triggers
      const recencyFactor = Math.max(0, 1 - (daysSinceDetection / 7))
      
      score += trigger.confidence * recencyFactor * 0.25
    })
    
    // Add some randomness for demo purposes
    score = Math.min(1, score + Math.random() * 0.1)
    
    return parseFloat(score.toFixed(2))
  }
  
  const riskScore = calculateRiskScore()
  
  // Get risk level based on score
  const getRiskLevel = (score: number): string => {
    if (score < 0.3) return 'Low'
    if (score < 0.6) return 'Moderate'
    return 'High'
  }
  
  // Get color based on risk level
  const getRiskColor = (score: number): string => {
    if (score < 0.3) return 'bg-green-500'
    if (score < 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }
  
  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading patient data...</p>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="w-full p-8 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto" />
          <p className="mt-4 text-lg font-semibold">{error}</p>
          <Button className="mt-4" onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </div>
    )
  }
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Patient Dashboard</h1>
          <p className="text-muted-foreground">
            View patient physiological data and risk assessment
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-sm">
            Patient ID: {patientId}
          </Badge>
          
          <Button size="sm" variant="outline">
            Export Data
          </Button>
        </div>
      </div>
      
      <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        {/* Risk Score Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Risk Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="text-2xl font-bold">{(riskScore * 100).toFixed(0)}%</div>
              <Badge className={`${getRiskColor(riskScore)} text-white`}>
                {getRiskLevel(riskScore)}
              </Badge>
            </div>
            <Progress
              value={riskScore * 100}
              className={`h-2 mt-2 ${getRiskColor(riskScore)}`}
            />
            <p className="text-xs text-muted-foreground mt-2">
              Based on {triggers.length} identified triggers
            </p>
          </CardContent>
        </Card>
        
        {/* Heart Rate Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Heart Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-end justify-between">
              <div className="text-2xl font-bold">{heartRateData.current} <span className="text-sm font-normal text-muted-foreground">bpm</span></div>
              <div className="flex items-center">
                <Activity className="mr-1 h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">
                  {heartRateData.min}-{heartRateData.max} bpm
                </span>
              </div>
            </div>
            <div className="mt-4 h-[40px] flex items-end justify-between">
              {/* Simple heart rate visualization */}
              {Array.from({ length: 10 }).map((_, i) => (
                <div 
                  key={i}
                  className="bg-primary w-[8%] rounded-full"
                  style={{ 
                    height: `${Math.max(15, Math.min(100, 20 + Math.sin(i * 0.9) * 60 + Math.random() * 20))}%`
                  }}
                />
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Last updated: {new Date(heartRateData.timestamp).toLocaleTimeString()}
            </p>
          </CardContent>
        </Card>
        
        {/* Data Points Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Collected Data</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dataSummary?.physiological_count.toLocaleString() || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Physiological data points
            </p>
            
            <div className="grid grid-cols-2 gap-2 mt-4">
              <div>
                <div className="text-sm font-medium">
                  {dataSummary?.environmental_count.toLocaleString() || 0}
                </div>
                <p className="text-xs text-muted-foreground">Environmental</p>
              </div>
              <div>
                <div className="text-sm font-medium">
                  {dataSummary?.behavioral_count.toLocaleString() || 0}
                </div>
                <p className="text-xs text-muted-foreground">Behavioral</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Monitoring Period Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Monitoring Period</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-2 text-muted-foreground" />
              <div className="text-sm font-medium">
                {dataSummary?.data_period.start && dataSummary?.data_period.end ? (
                  `${Math.ceil((new Date(dataSummary.data_period.end).getTime() - new Date(dataSummary.data_period.start).getTime()) / (1000 * 60 * 60 * 24))} days`
                ) : (
                  '7 days'
                )}
              </div>
            </div>
            
            <div className="mt-4 text-xs text-muted-foreground grid grid-cols-2 gap-x-2 gap-y-1">
              <div>Start:</div>
              <div>{dataSummary?.data_period.start ? new Date(dataSummary.data_period.start).toLocaleDateString() : '-'}</div>
              <div>End:</div>
              <div>{dataSummary?.data_period.end ? new Date(dataSummary.data_period.end).toLocaleDateString() : '-'}</div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="ecg" className="space-y-4">
        <TabsList>
          <TabsTrigger value="ecg">ECG</TabsTrigger>
          <TabsTrigger value="triggers">Triggers</TabsTrigger>
          <TabsTrigger value="vitals">Vitals</TabsTrigger>
          <TabsTrigger value="environment">Environment</TabsTrigger>
        </TabsList>
        
        <TabsContent value="ecg" className="space-y-4">
          <ECGVisualization 
            patientId={patientId}
            isMock={true}
            title="Electrocardiogram (ECG)"
            height="350px"
          />
        </TabsContent>
        
        <TabsContent value="triggers" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Identified Triggers</CardTitle>
              <CardDescription>
                Potential migraine triggers identified from patient data
              </CardDescription>
            </CardHeader>
            <CardContent>
              {triggers.length > 0 ? (
                <div className="space-y-4">
                  {triggers.map((trigger, index) => (
                    <div key={index} className="flex items-center justify-between pb-4 border-b last:border-0 last:pb-0">
                      <div>
                        <div className="font-medium">{trigger.trigger}</div>
                        <div className="text-sm text-muted-foreground">
                          {trigger.occurrences} occurrences • Last detected: {new Date(trigger.last_detected).toLocaleDateString()}
                        </div>
                      </div>
                      <Badge className={`${trigger.confidence > 0.8 ? 'bg-red-500' : trigger.confidence > 0.6 ? 'bg-yellow-500' : 'bg-green-500'} text-white`}>
                        {(trigger.confidence * 100).toFixed(0)}% confidence
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-muted-foreground">
                  No triggers identified yet
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="vitals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Vital Signs</CardTitle>
              <CardDescription>
                Summary of patient's vital sign measurements
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Heart Rate Section */}
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    <Activity className="mr-2 h-4 w-4" />
                    Heart Rate
                  </div>
                  <div className="text-2xl font-bold">{heartRateData.current} bpm</div>
                  <div className="text-sm text-muted-foreground">
                    Average: {heartRateData.average} bpm • Range: {heartRateData.min}-{heartRateData.max} bpm
                  </div>
                  <Progress 
                    value={((heartRateData.current - 40) / (200 - 40)) * 100} 
                    className="h-2"
                  />
                </div>
                
                {/* Body Temperature Section (mock data) */}
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    <Thermometer className="mr-2 h-4 w-4" />
                    Body Temperature
                  </div>
                  <div className="text-2xl font-bold">36.8°C</div>
                  <div className="text-sm text-muted-foreground">
                    Average: 36.5°C • Range: 36.2-37.1°C
                  </div>
                  <Progress 
                    value={((36.8 - 35) / (41 - 35)) * 100} 
                    className="h-2"
                  />
                </div>
                
                {/* Mock other vital signs */}
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Blood Pressure
                  </div>
                  <div className="text-2xl font-bold">124/82 mmHg</div>
                  <div className="text-sm text-muted-foreground">
                    Average: 120/80 mmHg
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Respiratory Rate
                  </div>
                  <div className="text-2xl font-bold">16 bpm</div>
                  <div className="text-sm text-muted-foreground">
                    Average: 15 bpm • Range: 12-20 bpm
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* HRV Visualization */}
          <HRVVisualization
            patientId={patientId}
            isMock={true}
            title="Heart Rate Variability"
            height="300px"
          />
        </TabsContent>
        
        <TabsContent value="environment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Environmental Factors</CardTitle>
              <CardDescription>
                Environmental conditions that may affect the patient
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                {/* Mock environmental data */}
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Temperature
                  </div>
                  <div className="text-2xl font-bold">23.5°C</div>
                  <div className="text-sm text-muted-foreground">
                    Average indoor temperature
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Humidity
                  </div>
                  <div className="text-2xl font-bold">42%</div>
                  <div className="text-sm text-muted-foreground">
                    Indoor relative humidity
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Barometric Pressure
                  </div>
                  <div className="text-2xl font-bold">1013 hPa</div>
                  <div className="text-sm text-muted-foreground">
                    Slight drop in last 24 hours
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Light Level
                  </div>
                  <div className="text-2xl font-bold">320 lux</div>
                  <div className="text-sm text-muted-foreground">
                    Average light exposure
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Noise Level
                  </div>
                  <div className="text-2xl font-bold">48 dB</div>
                  <div className="text-sm text-muted-foreground">
                    Average ambient noise
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center text-sm font-medium">
                    Air Quality
                  </div>
                  <div className="text-2xl font-bold">Good</div>
                  <div className="text-sm text-muted-foreground">
                    AQI: 35 (PM2.5: 8 μg/m³)
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 