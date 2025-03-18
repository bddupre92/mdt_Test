"use client"

import { useState } from "react"
import { useSearchParams } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { Input } from "../../components/ui/input"
import { Button } from "../../components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import ECGVisualization from "../../components/physiological/ecg-visualization"
import HRVVisualization from "../../components/physiological/hrv-visualization"
import PatientDashboard from "../../components/physiological/patient-dashboard"

export default function PhysiologicalPage() {
  const searchParams = useSearchParams()
  const initialPatientId = searchParams.get('patientId') || 'demo-patient'
  
  const [patientId, setPatientId] = useState<string>(initialPatientId)
  const [inputPatientId, setInputPatientId] = useState<string>(initialPatientId)
  const [selectedView, setSelectedView] = useState<string>("dashboard")
  
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setPatientId(inputPatientId)
  }
  
  return (
    <div className="container py-8 space-y-8">
      <div className="flex flex-col md:flex-row justify-between gap-4 items-start md:items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Physiological Data Explorer</h1>
          <p className="text-muted-foreground mt-1">
            Visualize and analyze patient physiological signals
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex w-full md:w-auto gap-2">
          <Input
            placeholder="Enter patient ID"
            value={inputPatientId}
            onChange={(e) => setInputPatientId(e.target.value)}
            className="w-full md:w-auto"
          />
          <Button type="submit">Load</Button>
        </form>
      </div>
      
      <div>
        <Tabs defaultValue={selectedView} onValueChange={setSelectedView} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard">Patient Dashboard</TabsTrigger>
            <TabsTrigger value="ecg">ECG Analysis</TabsTrigger>
            <TabsTrigger value="hrv">HRV Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="dashboard" className="mt-6">
            <PatientDashboard patientId={patientId} />
          </TabsContent>
          
          <TabsContent value="ecg" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Electrocardiogram (ECG) Analysis</CardTitle>
                <CardDescription>
                  Detailed visualization of electrical activity of the heart
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
                  <div className="lg:col-span-2">
                    <ECGVisualization
                      patientId={patientId}
                      isMock={true}
                      title="Real-time ECG"
                      height="350px"
                      showControls={true}
                    />
                  </div>
                  
                  <div>
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">ECG Parameters</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <div className="text-sm font-medium">Heart Rate</div>
                            <div className="text-2xl font-bold">72 bpm</div>
                            <div className="text-xs text-muted-foreground">Normal range: 60-100 bpm</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">QT Interval</div>
                            <div className="text-2xl font-bold">420 ms</div>
                            <div className="text-xs text-muted-foreground">Corrected QT: 440 ms</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">PR Interval</div>
                            <div className="text-2xl font-bold">160 ms</div>
                            <div className="text-xs text-muted-foreground">Normal range: 120-200 ms</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">QRS Duration</div>
                            <div className="text-2xl font-bold">90 ms</div>
                            <div className="text-xs text-muted-foreground">Normal range: 80-120 ms</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ECGVisualization
                    patientId={patientId}
                    isMock={true}
                    title="12-Lead ECG"
                    height="300px"
                    showControls={false}
                  />
                  
                  <ECGVisualization
                    patientId={patientId}
                    isMock={true}
                    title="ECG with Events"
                    height="300px"
                    showControls={false}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="hrv" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Heart Rate Variability (HRV) Analysis</CardTitle>
                <CardDescription>
                  Analysis of variations in time intervals between heartbeats
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
                  <div className="lg:col-span-2">
                    <HRVVisualization
                      patientId={patientId}
                      isMock={true}
                      title="Heart Rate Variability (Time Domain)"
                      height="350px"
                      showControls={true}
                    />
                  </div>
                  
                  <div>
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-lg">HRV Metrics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <div className="text-sm font-medium">SDNN</div>
                            <div className="text-2xl font-bold">42 ms</div>
                            <div className="text-xs text-muted-foreground">Standard deviation of NN intervals</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">RMSSD</div>
                            <div className="text-2xl font-bold">38 ms</div>
                            <div className="text-xs text-muted-foreground">Root mean square of successive differences</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">pNN50</div>
                            <div className="text-2xl font-bold">26.4%</div>
                            <div className="text-xs text-muted-foreground">Percentage of successive RR intervals that differ by more than 50 ms</div>
                          </div>
                          
                          <div>
                            <div className="text-sm font-medium">HRV Triangular Index</div>
                            <div className="text-2xl font-bold">14.2</div>
                            <div className="text-xs text-muted-foreground">Total number of NN intervals / height of the histogram</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-md">Frequency Domain Analysis</CardTitle>
                      <CardDescription>Power spectral density analysis</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[300px] flex items-center justify-center">
                      <div className="text-muted-foreground">Frequency domain visualization coming soon</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-md">Poincaré Plot</CardTitle>
                      <CardDescription>Visualization of the correlation between successive RR intervals</CardDescription>
                    </CardHeader>
                    <CardContent className="h-[300px] flex items-center justify-center">
                      <div className="text-muted-foreground">Poincaré plot visualization coming soon</div>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
} 