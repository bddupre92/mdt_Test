"use client"

import { useState } from "react"
import { Button } from "../../components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "../../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { LineChart } from "../../components/charts"
import { Label } from "../../components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import { Input } from "../../components/ui/input"
import { Slider } from "../../components/ui/slider"
import { Switch } from "../../components/ui/switch"
import { Info } from "lucide-react"

// Define local interfaces to match charts.tsx
interface DataPoint {
  x: string | number
  y: number
  color?: string
  size?: number
}

export default function PredictionsPage() {
  const [selectedPatient, setSelectedPatient] = useState<string>("")
  const [timeframe, setTimeframe] = useState<string>("7d")
  const [threshold, setThreshold] = useState<number>(70)
  const [algorithm, setAlgorithm] = useState<string>("meta-optimizer")
  const [predictionResults, setPredictionResults] = useState<any>(null)
  
  const generatePredictions = () => {
    // Simulated prediction generation
    const results = {
      patientId: selectedPatient,
      predictions: Array.from({ length: 14 }, (_, i) => ({
        date: new Date(Date.now() + i * 24 * 60 * 60 * 1000),
        probability: Math.random() * 100,
        features: {
          stress: Math.random() * 10,
          sleep: Math.random() * 10,
          hydration: Math.random() * 10,
          weather: Math.random() * 10,
        }
      }))
    }
    
    setPredictionResults(results)
  }
  
  // Convert prediction results to compatible DataPoint format
  const probabilityData: DataPoint[] = predictionResults ? 
    predictionResults.predictions.map((p: any) => ({
      x: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      y: p.probability
    })) : []
  
  // Feature importance data in DataPoint format
  const featureImportanceData: DataPoint[] = predictionResults ? [
    { x: 'Stress', y: Math.random() * 100 },
    { x: 'Sleep', y: Math.random() * 100 },
    { x: 'Hydration', y: Math.random() * 100 },
    { x: 'Weather', y: Math.random() * 100 }
  ] : []

  return (
    <div className="container py-10">
      <h1 className="text-3xl font-bold tracking-tight mb-6">Migraine Predictions</h1>
      <p className="text-muted-foreground mb-8">
        View and analyze migraine predictions based on physiological data and optimized algorithms.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Prediction Controls</CardTitle>
            <CardDescription>Configure prediction parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="patient">Patient ID</Label>
              <Select value={selectedPatient} onValueChange={setSelectedPatient}>
                <SelectTrigger id="patient">
                  <SelectValue placeholder="Select patient" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="patient-001">Patient 001</SelectItem>
                  <SelectItem value="patient-002">Patient 002</SelectItem>
                  <SelectItem value="patient-003">Patient 003</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="timeframe">Prediction Timeframe</Label>
              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger id="timeframe">
                  <SelectValue placeholder="Select timeframe" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3d">3 Days</SelectItem>
                  <SelectItem value="7d">7 Days</SelectItem>
                  <SelectItem value="14d">14 Days</SelectItem>
                  <SelectItem value="30d">30 Days</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="threshold">Probability Threshold: {threshold}%</Label>
              </div>
              <Slider 
                id="threshold" 
                min={0} 
                max={100} 
                step={1} 
                value={[threshold]} 
                onValueChange={(value) => setThreshold(value[0])}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="algorithm">Prediction Algorithm</Label>
              <Select value={algorithm} onValueChange={setAlgorithm}>
                <SelectTrigger id="algorithm">
                  <SelectValue placeholder="Select algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="differential-evolution">Differential Evolution</SelectItem>
                  <SelectItem value="evolution-strategy">Evolution Strategy</SelectItem>
                  <SelectItem value="ant-colony">Ant Colony Optimization</SelectItem>
                  <SelectItem value="grey-wolf">Grey Wolf Optimizer</SelectItem>
                  <SelectItem value="meta-optimizer">Meta-Optimizer (Ensemble)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch id="use-physiological" />
              <Label htmlFor="use-physiological">Include physiological data</Label>
            </div>
          </CardContent>
          <CardFooter>
            <Button className="w-full" onClick={generatePredictions}>Generate Predictions</Button>
          </CardFooter>
        </Card>
        
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>
              {predictionResults ? 
                `Predictions for Patient ${selectedPatient} using ${algorithm}` : 
                "Generate predictions to see results"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {predictionResults ? (
              <Tabs defaultValue="probability">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="probability">Probability Timeline</TabsTrigger>
                  <TabsTrigger value="features">Feature Importance</TabsTrigger>
                </TabsList>
                <TabsContent value="probability" className="h-[300px]">
                  <LineChart data={probabilityData} />
                </TabsContent>
                <TabsContent value="features" className="h-[300px]">
                  <LineChart data={featureImportanceData} />
                </TabsContent>
              </Tabs>
            ) : (
              <div className="h-[300px] flex items-center justify-center border border-dashed rounded-md">
                <div className="text-center">
                  <p className="text-muted-foreground mb-4">Select a patient and generate predictions</p>
                  <Info className="mx-auto h-12 w-12 text-muted-foreground opacity-50" />
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 