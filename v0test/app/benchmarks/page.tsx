"use client"

import { useState } from "react"
import { Button } from "../../components/ui/button"
import { BenchmarkComparison } from "../../components/benchmark-comparison"
import { OptimizerResult, OptimizerType } from "../../lib/optimizers"
import SATZillaPrediction from "../../components/satzilla-prediction"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"

// Define the structure for SATZilla-specific data
interface SATZillaData {
  mlPredictions?: Array<{
    optimizerId: string;
    confidence: number;
  }>;
  problemFeatures?: Record<string, string | number>;
  predictionQuality?: Array<{
    algorithm: string;
    predictedPerformance: number;
    actualPerformance: number;
  }>;
  metrics?: {
    selectedOptimizer?: string;
    selectionAccuracy?: number;
    [key: string]: any;
  };
}

// Define optimizers for the benchmark
const optimizers = [
  { id: "genetic", name: "Genetic Algorithm" },
  { id: "particle_swarm", name: "Particle Swarm Optimization" },
  { id: "simulated_annealing", name: "Simulated Annealing" },
  { id: "differential_evolution", name: "Differential Evolution" },
  { id: "cmaes", name: "CMA-ES" },
];

export default function BenchmarksPage() {
  const [results, setResults] = useState<(Record<OptimizerType, OptimizerResult[]> & SATZillaData) | null>(null)
  const [activeBenchmark, setActiveBenchmark] = useState<string | null>(null)
  const [surrogateModeEnabled, setSurrogateModeEnabled] = useState(false)
  
  const handleRunBenchmark = (newResults: Record<OptimizerType, OptimizerResult[]> & SATZillaData) => {
    setResults(newResults)
    // Set active benchmark and enable surrogate mode by default
    setActiveBenchmark("Sphere")
    setSurrogateModeEnabled(true)
  }

  return (
    <div className="container py-10">
      <h1 className="text-3xl font-bold tracking-tight mb-6">Model Benchmarks</h1>
      <p className="text-muted-foreground mb-8">
        Compare and evaluate different algorithm performances for migraine prediction.
      </p>
      
      <BenchmarkComparison 
        results={results}
        onRunBenchmark={handleRunBenchmark}
      />

      {activeBenchmark && results && (
        <div className="mt-6 space-y-6">
          <h2 className="text-2xl font-bold">Benchmark Results</h2>
          
          <Tabs defaultValue="charts" className="mt-4">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="charts">Performance Charts</TabsTrigger>
              <TabsTrigger value="metrics">Detailed Metrics</TabsTrigger>
              {surrogateModeEnabled && <TabsTrigger value="ml-selection">ML Selection</TabsTrigger>}
            </TabsList>
            
            <TabsContent value="charts">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Keep existing chart components */}
              </div>
            </TabsContent>
            
            <TabsContent value="metrics">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Keep existing metric components */}
              </div>
            </TabsContent>
            
            <TabsContent value="ml-selection">
              {surrogateModeEnabled && results && (
                <SATZillaPrediction
                  predictions={results.mlPredictions || []}
                  problemFeatures={results.problemFeatures || {}}
                  predictionQuality={results.predictionQuality || []}
                  selectedOptimizer={results.metrics?.selectedOptimizer || ""}
                  algorithmNames={optimizers.reduce((acc: Record<string, string>, opt: { id: string, name: string }) => {
                    acc[opt.id] = opt.name;
                    return acc;
                  }, {})}
                />
              )}
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  )
} 