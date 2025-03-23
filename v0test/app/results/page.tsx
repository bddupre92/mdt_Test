"use client"

import { useState, useEffect } from "react"
import { ResultsDashboard } from "@/components/results/ResultsDashboard"
import { RunControlPanel } from "@/components/meta-optimizer/RunControlPanel"
import { MetaOptimizerVisualization } from "@/components/visualizations/MetaOptimizerVisualization"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function ResultsPage() {
  const [metaOptimizerResults, setMetaOptimizerResults] = useState<any>(null);

  const handleResultsReceived = (results: any) => {
    console.log("Received Meta-Optimizer results:", results);
    setMetaOptimizerResults(results);
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Results Dashboard</h1>
      
      <Tabs defaultValue="dashboard" className="mb-8">
        <TabsList className="mb-4">
          <TabsTrigger value="dashboard">Results Dashboard</TabsTrigger>
          <TabsTrigger value="meta-optimizer">Meta-Optimizer Analysis</TabsTrigger>
        </TabsList>
        
        <TabsContent value="dashboard">
          <ResultsDashboard 
            showSearch={true}
            showCompare={true}
          />
        </TabsContent>
        
        <TabsContent value="meta-optimizer">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <RunControlPanel onResultsReceived={handleResultsReceived} />
            </div>
            
            <div className="lg:col-span-2">
              {metaOptimizerResults ? (
                <MetaOptimizerVisualization result={metaOptimizerResults} />
              ) : (
                <div className="bg-muted p-8 rounded-lg text-center">
                  <h3 className="text-lg font-medium mb-2">No Meta-Optimizer Results</h3>
                  <p className="text-muted-foreground">
                    Run a Meta-Optimizer analysis to see visualization results.
                  </p>
                </div>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 