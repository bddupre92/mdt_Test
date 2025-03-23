"use client";

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { RunControlPanel } from '@/components/meta-optimizer/RunControlPanel'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'

export default function MetaOptimizerPage() {
  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-6">Meta-Optimizer Framework</h1>
      
      <Tabs defaultValue="runner" className="space-y-4">
        <TabsList>
          <TabsTrigger value="runner">Framework Runner</TabsTrigger>
          <TabsTrigger value="results">Results Dashboard</TabsTrigger>
          <TabsTrigger value="docs">Documentation</TabsTrigger>
        </TabsList>
        
        <TabsContent value="runner" className="space-y-4">
          <RunControlPanel />
        </TabsContent>
        
        <TabsContent value="results" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Results Dashboard</CardTitle>
              <CardDescription>
                View and compare results from previous Meta-Optimizer runs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Run the Meta-Optimizer from the Framework Runner tab to generate results.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="docs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Meta-Optimizer Documentation</CardTitle>
              <CardDescription>
                Learn more about the Meta-Optimizer framework
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="bg-muted rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">Command Line Interface</h3>
                  <p className="text-sm mb-2">
                    Learn how to use the Meta-Optimizer from the command line.
                  </p>
                  <a 
                    href="/docs/command_line_interface.md" 
                    target="_blank" 
                    className="text-sm text-blue-500 hover:underline"
                  >
                    View Documentation
                  </a>
                </div>
                
                <div className="bg-muted rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">Visualization Guide</h3>
                  <p className="text-sm mb-2">
                    Understand the different visualizations produced by the Meta-Optimizer.
                  </p>
                  <a 
                    href="/docs/visualization_guide.md" 
                    target="_blank" 
                    className="text-sm text-blue-500 hover:underline"
                  >
                    View Documentation
                  </a>
                </div>
                
                <div className="bg-muted rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">Drift Detection</h3>
                  <p className="text-sm mb-2">
                    Learn about the drift detection capabilities of the Meta-Optimizer.
                  </p>
                  <a 
                    href="/docs/drift_detection_guide.md" 
                    target="_blank" 
                    className="text-sm text-blue-500 hover:underline"
                  >
                    View Documentation
                  </a>
                </div>
                
                <div className="bg-muted rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-2">Dynamic Optimization</h3>
                  <p className="text-sm mb-2">
                    Learn about the dynamic optimization capabilities of the Meta-Optimizer.
                  </p>
                  <a 
                    href="/docs/dynamic_optimization_guide.md" 
                    target="_blank" 
                    className="text-sm text-blue-500 hover:underline"
                  >
                    View Documentation
                  </a>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 