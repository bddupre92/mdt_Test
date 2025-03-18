"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"

interface FrameworkVisualizationProps {
  onComponentSelect?: (component: string) => void
}

export function FrameworkVisualization({ onComponentSelect }: FrameworkVisualizationProps) {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)

  const handleComponentClick = (component: string) => {
    setSelectedComponent(component)
    if (onComponentSelect) {
      onComponentSelect(component)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Framework Architecture</CardTitle>
        <CardDescription>Interactive visualization of the optimization framework</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="diagram">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="diagram">Architecture</TabsTrigger>
            <TabsTrigger value="components">Components</TabsTrigger>
            <TabsTrigger value="flow">Data Flow</TabsTrigger>
          </TabsList>

          <TabsContent value="diagram" className="space-y-4">
            <div className="relative w-full h-[400px] border rounded-md p-4 overflow-hidden">
              {/* Main components */}
              <div
                className={`absolute top-4 left-1/2 transform -translate-x-1/2 w-40 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                  selectedComponent === "main" ? "bg-primary/20 border-primary" : "bg-muted hover:bg-muted/80"
                }`}
                onClick={() => handleComponentClick("main")}
              >
                <span className="font-medium text-sm">main.py</span>
              </div>

              <div className="absolute top-32 left-1/2 transform -translate-x-1/2 w-1 h-12 bg-muted-foreground/50"></div>

              {/* Component row 1 */}
              <div className="absolute top-44 left-0 right-0 flex justify-center space-x-8">
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "optimizers" ? "bg-primary/20 border-primary" : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("optimizers")}
                >
                  <span className="font-medium text-sm">Optimizers</span>
                </div>
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "meta-learner" ? "bg-primary/20 border-primary" : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("meta-learner")}
                >
                  <span className="font-medium text-sm">Meta-Learner</span>
                </div>
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "explainability"
                      ? "bg-primary/20 border-primary"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("explainability")}
                >
                  <span className="font-medium text-sm">Explainability</span>
                </div>
              </div>

              {/* Connecting lines */}
              <div className="absolute top-60 left-[calc(50%-64px)] w-1 h-12 bg-muted-foreground/50"></div>
              <div className="absolute top-60 left-1/2 transform -translate-x-1/2 w-1 h-12 bg-muted-foreground/50"></div>
              <div className="absolute top-60 left-[calc(50%+64px)] w-1 h-12 bg-muted-foreground/50"></div>

              {/* Component row 2 */}
              <div className="absolute top-72 left-0 right-0 flex justify-center space-x-8">
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "benchmarking" ? "bg-primary/20 border-primary" : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("benchmarking")}
                >
                  <span className="font-medium text-sm">Benchmarking</span>
                </div>
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "drift-detection"
                      ? "bg-primary/20 border-primary"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("drift-detection")}
                >
                  <span className="font-medium text-sm">Drift Detection</span>
                </div>
                <div
                  className={`w-32 h-16 border rounded-md flex items-center justify-center cursor-pointer ${
                    selectedComponent === "utilities" ? "bg-primary/20 border-primary" : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => handleComponentClick("utilities")}
                >
                  <span className="font-medium text-sm">Utilities</span>
                </div>
              </div>
            </div>

            {selectedComponent && (
              <div className="p-4 border rounded-md bg-muted/30">
                <h3 className="text-sm font-medium mb-2">
                  {selectedComponent.charAt(0).toUpperCase() + selectedComponent.slice(1)} Component
                </h3>
                <p className="text-sm text-muted-foreground">
                  {selectedComponent === "main" &&
                    "Main entry point for the framework. Handles command-line arguments and orchestrates component interactions."}
                  {selectedComponent === "optimizers" &&
                    "Implementations of various optimization algorithms including Differential Evolution, Evolution Strategy, Ant Colony, and Grey Wolf."}
                  {selectedComponent === "meta-learner" &&
                    "System for selecting the best optimizer for a given problem based on problem characteristics and historical performance."}
                  {selectedComponent === "explainability" &&
                    "Tools for explaining optimizer behavior and model predictions using techniques like SHAP, LIME, and feature importance."}
                  {selectedComponent === "benchmarking" &&
                    "Tools for evaluating and comparing optimizers on test functions and real-world problems."}
                  {selectedComponent === "drift-detection" &&
                    "System for detecting and adapting to concept drift in data distributions."}
                  {selectedComponent === "utilities" &&
                    "Common utilities used across the framework including logging, visualization, and configuration management."}
                </p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="components">
            <ScrollArea className="h-[400px] w-full rounded-md border p-4">
              <div className="space-y-4">
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Optimizers</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("differential-evolution")}
                    >
                      Differential Evolution
                      <Badge className="ml-2" variant="outline">
                        DE
                      </Badge>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("evolution-strategy")}
                    >
                      Evolution Strategy
                      <Badge className="ml-2" variant="outline">
                        ES
                      </Badge>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("ant-colony")}
                    >
                      Ant Colony
                      <Badge className="ml-2" variant="outline">
                        ACO
                      </Badge>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("grey-wolf")}
                    >
                      Grey Wolf
                      <Badge className="ml-2" variant="outline">
                        GWO
                      </Badge>
                    </Button>
                  </div>
                </div>

                <Separator />

                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Meta-Learning</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("meta-optimizer")}
                    >
                      Meta-Optimizer
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("meta-learner")}
                    >
                      Meta-Learner
                    </Button>
                  </div>
                </div>

                <Separator />

                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Explainability</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("shap")}
                    >
                      SHAP Explainer
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("lime")}
                    >
                      LIME Explainer
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("feature-importance")}
                    >
                      Feature Importance
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("optimizer-explainer")}
                    >
                      Optimizer Explainer
                    </Button>
                  </div>
                </div>

                <Separator />

                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Drift Detection</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("drift-detector")}
                    >
                      Drift Detector
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("drift-analyzer")}
                    >
                      Drift Analyzer
                    </Button>
                  </div>
                </div>

                <Separator />

                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Benchmarking</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("test-functions")}
                    >
                      Test Functions
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-start"
                      onClick={() => handleComponentClick("performance-metrics")}
                    >
                      Performance Metrics
                    </Button>
                  </div>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="flow">
            <div className="p-4 border rounded-md bg-muted/30 h-[400px] overflow-auto">
              <h3 className="text-sm font-medium mb-2">Data Flow</h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <h4 className="text-xs font-medium">1. Optimization Flow</h4>
                  <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                    <li>User specifies optimization parameters</li>
                    <li>OptimizerFactory creates optimizer instances</li>
                    <li>Optimizers run on specified problems</li>
                    <li>Results are collected and visualized</li>
                  </ul>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-medium">2. Meta-Learning Flow</h4>
                  <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                    <li>User specifies meta-learning parameters</li>
                    <li>MetaLearner extracts problem characteristics</li>
                    <li>MetaLearner selects the best optimizer based on historical performance</li>
                    <li>Selected optimizer runs on the problem</li>
                    <li>Results are collected and used to update the meta-model</li>
                  </ul>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-medium">3. Explainability Flow</h4>
                  <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                    <li>User specifies explainability parameters</li>
                    <li>ExplainerFactory creates explainer instances</li>
                    <li>Explainers generate explanations for optimizer behavior or model predictions</li>
                    <li>Explanations are visualized and summarized</li>
                  </ul>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-medium">4. Drift Detection Flow</h4>
                  <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                    <li>User specifies drift detection parameters</li>
                    <li>DriftDetector monitors data streams for drift</li>
                    <li>When drift is detected, DriftAdapter adapts the system</li>
                    <li>Results are collected and visualized</li>
                  </ul>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

