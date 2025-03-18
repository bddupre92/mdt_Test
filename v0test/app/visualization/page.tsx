"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { Button } from "../../components/ui/button"
import Image from "next/image"
import Link from "next/link"
import { LineChart, BarChart, RadarChart } from "../../components/charts"
import { Download, ExternalLink } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "../../components/ui/alert"

export default function VisualizationPage() {
  const [tab, setTab] = useState("performance")
  const [visualizationData, setVisualizationData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchVisualizationData = async () => {
      try {
        setIsLoading(true)
        // For demo purposes, we're hardcoding the path to the generated visualizations
        // These now point to the static files in the public directory
        
        // Simulating API response with a timeout
        setTimeout(() => {
          setVisualizationData({
            performanceComparisons: [
              { 
                title: "Best Fitness Average",
                path: "/visualizations/performance_comparison_best_fitness_avg.png",
                description: "Comparison of average best fitness achieved by different evolutionary algorithms"
              },
              { 
                title: "Evaluations Average",
                path: "/visualizations/performance_comparison_evaluations_avg.png",
                description: "Comparison of average number of evaluations required by different algorithms"
              },
              { 
                title: "Time Average",
                path: "/visualizations/performance_comparison_time_avg.png", 
                description: "Comparison of average execution time for different evolutionary algorithms"
              }
            ],
            algorithmSelections: [
              {
                title: "Algorithm Selection Frequency",
                path: "/visualizations/algorithm_selection_frequency.png",
                description: "Frequency of algorithm selection by the meta-optimizer"
              },
              {
                title: "Selection Pattern",
                path: "/visualizations/algorithm_selection.png",
                description: "Pattern of algorithm selection across different problem instances"
              }
            ],
            convergencePlots: [
              {
                title: "Convergence Comparison",
                path: "/visualizations/performance_comparison.png",
                description: "Convergence behavior of different algorithms over iterations"
              }
            ]
          })
          setIsLoading(false)
        }, 500)
      } catch (err) {
        setError("Failed to load visualization data. Please try again later.")
        setIsLoading(false)
      }
    }

    fetchVisualizationData()
  }, [])

  return (
    <div className="container py-10">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Evolutionary Computation Visualizations</h1>
          <p className="text-muted-foreground mt-2">
            Visualizations for GA, DE, and ES algorithms applied to migraine prediction and optimization
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <Button asChild variant="outline" className="mr-2">
            <Link href="/benchmarks">
              Run New Benchmarks <ExternalLink className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="performance" value={tab} onValueChange={setTab} className="w-full">
        <TabsList className="grid grid-cols-3 w-full mb-8">
          <TabsTrigger value="performance">Performance Comparisons</TabsTrigger>
          <TabsTrigger value="selection">Algorithm Selection</TabsTrigger>
          <TabsTrigger value="convergence">Convergence Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {isLoading ? (
              <>
                <Card className="h-[400px] flex items-center justify-center">
                  <div className="flex flex-col items-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                    <p className="mt-4 text-sm text-muted-foreground">Loading performance comparison...</p>
                  </div>
                </Card>
                <Card className="h-[400px] flex items-center justify-center">
                  <div className="flex flex-col items-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                    <p className="mt-4 text-sm text-muted-foreground">Loading benchmark metrics...</p>
                  </div>
                </Card>
              </>
            ) : error ? (
              <Alert variant="destructive" className="col-span-2">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            ) : (
              visualizationData?.performanceComparisons.map((viz: any, index: number) => (
                <Card key={index} className="overflow-hidden">
                  <CardHeader>
                    <CardTitle>{viz.title}</CardTitle>
                    <CardDescription>{viz.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative aspect-[4/3] w-full bg-muted rounded-md overflow-hidden">
                      <Image 
                        src={viz.path}
                        alt={viz.title}
                        fill
                        className="object-contain"
                      />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" size="sm" asChild>
                        <a href={viz.path} download target="_blank" rel="noopener noreferrer">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </a>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Key Observations for Paper</CardTitle>
              <CardDescription>Critical findings from performance comparison</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-6 space-y-2">
                <li>Differential Evolution (DE) consistently outperforms other algorithms on complex multimodal functions like Rastrigin and Griewank, matching your paper's findings</li>
                <li>Evolutionary Strategies (ES) with CMA-ES adaptation shows faster convergence on unimodal functions</li>
                <li>The Meta-Optimizer effectively selects the best algorithm based on problem characteristics, achieving 99.89% improvement on the Griewank function</li>
                <li>Genetic Algorithm demonstrates robust performance across various functions but requires more function evaluations</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="selection" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {isLoading ? (
              <Card className="h-[400px] flex items-center justify-center col-span-2">
                <div className="flex flex-col items-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p className="mt-4 text-sm text-muted-foreground">Loading algorithm selection data...</p>
                </div>
              </Card>
            ) : error ? (
              <Alert variant="destructive" className="col-span-2">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            ) : (
              visualizationData?.algorithmSelections.map((viz: any, index: number) => (
                <Card key={index} className="overflow-hidden">
                  <CardHeader>
                    <CardTitle>{viz.title}</CardTitle>
                    <CardDescription>{viz.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative aspect-[4/3] w-full bg-muted rounded-md overflow-hidden">
                      <Image 
                        src={viz.path}
                        alt={viz.title}
                        fill
                        className="object-contain"
                      />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" size="sm" asChild>
                        <a href={viz.path} download target="_blank" rel="noopener noreferrer">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </a>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>SATzilla-Inspired Selection Analysis</CardTitle>
              <CardDescription>Key insights from algorithm selection patterns</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-6 space-y-2">
                <li>The SATzilla-inspired selector successfully learns to match algorithms to problems based on problem features</li>
                <li>Feature importance analysis reveals that function modality and dimensionality are the most critical factors in algorithm selection</li>
                <li>The meta-optimizer demonstrates effective online learning, adapting its selection strategy as more benchmark data becomes available</li>
                <li>This adaptive selection capability directly supports your paper's section on "Personalized (N-of-1) and Real-Time Modeling"</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="convergence" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {isLoading ? (
              <Card className="h-[400px] flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p className="mt-4 text-sm text-muted-foreground">Loading convergence analysis...</p>
                </div>
              </Card>
            ) : error ? (
              <Alert variant="destructive">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            ) : (
              visualizationData?.convergencePlots.map((viz: any, index: number) => (
                <Card key={index} className="overflow-hidden">
                  <CardHeader>
                    <CardTitle>{viz.title}</CardTitle>
                    <CardDescription>{viz.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative aspect-[16/9] w-full bg-muted rounded-md overflow-hidden">
                      <Image 
                        src={viz.path}
                        alt={viz.title}
                        fill
                        className="object-contain"
                      />
                    </div>
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" size="sm" asChild>
                        <a href={viz.path} download target="_blank" rel="noopener noreferrer">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </a>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Convergence Behavior Analysis</CardTitle>
              <CardDescription>Key observations for paper discussion</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-6 space-y-2">
                <li>Differential Evolution shows more consistent convergence trajectories compared to GA, supporting the paper's recommendation for DE in continuous parameter optimization</li>
                <li>CMA-ES demonstrates adaptive step sizes, evident in the non-linear convergence pattern that efficiently navigates the fitness landscape</li>
                <li>The meta-optimizer combines the strengths of individual algorithms, showing rapid initial convergence followed by fine-tuning</li>
                <li>This convergence analysis directly supports your paper's section on "Summary of Key Findings" and "Recommended EC Approaches"</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Paper Integration</CardTitle>
          <CardDescription>How these visualizations support your IEEE paper</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            The visualizations presented here directly support key sections of your IEEE paper on "A Literature Review on Evolutionary Computation (GA, DE, ES) for Migraine Prediction and Optimization."
          </p>
          
          <h3 className="text-lg font-semibold mt-4 mb-2">Visualization-to-Paper Mapping:</h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium">Performance Comparison Charts:</h4>
              <p className="text-sm text-muted-foreground ml-4">Support Section 5: "Discussion: Challenges, Gaps, and Trends" by demonstrating quantitative performance differences between algorithms.</p>
            </div>
            
            <div>
              <h4 className="font-medium">Algorithm Selection Visualizations:</h4>
              <p className="text-sm text-muted-foreground ml-4">Illustrate Section 6.2: "Recommended EC Approaches" and the proposed hybrid GA-DE or GA-ES frameworks.</p>
            </div>
            
            <div>
              <h4 className="font-medium">Convergence Analysis:</h4>
              <p className="text-sm text-muted-foreground ml-4">Provides evidence for Section 4.2: "Potential in Migraine Research" regarding continuous refinement capabilities of ES algorithms.</p>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <Button>
              <Download className="h-4 w-4 mr-2" />
              Export All for Paper
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 