"use client";

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "../ui/input"
import { Label } from "../ui/label"
import { Loader2, Play } from "lucide-react"
import { Spinner } from "@/components/ui/spinner"
import { MetaOptimizerVisualizer } from '@/components/visualizations/MetaOptimizerVisualizer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from "@/components/ui/use-toast"
import path from "path"

interface VisualizationFile {
  name: string
  path: string
  url: string
}

export function RunControlPanel() {
  const [selectedCommand, setSelectedCommand] = useState<string>('baseline_comparison')
  const [loading, setLoading] = useState<boolean>(false)
  const [commandOutput, setCommandOutput] = useState<string>('')
  const [outputPath, setOutputPath] = useState<string>('')
  const [resultData, setResultData] = useState<any>(null)
  const [visualizationFiles, setVisualizationFiles] = useState<VisualizationFile[]>([])
  const { toast } = useToast()

  // Define available commands with display names and descriptions
  const commands = [
    {
      value: 'baseline_comparison',
      label: 'Baseline Comparison',
      description: 'Compare multiple optimization algorithms on benchmark functions'
    },
    {
      value: 'train_satzilla',
      label: 'Train SATzilla',
      description: 'Train the SATzilla algorithm selector using benchmark data'
    },
    {
      value: 'meta',
      label: 'Meta-Optimizer',
      description: 'Run the meta-optimizer on benchmark problems'
    },
    {
      value: 'dynamic_optimization',
      label: 'Dynamic Optimization',
      description: 'Test optimization algorithms on dynamic problems'
    }
  ]

  // Helper function to execute meta-optimizer command
  const executeMetaOptimizerCommand = async (commandName: string) => {
    try {
      console.log('Sending meta-optimizer command to API:', commandName);
      
      const response = await fetch('/api/framework/execute-meta-optimizer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          command: commandName
        }),
      });
      
      // Log raw response status
      console.log('API response status:', response.status);
      
      // Always get the response text first to ensure we can see it
      const responseText = await response.text();
      console.log('API response text:', responseText);
      
      // Try to parse the JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse response as JSON:', e);
        return {
          success: false,
          error: 'Invalid response format from server',
          responseText
        };
      }
      
      if (!response.ok) {
        console.error('Response not OK:', data);
        return {
          success: false,
          error: data?.error || 'Command execution failed',
          details: data?.details,
          status: response.status
        };
      }
      
      return {
        success: true,
        ...data
      };
    } catch (error) {
      console.error('Failed to execute command:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  };

  const runMetaOptimizerAnalysis = async () => {
    try {
      setLoading(true)
      setCommandOutput('')
      setVisualizationFiles([])
      setResultData(null)
      
      // Execute the command using our helper function
      const result = await executeMetaOptimizerCommand(selectedCommand);
      
      // Update state with results
      setResultData(result.result || result);
      setCommandOutput(result.success ? result.output || 'Command executed successfully' : `Error: ${result.error}`);
      setOutputPath(result.outputDir || '');
      
      // Check for visualization files
      if (result.visualizationFiles && result.visualizationFiles.length > 0) {
        setVisualizationFiles(result.visualizationFiles);
      } else if (result.outputDir) {
        // If API didn't return visualization files, check the output directory manually
        await checkDirectoryForVisualizationFiles(result.outputDir);
      }
      
      // Show success notification
      toast({
        title: 'Analysis Complete',
        description: `${commands.find(c => c.value === selectedCommand)?.label} completed successfully`,
      });
    } catch (error: any) {
      console.error('Error running Meta-Optimizer:', error);
      setCommandOutput(`Error: ${error.message}`);
      toast({
        title: 'Analysis Failed',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }
  
  // Function to manually check for visualization files in the output directory
  const checkDirectoryForVisualizationFiles = async (directory: string) => {
    try {
      const response = await fetch(`/api/file?path=${encodeURIComponent(directory)}&list=true`)
      
      if (!response.ok) {
        console.warn(`Could not check directory ${directory}: ${response.statusText}`)
        return
      }
      
      const data = await response.json()
      
      if (data.files && Array.isArray(data.files)) {
        const imageFiles = data.files
          .filter((file: any) => 
            file.name.endsWith('.png') || 
            file.name.endsWith('.jpg') || 
            file.name.endsWith('.svg')
          )
          .map((file: any) => ({
            name: file.name,
            path: `${directory}/${file.name}`,
            url: `/api/file?path=${encodeURIComponent(`${directory}/${file.name}`)}`
          }))
        
        if (imageFiles.length > 0) {
          setVisualizationFiles(imageFiles)
        }
      }
    } catch (err) {
      console.warn('Error checking for visualization files:', err)
    }
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Framework Runner</CardTitle>
          <CardDescription>
            Run Meta-Optimizer analyses and visualize results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid gap-4">
              <div className="space-y-2">
                <label htmlFor="command-select" className="text-sm font-medium">
                  Select Analysis
                </label>
                <Select
                  value={selectedCommand}
                  onValueChange={setSelectedCommand}
                  disabled={loading}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select analysis type" />
                  </SelectTrigger>
                  <SelectContent>
                    {commands.map((command) => (
                      <SelectItem key={command.value} value={command.value}>
                        <div className="flex flex-col">
                          <span>{command.label}</span>
                          <span className="text-xs text-muted-foreground">{command.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          <div className="text-sm text-muted-foreground">
            {selectedCommand && commands.find(c => c.value === selectedCommand)?.description}
          </div>
          <Button 
            onClick={runMetaOptimizerAnalysis} 
            disabled={loading}
          >
            {loading && <Spinner className="mr-2 h-4 w-4" />}
            {loading ? 'Running...' : 'Run Analysis'}
          </Button>
        </CardFooter>
      </Card>
      
      {/* Results display */}
      {(visualizationFiles.length > 0 || loading || commandOutput) && (
        <Tabs defaultValue="visualizations">
          <TabsList>
            <TabsTrigger value="visualizations">
              Visualizations {visualizationFiles.length > 0 && `(${visualizationFiles.length})`}
            </TabsTrigger>
            <TabsTrigger value="output">Command Output</TabsTrigger>
          </TabsList>
          
          <TabsContent value="visualizations" className="mt-4">
            <MetaOptimizerVisualizer 
              visualizationFiles={visualizationFiles}
              resultData={resultData}
              outputDir={outputPath}
              isLoading={loading}
              onRefresh={() => outputPath && checkDirectoryForVisualizationFiles(outputPath)}
            />
          </TabsContent>
          
          <TabsContent value="output" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Command Output</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center p-4">
                    <Spinner className="mr-2 h-4 w-4" />
                    <span>Running command...</span>
                  </div>
                ) : commandOutput ? (
                  <pre className="bg-muted p-4 rounded-md overflow-auto max-h-[400px] text-sm">
                    {commandOutput}
                  </pre>
                ) : (
                  <p className="text-muted-foreground">No output available</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
} 