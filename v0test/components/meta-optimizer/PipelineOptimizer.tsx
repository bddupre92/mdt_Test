import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, ArrowRight, Check, Edit, Trash2, Plus, MoveHorizontal, Loader2, LineChart } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { DragDropContext, Droppable, Draggable, DropResult } from "react-beautiful-dnd";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  savePipeline as apiSavePipeline, 
  optimizePipeline,
  deletePipeline as apiDeletePipeline,
  listPipelines as apiListPipelines,
  getPipelineExecutionHistory,
  Pipeline,
  PipelineStage,
  PipelineExecutionResult,
  PipelineOptimizationConfig
} from '@/lib/api/meta-optimizer/pipeline';

// Define ProblemCharacteristics interface since it's not exported
interface ProblemCharacteristics {
  datasetSize: number;
  featureCount: number;
  targetType: 'binary' | 'multiclass' | 'regression';
  missingValues: boolean;
  imbalanced: boolean;
}

// Update PipelineExecutionResult interface to include optimization path
interface ExtendedPipelineExecutionResult extends PipelineExecutionResult {
  optimizationPath?: Array<{
    iteration: number;
    metrics: Record<string, number>;
    parameters: Record<string, any>;
  }>;
}

// Update optimization result interface from API
interface OptimizationResult {
  optimizedPipeline: Pipeline;
  improvementMetrics: Record<string, { 
    before: number; 
    after: number; 
    improvement: string; 
  }>;
  optimizationPath: Array<{
    iteration: number;
    parameters: Record<string, any>;
    score: number;
    metrics: Record<string, number>;
  }>;
  executionId: string;
}

// Generate a unique ID
const generateId = () => Math.random().toString(36).substring(2, 9);

export interface OptimizerConfig {
  algorithm: string;
  parameters: Record<string, unknown>;
  metrics: string[];
}

export interface PipelineOptimizerProps {
  availableAlgorithms: Array<{ id: string; name: string; category: string; description?: string }>;
  availableMetrics: Array<{ id: string; name: string; description?: string }>;
  pipelines?: Pipeline[];
  optimizationResults?: OptimizationResult[];
  onSavePipeline?: (pipeline: Pipeline) => Promise<void>;
  onRunPipeline?: (pipelineId: string, optimizerConfig: OptimizerConfig) => Promise<void>;
  onDeletePipeline?: (pipelineId: string) => Promise<void>;
  isRunning?: boolean;
  progress?: number;
}

// Add new interfaces
interface Dataset {
  id: string;
  name: string;
  description?: string;
  characteristics?: ProblemCharacteristics;
}

interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

export default function PipelineOptimizer({
  availableAlgorithms = [],
  availableMetrics = [],
  pipelines: initialPipelines = [],
  optimizationResults: initialResults = [],
  onSavePipeline,
  onRunPipeline,
  onDeletePipeline,
  isRunning: initialIsRunning = false,
  progress: initialProgress = 0
}: PipelineOptimizerProps) {
  // State for pipelines and UI
  const [pipelines, setPipelines] = useState<Pipeline[]>(initialPipelines);
  const [activePipeline, setActivePipeline] = useState<Pipeline | null>(
    initialPipelines.length > 0 ? initialPipelines[0] : null
  );
  const [editMode, setEditMode] = useState(initialPipelines.length === 0);
  
  // State for optimization configuration
  const [selectedOptimizer, setSelectedOptimizer] = useState('ga');
  const [autoConfigOptimizer, setAutoConfigOptimizer] = useState(true);
  const [optimizerParams, setOptimizerParams] = useState({
    populationSize: 50,
    generations: 100,
    crossoverRate: 0.8,
    mutationRate: 0.1
  });
  
  // State for API operations
  const [isLoading, setIsLoading] = useState<Record<string, boolean>>({
    pipelines: false,
    saving: false,
    deleting: false,
    running: false
  });
  const [isRunning, setIsRunning] = useState(initialIsRunning);
  const [progress, setProgress] = useState(initialProgress);
  const [optimizationResults, setOptimizationResults] = useState<OptimizationResult[]>(initialResults);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  
  // Add new state variables after the existing state declarations
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const [executionHistory, setExecutionHistory] = useState<ExtendedPipelineExecutionResult[]>([]);
  const [optimizationPath, setOptimizationPath] = useState<Array<{
    iteration: number;
    metrics: Record<string, number>;
    parameters: Record<string, any>;
  }>>([]);
  
  // Load pipelines from API when component mounts
  useEffect(() => {
    if (initialPipelines.length === 0) {
      fetchPipelines();
    }
  }, [initialPipelines.length]);
  
  // Fetch pipelines from API
  const fetchPipelines = async () => {
    setIsLoading(prev => ({ ...prev, pipelines: true }));
    try {
      const fetchedPipelines = await apiListPipelines();
      setPipelines(fetchedPipelines);
      
      if (!activePipeline && fetchedPipelines.length > 0) {
        setActivePipeline(fetchedPipelines[0]);
        // Fetch execution history for the first pipeline
        fetchExecutionHistory(fetchedPipelines[0].id);
      }
    } catch (error) {
      handleApiError(error, 'fetching pipelines');
    } finally {
      setIsLoading(prev => ({ ...prev, pipelines: false }));
    }
  };
  
  // Add new function to fetch execution history
  const fetchExecutionHistory = async (pipelineId: string) => {
    try {
      const history = await getPipelineExecutionHistory(pipelineId);
      setExecutionHistory(history as ExtendedPipelineExecutionResult[]);
      
      if (history.length > 0 && (history[0] as ExtendedPipelineExecutionResult).optimizationPath) {
        setOptimizationPath((history[0] as ExtendedPipelineExecutionResult).optimizationPath!);
      }
    } catch (error) {
      handleApiError(error, 'fetching execution history');
    }
  };
  
  // Create a new pipeline or reset current one
  const createNewPipeline = () => {
    const newPipeline: Pipeline = {
      id: generateId(),
      name: "New Pipeline",
      description: "Pipeline description",
      stages: [],
      status: 'draft'
    };
    
    setActivePipeline(newPipeline);
    setEditMode(true);
  };
  
  // Add a new stage to the pipeline
  const addStage = () => {
    if (!activePipeline) return;
    
    const defaultAlgorithm = availableAlgorithms[0] || { id: 'default', name: 'Default Algorithm' };
    
    const newStage: PipelineStage = {
      id: generateId(),
      name: `Stage ${activePipeline.stages.length + 1}`,
      algorithmId: defaultAlgorithm.id,
      algorithmName: defaultAlgorithm.name,
      parameters: {},
      description: "Stage description"
    };
    
    setActivePipeline({
      ...activePipeline,
      stages: [...activePipeline.stages, newStage]
    });
  };
  
  // Remove a stage from the pipeline
  const removeStage = (stageId: string) => {
    if (!activePipeline) return;
    
    setActivePipeline({
      ...activePipeline,
      stages: activePipeline.stages.filter(stage => stage.id !== stageId)
    });
  };
  
  // Update a stage
  const updateStage = (stageId: string, updates: Partial<PipelineStage>) => {
    if (!activePipeline) return;
    
    setActivePipeline({
      ...activePipeline,
      stages: activePipeline.stages.map(stage => 
        stage.id === stageId ? { ...stage, ...updates } : stage
      )
    });
  };
  
  // Handle stage reordering with drag and drop
  const handleDragEnd = (result: DropResult) => {
    if (!result.destination || !activePipeline) return;
    
    const stages = Array.from(activePipeline.stages);
    const [removed] = stages.splice(result.source.index, 1);
    stages.splice(result.destination.index, 0, removed);
    
    setActivePipeline({
      ...activePipeline,
      stages
    });
  };
  
  // Save the current pipeline
  const savePipeline = async () => {
    if (!activePipeline) return;
    
    setIsLoading(prev => ({ ...prev, saving: true }));
    try {
      // Use provided callback or API directly
      if (onSavePipeline) {
        await onSavePipeline(activePipeline);
      } else {
        const savedPipeline = await apiSavePipeline(activePipeline);
        
        // Update pipelines list
        setPipelines(prev => {
          const index = prev.findIndex(p => p.id === savedPipeline.id);
          if (index >= 0) {
            // Update existing pipeline
            return [...prev.slice(0, index), savedPipeline, ...prev.slice(index + 1)];
          } else {
            // Add new pipeline
            return [...prev, savedPipeline];
          }
        });
        
        // Update active pipeline with saved version
        setActivePipeline(savedPipeline);
      }
      
      setEditMode(false);
    } catch (error) {
      console.error('Error saving pipeline:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, saving: false }));
    }
  };
  
  // Run the current pipeline optimization
  const runPipeline = async () => {
    if (!activePipeline || !selectedDatasetId) return;
    
    setIsRunning(true);
    setProgress(0);
    setIsLoading(prev => ({ ...prev, running: true }));
    
    try {
      const optimizerConfig: PipelineOptimizationConfig = {
        metricToOptimize: availableMetrics[0]?.id || 'accuracy',
        optimizerAlgorithm: selectedOptimizer,
        optimizerParameters: optimizerParams,
        maxIterations: optimizerParams.generations as number,
        parallelRuns: 1
      };
      
      if (onRunPipeline) {
        const componentOptimizerConfig: OptimizerConfig = {
          algorithm: selectedOptimizer,
          parameters: optimizerParams,
          metrics: availableMetrics.map(m => m.id)
        };
        await onRunPipeline(activePipeline.id, componentOptimizerConfig);
      } else {
        const result = await optimizePipeline(activePipeline.id, selectedDatasetId, optimizerConfig) as OptimizationResult;
        
        if (result.optimizationPath) {
          setOptimizationPath(result.optimizationPath.map(path => ({
            iteration: path.iteration,
            metrics: path.metrics,
            parameters: path.parameters
          })));
        }
        
        startProgressPolling(result.executionId, activePipeline.id);
      }
    } catch (error) {
      handleApiError(error, 'running pipeline optimization');
      setIsRunning(false);
    } finally {
      setIsLoading(prev => ({ ...prev, running: false }));
    }
  };
  
  // Poll for optimization progress
  const startProgressPolling = (executionId: string, pipelineId: string) => {
    // Clear any existing interval
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
    
    // Set up new polling interval
    const interval = setInterval(async () => {
      try {
        // This would be a real API call to get the current execution status
        // For now we'll just simulate progress
        setProgress(prev => {
          const newProgress = Math.min(100, prev + Math.random() * 10);
          if (newProgress >= 100) {
            setIsRunning(false);
            clearInterval(interval);
            setRefreshInterval(null);
            
            // Add mock result
            const mockResult: OptimizationResult = {
              optimizedPipeline: activePipeline!,
              improvementMetrics: {},
              optimizationPath: [],
              executionId: `opt_${Math.random().toString(36).substring(2, 9)}`
            };
            
            setOptimizationResults(prev => [...prev, mockResult]);
          }
          return newProgress;
        });
      } catch (error) {
        console.error('Error polling optimization progress:', error);
      }
    }, 1000);
    
    setRefreshInterval(interval);
  };
  
  // Delete the current pipeline
  const deletePipeline = async () => {
    if (!activePipeline) return;
    
    if (confirm(`Are you sure you want to delete pipeline "${activePipeline.name}"?`)) {
      setIsLoading(prev => ({ ...prev, deleting: true }));
      try {
        // Use provided callback or API directly
        if (onDeletePipeline) {
          await onDeletePipeline(activePipeline.id);
        } else {
          await apiDeletePipeline(activePipeline.id);
        }
        
        // Update pipelines list
        setPipelines(prev => prev.filter(p => p.id !== activePipeline.id));
        
        if (pipelines.length > 1) {
          // Select another pipeline
          const nextPipeline = pipelines.find(p => p.id !== activePipeline.id);
          setActivePipeline(nextPipeline || null);
        } else {
          // No pipelines left
          setActivePipeline(null);
          createNewPipeline();
        }
      } catch (error) {
        console.error('Error deleting pipeline:', error);
      } finally {
        setIsLoading(prev => ({ ...prev, deleting: false }));
      }
    }
  };

  // Get results for the active pipeline
  const getActiveResults = () => {
    if (!activePipeline) return null;
    return optimizationResults.find(r => r.optimizedPipeline.id === activePipeline.id);
  };
  
  const activeResults = getActiveResults();
  
  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [refreshInterval]);
  
  // Add new function to handle API errors
  const handleApiError = (error: any, context: string) => {
    console.error(`Error ${context}:`, error);
    setApiError({
      message: error.message || `An error occurred while ${context}`,
      code: error.code,
      details: error.details
    });
    
    // Clear error after 5 seconds
    setTimeout(() => setApiError(null), 5000);
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>Pipeline Optimizer</CardTitle>
            <CardDescription>
              Configure and optimize multi-stage algorithm pipelines
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            {isLoading.pipelines ? (
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Loading pipelines...</span>
              </div>
            ) : (
              <>
                <Select 
                  value={activePipeline?.id || ''}
                  onValueChange={(value) => {
                    const selected = pipelines.find(p => p.id === value);
                    if (selected) {
                      setActivePipeline(selected);
                      setEditMode(false);
                    }
                  }}
                  disabled={editMode || pipelines.length === 0}
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select pipeline" />
                  </SelectTrigger>
                  <SelectContent>
                    {pipelines.map(pipeline => (
                      <SelectItem key={pipeline.id} value={pipeline.id}>
                        {pipeline.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={createNewPipeline}
                >
                  New
                </Button>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!activePipeline ? (
          <div className="text-center py-10 text-muted-foreground">
            <p>No pipelines yet. Create a new pipeline to get started.</p>
            <Button className="mt-4" onClick={createNewPipeline}>Create Pipeline</Button>
          </div>
        ) : (
          <Tabs defaultValue="design" className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="design">Pipeline Design</TabsTrigger>
              <TabsTrigger value="optimization">Optimization</TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>
            
            <TabsContent value="design" className="space-y-6">
              <div className="space-y-4">
                {editMode ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="pipeline-name">Pipeline Name</Label>
                        <Input
                          id="pipeline-name"
                          value={activePipeline.name}
                          onChange={(e) => setActivePipeline({
                            ...activePipeline, 
                            name: e.target.value
                          })}
                          placeholder="Enter pipeline name"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="pipeline-description">Description</Label>
                        <Input
                          id="pipeline-description"
                          value={activePipeline.description || ''}
                          onChange={(e) => setActivePipeline({
                            ...activePipeline, 
                            description: e.target.value
                          })}
                          placeholder="Enter pipeline description"
                        />
                      </div>
                    </div>
                    
                    <Separator className="my-4" />
                    
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-medium">Pipeline Stages</h3>
                        <Button 
                          size="sm" 
                          onClick={addStage}
                          disabled={activePipeline.stages.length >= 5}
                        >
                          <Plus className="h-4 w-4 mr-1" /> Add Stage
                        </Button>
                      </div>
                      
                      {activePipeline.stages.length === 0 ? (
                        <div className="text-center py-8 border rounded-md bg-muted/20">
                          <p className="text-muted-foreground">No stages in this pipeline. Add a stage to get started.</p>
                          <Button 
                            variant="outline" 
                            className="mt-2"
                            onClick={addStage}
                          >
                            <Plus className="h-4 w-4 mr-1" /> Add First Stage
                          </Button>
                        </div>
                      ) : (
                        <DragDropContext onDragEnd={handleDragEnd}>
                          <Droppable droppableId="pipeline-stages">
                            {(provided) => (
                              <div
                                {...provided.droppableProps}
                                ref={provided.innerRef}
                                className="space-y-3"
                              >
                                {activePipeline.stages.map((stage, index) => (
                                  <Draggable key={stage.id} draggableId={stage.id} index={index}>
                                    {(provided) => (
                                      <div
                                        ref={provided.innerRef}
                                        {...provided.draggableProps}
                                        className="border rounded-md p-4 bg-card"
                                      >
                                        <div className="flex justify-between items-start">
                                          <div className="flex items-center space-x-2">
                                            <div 
                                              {...provided.dragHandleProps}
                                              className="cursor-move p-1 rounded-md hover:bg-muted"
                                            >
                                              <MoveHorizontal className="h-5 w-5 text-muted-foreground" />
                                            </div>
                                            <Badge variant="outline" className="mr-2">
                                              Stage {index + 1}
                                            </Badge>
                                          </div>
                                          <div className="flex items-center space-x-1">
                                            <Button 
                                              size="sm" 
                                              variant="ghost"
                                              onClick={() => removeStage(stage.id)}
                                            >
                                              <Trash2 className="h-4 w-4 text-muted-foreground" />
                                            </Button>
                                          </div>
                                        </div>
                                        
                                        <div className="grid grid-cols-1 gap-3 mt-3">
                                          <div className="space-y-2">
                                            <Label htmlFor={`stage-name-${stage.id}`}>Stage Name</Label>
                                            <Input
                                              id={`stage-name-${stage.id}`}
                                              value={stage.name}
                                              onChange={(e) => updateStage(stage.id, { name: e.target.value })}
                                              placeholder="Enter stage name"
                                            />
                                          </div>
                                          
                                          <div className="space-y-2">
                                            <Label htmlFor={`stage-algorithm-${stage.id}`}>Algorithm</Label>
                                            <Select
                                              value={stage.algorithmId}
                                              onValueChange={(value) => {
                                                const selectedAlgorithm = availableAlgorithms.find(a => a.id === value);
                                                updateStage(stage.id, { 
                                                  algorithmId: value,
                                                  algorithmName: selectedAlgorithm?.name || value
                                                });
                                              }}
                                            >
                                              <SelectTrigger id={`stage-algorithm-${stage.id}`}>
                                                <SelectValue placeholder="Select algorithm" />
                                              </SelectTrigger>
                                              <SelectContent>
                                                {availableAlgorithms.map(algorithm => (
                                                  <SelectItem key={algorithm.id} value={algorithm.id}>
                                                    {algorithm.name}
                                                  </SelectItem>
                                                ))}
                                              </SelectContent>
                                            </Select>
                                          </div>
                                          
                                          <div className="space-y-2">
                                            <Label htmlFor={`stage-description-${stage.id}`}>Description</Label>
                                            <Input
                                              id={`stage-description-${stage.id}`}
                                              value={stage.description || ''}
                                              onChange={(e) => updateStage(stage.id, { description: e.target.value })}
                                              placeholder="Enter stage description"
                                            />
                                          </div>
                                        </div>
                                      </div>
                                    )}
                                  </Draggable>
                                ))}
                                {provided.placeholder}
                              </div>
                            )}
                          </Droppable>
                        </DragDropContext>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex justify-between items-start">
                      <div>
                        <h2 className="text-2xl font-bold">{activePipeline.name}</h2>
                        <p className="text-muted-foreground">{activePipeline.description}</p>
                        
                        {activePipeline.status && (
                          <Badge 
                            className="mt-2"
                            variant={activePipeline.status === 'completed' ? 'default' : 
                                     activePipeline.status === 'failed' ? 'destructive' : 'outline'}
                          >
                            {activePipeline.status.charAt(0).toUpperCase() + activePipeline.status.slice(1)}
                          </Badge>
                        )}
                      </div>
                      <Button 
                        variant="outline" 
                        onClick={() => setEditMode(true)}
                      >
                        <Edit className="h-4 w-4 mr-1" /> Edit
                      </Button>
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium">Pipeline Stages</h3>
                      
                      {activePipeline.stages.length === 0 ? (
                        <div className="text-center py-6 bg-muted/20 rounded-md">
                          <p className="text-muted-foreground">No stages defined in this pipeline.</p>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {activePipeline.stages.map((stage, index) => (
                            <div key={stage.id} className="flex items-start space-x-2">
                              <div className="bg-muted w-8 h-8 rounded-full flex items-center justify-center shrink-0">
                                <span className="text-sm font-medium">{index + 1}</span>
                              </div>
                              <div className="flex-1 border rounded-md p-4">
                                <div className="flex justify-between items-start">
                                  <div>
                                    <h4 className="font-medium">{stage.name}</h4>
                                    <p className="text-sm text-muted-foreground">{stage.description}</p>
                                  </div>
                                  <Badge variant="outline">{stage.algorithmName}</Badge>
                                </div>
                              </div>
                              {index < activePipeline.stages.length - 1 && (
                                <div className="flex items-center justify-center h-full mt-4 px-2">
                                  <ArrowRight className="h-5 w-5 text-muted-foreground" />
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
            
            <TabsContent value="optimization" className="space-y-6">
              <div className="space-y-4">
                {apiError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{apiError.message}</AlertDescription>
                  </Alert>
                )}
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Dataset Selection</h3>
                  <div className="space-y-2">
                    <Label htmlFor="dataset">Select Dataset</Label>
                    <Select
                      value={selectedDatasetId}
                      onValueChange={setSelectedDatasetId}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets.map(dataset => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            {dataset.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {selectedDatasetId && datasets.find(d => d.id === selectedDatasetId)?.characteristics && (
                      <div className="mt-2 p-3 bg-muted/20 rounded-md">
                        <h4 className="font-medium mb-2">Problem Characteristics</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          {Object.entries(datasets.find(d => d.id === selectedDatasetId)!.characteristics!).map(([key, value]) => (
                            <div key={key} className="flex items-center space-x-2">
                              <span className="text-muted-foreground">{key}:</span>
                              <span>{value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium mb-2">Pipeline Optimization Settings</h3>
                  <div className="bg-muted/40 p-4 rounded-md border mb-4">
                    <div className="flex items-start space-x-2">
                      <AlertCircle className="h-5 w-5 text-muted-foreground mt-0.5" />
                      <div>
                        <p className="text-sm">
                          Pipeline optimization will search for the best configuration of algorithms and parameters
                          across all stages to maximize performance on your selected metrics.
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="optimizer-algorithm">Optimizer Algorithm</Label>
                      <Select
                        value={selectedOptimizer}
                        onValueChange={setSelectedOptimizer}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select optimizer algorithm" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ga">Genetic Algorithm</SelectItem>
                          <SelectItem value="pso">Particle Swarm Optimization</SelectItem>
                          <SelectItem value="bo">Bayesian Optimization</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="auto-config">Auto-configure Parameters</Label>
                        <Switch 
                          id="auto-config" 
                          checked={autoConfigOptimizer}
                          onCheckedChange={setAutoConfigOptimizer}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Optimization Parameters</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(optimizerParams).map(([key, value]) => (
                      <div key={key} className="space-y-2">
                        <Label htmlFor={`param-${key}`}>
                          {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        </Label>
                        <Input
                          id={`param-${key}`}
                          type="number"
                          value={value}
                          disabled={autoConfigOptimizer}
                          onChange={(e) => {
                            const newValue = parseFloat(e.target.value);
                            setOptimizerParams(prev => ({
                              ...prev,
                              [key]: isNaN(newValue) ? 0 : newValue
                            }));
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Evaluation Metrics</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {availableMetrics.map(metric => (
                      <div 
                        key={metric.id}
                        className="flex items-center space-x-2 p-2 border rounded-md"
                      >
                        <div className="h-4 w-4 border rounded-sm flex items-center justify-center bg-primary">
                          <Check className="h-3 w-3 text-primary-foreground" />
                        </div>
                        <div>
                          <p className="font-medium">{metric.name}</p>
                          {metric.description && (
                            <p className="text-sm text-muted-foreground">{metric.description}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="results" className="space-y-6">
              {isRunning && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Optimizing pipeline...</span>
                    <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
              
              {activeResults ? (
                <div className="space-y-6">
                  <div className="bg-muted/20 p-4 rounded-md border">
                    <h3 className="text-lg font-medium mb-2">Performance Improvements</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(activeResults.improvementMetrics).map(([metricId, values]) => {
                        const metric = availableMetrics.find(m => m.id === metricId);
                        return (
                          <div key={metricId} className="bg-background p-3 rounded-md border">
                            <p className="text-sm text-muted-foreground">{metric?.name || metricId}</p>
                            <p className="text-2xl font-bold">{values.after.toFixed(4)}</p>
                            <p className="text-sm text-green-600">
                              {values.improvement}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  
                  <h3 className="text-lg font-medium">Stage Performance</h3>
                  <div className="space-y-4">
                    {activeResults.optimizedPipeline.stages.map((stage) => (
                      <div key={stage.id} className="border rounded-md p-4">
                        <h4 className="font-medium">{stage.name}</h4>
                        <p className="text-sm text-muted-foreground mb-3">{stage.algorithmName}</p>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          {Object.entries(stage.parameters).map(([paramName, value]) => (
                            <div key={paramName} className="bg-muted/20 p-2 rounded-md">
                              <p className="text-xs text-muted-foreground">{paramName}</p>
                              <p className="text-lg font-medium">
                                {typeof value === 'number' ? value.toFixed(4) : String(value)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-10 text-muted-foreground">
                  <p>No optimization results available for this pipeline.</p>
                  <p className="mt-1">Configure and run the pipeline optimization to see results.</p>
                </div>
              )}
              
              {optimizationPath.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Optimization Progress</h3>
                  <div className="h-[300px] border rounded-md p-4">
                    <div className="flex items-center space-x-2 mb-4">
                      <LineChart className="h-5 w-5" />
                      <span className="font-medium">Metric Progress Over Iterations</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Showing progress over {optimizationPath.length} iterations
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        {editMode ? (
          <div className="flex justify-between w-full">
            <Button 
              variant="outline" 
              onClick={() => {
                if (pipelines.find(p => p.id === activePipeline?.id)) {
                  // Pipeline exists, revert to view mode
                  setEditMode(false);
                } else {
                  // New pipeline, go back to selection
                  setActivePipeline(pipelines.length > 0 ? pipelines[0] : null);
                  setEditMode(false);
                }
              }}
            >
              Cancel
            </Button>
            <Button 
              onClick={savePipeline}
              disabled={!activePipeline || activePipeline.stages.length === 0 || isLoading.saving}
            >
              {isLoading.saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Pipeline'
              )}
            </Button>
          </div>
        ) : (
          <div className="flex justify-between w-full">
            <Button 
              variant="outline" 
              onClick={deletePipeline}
              disabled={!activePipeline || isLoading.deleting}
            >
              {isLoading.deleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete Pipeline'
              )}
            </Button>
            <Button 
              onClick={runPipeline}
              disabled={!activePipeline || isRunning || activePipeline.stages.length === 0 || isLoading.running}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Optimizing...
                </>
              ) : isLoading.running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Preparing...
                </>
              ) : (
                'Run Optimization'
              )}
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
} 