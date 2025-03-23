"use client"

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  AlertCircle, AlertTriangle, CheckCircle, Clock, CpuIcon, 
  BarChart3, Loader2, Play, StopCircle, Timer, RefreshCw, Settings
} from 'lucide-react';
import { Progress } from '@/components/ui/progress';

import { ExecutionLogViewer } from './ExecutionLogViewer';
import { startExecution, stopExecution, fetchExecutionStatus, fetchExecutionMetrics, setupExecutionWebSocket, ExecutionMetrics } from '@/lib/api/execution';
import { formatElapsedTime, calculateEta, getStatusColor, getStatusIcon, generateExecutionSummary } from '@/lib/utils/execution-status';
import { ExecutionData } from '@/lib/utils/workflow-validation';

interface ExecutionControlProps {
  algorithmId: string;
  datasetId: string;
  parameters: Record<string, any>;
  onExecutionComplete?: (executionId: string) => void;
  onExecutionFailed?: (error: string) => void;
  className?: string;
}

export function ExecutionControl({
  algorithmId,
  datasetId,
  parameters,
  onExecutionComplete,
  onExecutionFailed,
  className = '',
}: ExecutionControlProps) {
  // Execution state
  const [executionData, setExecutionData] = useState<ExecutionData | null>(null);
  const [executionMetrics, setExecutionMetrics] = useState<ExecutionMetrics | null>(null);
  
  // Settings state
  const [stoppingCriteria, setStoppingCriteria] = useState({
    maxIterations: 100,
    targetMetric: 'loss',
    targetValue: 0.01,
    maxTime: 3600 // 1 hour in seconds
  });
  
  const [resourceSettings, setResourceSettings] = useState({
    parallelJobs: 1,
    gpuEnabled: false,
    memoryLimit: 4096 // 4GB in MB
  });
  
  // Loading state
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  
  // Error state
  const [error, setError] = useState<string | null>(null);
  
  // Cleanup function for WebSocket
  const [cleanupWebSocket, setCleanupWebSocket] = useState<(() => void) | null>(null);
  
  // Get metrics at regular intervals
  useEffect(() => {
    let metricsInterval: NodeJS.Timeout | null = null;
    
    if (executionData && executionData.status === 'running') {
      metricsInterval = setInterval(async () => {
        try {
          const metrics = await fetchExecutionMetrics(executionData.executionId);
          setExecutionMetrics(metrics);
        } catch (err) {
          console.error('Failed to fetch metrics:', err);
        }
      }, 2000);
    }
    
    return () => {
      if (metricsInterval) clearInterval(metricsInterval);
    };
  }, [executionData]);
  
  // Start execution
  const handleStartExecution = async () => {
    setError(null);
    setIsStarting(true);
    
    try {
      const result = await startExecution({
        algorithmId,
        datasetId,
        parameters,
        stoppingCriteria: {
          maxIterations: stoppingCriteria.maxIterations,
          targetMetric: stoppingCriteria.targetMetric,
          targetValue: stoppingCriteria.targetValue,
          maxTime: stoppingCriteria.maxTime
        },
        resources: {
          parallelJobs: resourceSettings.parallelJobs,
          gpuEnabled: resourceSettings.gpuEnabled,
          memoryLimit: resourceSettings.memoryLimit
        }
      });
      
      setExecutionData(result);
      
      // Set up WebSocket for updates
      const cleanup = setupExecutionWebSocket(result.executionId, (data) => {
        setExecutionData(data);
        
        // Handle completion
        if (data.status === 'completed' && onExecutionComplete) {
          onExecutionComplete(data.executionId);
        }
        
        // Handle failure
        if (data.status === 'failed' && onExecutionFailed) {
          onExecutionFailed(`Execution failed: ${data.logs[data.logs.length - 1] || 'Unknown error'}`);
        }
      });
      
      setCleanupWebSocket(() => cleanup);
    } catch (err: any) {
      console.error('Failed to start execution:', err);
      setError(`Failed to start execution: ${err.message || 'Unknown error'}`);
      
      if (onExecutionFailed) {
        onExecutionFailed(`Failed to start execution: ${err.message || 'Unknown error'}`);
      }
    } finally {
      setIsStarting(false);
    }
  };
  
  // Stop execution
  const handleStopExecution = async () => {
    if (!executionData) return;
    
    setIsStopping(true);
    
    try {
      const result = await stopExecution(executionData.executionId);
      setExecutionData(result);
      
      if (onExecutionFailed) {
        onExecutionFailed('Execution stopped by user');
      }
    } catch (err: any) {
      console.error('Failed to stop execution:', err);
      setError(`Failed to stop execution: ${err.message || 'Unknown error'}`);
    } finally {
      setIsStopping(false);
    }
  };
  
  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (cleanupWebSocket) cleanupWebSocket();
    };
  }, [cleanupWebSocket]);
  
  // Compute estimated time remaining
  const estimatedTimeRemaining = executionData && executionMetrics
    ? calculateEta(executionData.progress, executionMetrics.elapsedTime)
    : null;
  
  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>Execution Control</CardTitle>
              <CardDescription>
                Configure and monitor algorithm execution
              </CardDescription>
            </div>
            {executionData && (
              <div className="flex items-center">
                {executionData.status === 'running' ? (
                  <div className="flex items-center text-blue-500">
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                    <span>Running</span>
                  </div>
                ) : executionData.status === 'completed' ? (
                  <div className="flex items-center text-green-500">
                    <CheckCircle className="h-4 w-4 mr-1" />
                    <span>Completed</span>
                  </div>
                ) : executionData.status === 'failed' ? (
                  <div className="flex items-center text-red-500">
                    <AlertCircle className="h-4 w-4 mr-1" />
                    <span>Failed</span>
                  </div>
                ) : (
                  <div className="flex items-center text-gray-500">
                    <Clock className="h-4 w-4 mr-1" />
                    <span>Pending</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Show error if any */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {/* Execution Progress */}
          {executionData ? (
            <div className="space-y-6">
              {/* Progress Section */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-medium">Progress</h3>
                  <span className="text-sm">{executionData.progress}%</span>
                </div>
                
                <Progress value={executionData.progress} className="h-2" />
                
                <div className="flex justify-between items-center text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Timer className="h-3 w-3" />
                    <span>
                      {executionMetrics 
                        ? `Elapsed: ${formatElapsedTime(executionMetrics.elapsedTime)}` 
                        : 'Elapsed: 0s'}
                    </span>
                  </div>
                  
                  {estimatedTimeRemaining !== null && executionData.status === 'running' && (
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      <span>ETA: {formatElapsedTime(estimatedTimeRemaining)}</span>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Resource Utilization */}
              {executionMetrics && (
                <div className="space-y-2 pt-2 border-t">
                  <h3 className="text-sm font-medium">Resource Utilization</h3>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">CPU</span>
                        <span>{executionMetrics.resourceUtilization.cpuPercent}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500"
                          style={{ width: `${executionMetrics.resourceUtilization.cpuPercent}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Memory</span>
                        <span>{executionMetrics.resourceUtilization.memoryPercent}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500"
                          style={{ width: `${executionMetrics.resourceUtilization.memoryPercent}%` }}
                        />
                      </div>
                    </div>
                    
                    {executionMetrics.resourceUtilization.gpuPercent !== undefined && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">GPU</span>
                          <span>{executionMetrics.resourceUtilization.gpuPercent}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500"
                            style={{ width: `${executionMetrics.resourceUtilization.gpuPercent}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex justify-between text-xs text-muted-foreground pt-1">
                    <span>Iterations: {executionMetrics.iterationsCompleted}</span>
                    {executionMetrics.convergenceMetrics && Object.keys(executionMetrics.convergenceMetrics).length > 0 && (
                      <span>
                        Last {Object.entries(executionMetrics.convergenceMetrics)[0][0]}: 
                        {' '}{Object.entries(executionMetrics.convergenceMetrics)[0][1].slice(-1)[0].toFixed(4)}
                      </span>
                    )}
                  </div>
                </div>
              )}
              
              {/* Logs */}
              <div className="pt-2 border-t">
                <h3 className="text-sm font-medium mb-2">Execution Logs</h3>
                <ExecutionLogViewer 
                  executionId={executionData.executionId} 
                  maxHeight="200px"
                />
              </div>
              
              {/* Control Buttons */}
              <div className="flex justify-end space-x-2 pt-2">
                {executionData.status === 'running' && (
                  <Button 
                    variant="destructive" 
                    size="sm"
                    onClick={handleStopExecution}
                    disabled={isStopping}
                  >
                    {isStopping ? (
                      <>
                        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                        Stopping
                      </>
                    ) : (
                      <>
                        <StopCircle className="mr-1 h-3 w-3" />
                        Stop Execution
                      </>
                    )}
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Configuration Tabs */}
              <Tabs defaultValue="stopping">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="stopping">Stopping Criteria</TabsTrigger>
                  <TabsTrigger value="resources">Resources</TabsTrigger>
                </TabsList>
                
                {/* Stopping Criteria Tab */}
                <TabsContent value="stopping" className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="max-iterations">Maximum Iterations</Label>
                      <span className="text-sm">{stoppingCriteria.maxIterations}</span>
                    </div>
                    <Slider
                      id="max-iterations"
                      value={[stoppingCriteria.maxIterations]}
                      min={10}
                      max={1000}
                      step={10}
                      onValueChange={(value) => 
                        setStoppingCriteria({...stoppingCriteria, maxIterations: value[0]})
                      }
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="target-metric">Target Metric</Label>
                    <Input
                      id="target-metric"
                      value={stoppingCriteria.targetMetric}
                      onChange={(e) => 
                        setStoppingCriteria({...stoppingCriteria, targetMetric: e.target.value})
                      }
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="target-value">Target Value</Label>
                      <span className="text-sm">{stoppingCriteria.targetValue}</span>
                    </div>
                    <Slider
                      id="target-value"
                      value={[stoppingCriteria.targetValue]}
                      min={0.001}
                      max={0.1}
                      step={0.001}
                      onValueChange={(value) => 
                        setStoppingCriteria({...stoppingCriteria, targetValue: value[0]})
                      }
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="max-time">Maximum Time (hours)</Label>
                      <span className="text-sm">{(stoppingCriteria.maxTime / 3600).toFixed(1)}</span>
                    </div>
                    <Slider
                      id="max-time"
                      value={[stoppingCriteria.maxTime / 3600]}
                      min={0.1}
                      max={24}
                      step={0.1}
                      onValueChange={(value) => 
                        setStoppingCriteria({...stoppingCriteria, maxTime: Math.round(value[0] * 3600)})
                      }
                    />
                  </div>
                </TabsContent>
                
                {/* Resources Tab */}
                <TabsContent value="resources" className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="parallel-jobs">Parallel Jobs</Label>
                      <span className="text-sm">{resourceSettings.parallelJobs}</span>
                    </div>
                    <Slider
                      id="parallel-jobs"
                      value={[resourceSettings.parallelJobs]}
                      min={1}
                      max={16}
                      step={1}
                      onValueChange={(value) => 
                        setResourceSettings({...resourceSettings, parallelJobs: value[0]})
                      }
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="memory-limit">Memory Limit (GB)</Label>
                      <span className="text-sm">{(resourceSettings.memoryLimit / 1024).toFixed(1)}</span>
                    </div>
                    <Slider
                      id="memory-limit"
                      value={[resourceSettings.memoryLimit / 1024]}
                      min={0.5}
                      max={32}
                      step={0.5}
                      onValueChange={(value) => 
                        setResourceSettings({...resourceSettings, memoryLimit: Math.round(value[0] * 1024)})
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label htmlFor="gpu-enabled" className="cursor-pointer">Enable GPU Acceleration</Label>
                    <Switch
                      id="gpu-enabled"
                      checked={resourceSettings.gpuEnabled}
                      onCheckedChange={(checked) => 
                        setResourceSettings({...resourceSettings, gpuEnabled: checked})
                      }
                    />
                  </div>
                </TabsContent>
              </Tabs>
              
              {/* Start Button */}
              <div className="flex justify-end pt-4">
                <Button
                  onClick={handleStartExecution}
                  disabled={isStarting || !algorithmId || !datasetId}
                >
                  {isStarting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Execution
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 