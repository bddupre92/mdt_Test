"use client"

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Info, AlertCircle, CheckCircle2, Loader2, Settings, AlertTriangle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  fetchOptimizers,
  getOptimizerDetails,
  getRecommendedParameters,
  Optimizer,
  OptimizerParameters
} from '@/lib/api/optimizers';
import { AlgorithmConfigurationData } from '@/lib/utils/workflow-validation';
import { 
  ALGORITHMS, 
  Algorithm, 
  AlgorithmCategory, 
  getAlgorithmSuitability, 
  ALGORITHM_PRESETS, 
  ParameterPreset 
} from '@/lib/data/algorithm-metadata';

/**
 * Maps algorithm category names to more user-friendly display names
 */
const CATEGORY_LABELS: Record<string, string> = {
  'evolutionary': 'Evolutionary',
  'swarm': 'Swarm Intelligence',
  'classical': 'Classical',
  'bayesian': 'Bayesian',
  'hybrid': 'Hybrid Methods',
  'meta': 'Meta-Optimizers',
  'other': 'Other Algorithms'
};

/**
 * Parameter metadata for validation and UI display
 */
interface ParameterMetadata {
  name: string;
  type: 'number' | 'integer' | 'boolean' | 'string' | 'select';
  description: string;
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  category?: 'basic' | 'advanced' | 'expert';
  units?: string;
}

interface AlgorithmSelectorProps {
  /** Current selected algorithm configuration data */
  value?: AlgorithmConfigurationData;
  /** Callback when algorithm configuration changes */
  onChange?: (data: AlgorithmConfigurationData) => void;
  /** Dataset ID to potentially influence algorithm recommendations */
  datasetId?: string;
  /** Problem characteristics for algorithm recommendations */
  problemCharacteristics?: Record<string, any>;
  /** Whether to allow auto-configuration of parameters */
  enableAutoConfig?: boolean;
  /** Whether to show presets */
  enablePresets?: boolean;
  /** Whether to include the meta-optimizer as an option */
  includeMetaOptimizer?: boolean;
  /** CSS class name for the component */
  className?: string;
}

export function AlgorithmSelector({
  value,
  onChange,
  datasetId,
  problemCharacteristics,
  enableAutoConfig = true,
  enablePresets = true,
  includeMetaOptimizer = true,
  className,
}: AlgorithmSelectorProps) {
  // State for algorithms and categories
  const [optimizers, setOptimizers] = useState<Algorithm[]>([]);
  const [categories, setCategories] = useState<Record<string, Algorithm[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // State for selected algorithm and parameters
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState<string>(value?.algorithmId || '');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [algorithmParameters, setAlgorithmParameters] = useState<Record<string, any>>(value?.parameters || {});
  
  // State for parameter metadata
  const [parameterMetadata, setParameterMetadata] = useState<Record<string, ParameterMetadata>>({});
  
  // State for auto-configuration
  const [useAutoConfig, setUseAutoConfig] = useState(true);
  
  // State for parameter view
  const [parameterView, setParameterView] = useState<'basic' | 'advanced' | 'all'>('basic');
  
  // Initialize algorithms from metadata
  useEffect(() => {
    const initializeAlgorithms = () => {
      setLoading(true);
      try {
        // Flatten the algorithms from all categories
        const allAlgorithms = Object.values(ALGORITHMS).flat();
        
        // Filter out meta-optimizer if not included
        const filteredAlgorithms = includeMetaOptimizer 
          ? allAlgorithms 
          : allAlgorithms.filter(alg => alg.category !== 'meta');
        
        setOptimizers(filteredAlgorithms);
        
        // Group by category
        const categoryGroups: Record<string, Algorithm[]> = {};
        
        Object.entries(ALGORITHMS).forEach(([category, algorithms]) => {
          if (category !== 'meta' || includeMetaOptimizer) {
            categoryGroups[category] = algorithms;
          }
        });
        
        setCategories(categoryGroups);
        
        // If we have algorithms and none is selected, select the first one from the first category
        if (filteredAlgorithms.length > 0 && !selectedAlgorithmId) {
          const firstCategory = Object.keys(categoryGroups)[0];
          const firstAlgorithm = categoryGroups[firstCategory][0];
          setSelectedCategory(firstCategory);
          setSelectedAlgorithmId(firstAlgorithm.id);
          getAlgorithmDetails(firstAlgorithm.id);
        } else if (selectedAlgorithmId) {
          // If an algorithm is already selected, get its details
          getAlgorithmDetails(selectedAlgorithmId);
          
          // Find its category
          for (const [category, algorithms] of Object.entries(categoryGroups)) {
            if (algorithms.some(alg => alg.id === selectedAlgorithmId)) {
              setSelectedCategory(category);
              break;
            }
          }
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error initializing algorithms:', err);
        setError('Failed to initialize optimization algorithms. Please try again.');
        setLoading(false);
      }
    };
    
    initializeAlgorithms();
  }, [includeMetaOptimizer, selectedAlgorithmId]);
  
  // Get algorithm details when selection changes
  const getAlgorithmDetails = useCallback((algorithmId: string) => {
    if (!algorithmId) return;
    
    try {
      const algorithm = optimizers.find(alg => alg.id === algorithmId);
      
      if (algorithm) {
        // If auto-config is enabled, get recommended parameters
        if (useAutoConfig && problemCharacteristics) {
          try {
            // In a real implementation, you would call an API here
            // For now, we'll just use the default parameters
            setAlgorithmParameters(algorithm.defaultParameters || {});
          } catch (error) {
            console.error('Error getting recommended parameters:', error);
            // Fall back to default parameters
            setAlgorithmParameters(algorithm.defaultParameters || {});
          }
        } else {
          // Use default parameters
          setAlgorithmParameters(algorithm.defaultParameters || {});
        }
        
        // Create metadata for parameters
        createParameterMetadata(algorithm);
      }
    } catch (error) {
      console.error('Error fetching algorithm details:', error);
    }
  }, [useAutoConfig, problemCharacteristics, optimizers]);
  
  // Create parameter metadata for UI rendering
  const createParameterMetadata = (algorithm: Algorithm) => {
    const metadata: Record<string, ParameterMetadata> = {};
    
    if (algorithm.defaultParameters) {
      Object.entries(algorithm.defaultParameters).forEach(([key, value]) => {
        const type = typeof value;
        
        metadata[key] = {
          name: key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()),
          type: type === 'number' ? 'number' :
                type === 'boolean' ? 'boolean' : 
                type === 'string' ? 'string' : 'number',
          description: `Parameter ${key} for ${algorithm.name}`,
          default: value,
          category: key.includes('advanced') ? 'advanced' : 'basic'
        };
        
        // Add specific constraints based on parameter name patterns
        if (key.includes('size') || key.includes('population')) {
          metadata[key].min = 1;
          metadata[key].max = 1000;
          metadata[key].step = 1;
          metadata[key].type = 'integer';
        } else if (key.includes('rate') || key.includes('probability')) {
          metadata[key].min = 0;
          metadata[key].max = 1;
          metadata[key].step = 0.01;
        } else if (key.includes('iterations') || key.includes('generations')) {
          metadata[key].min = 1;
          metadata[key].max = 10000;
          metadata[key].step = 1;
          metadata[key].type = 'integer';
        }
      });
    }
    
    setParameterMetadata(metadata);
  };
  
  // Update parameter value
  const updateParameter = (key: string, value: any) => {
    setAlgorithmParameters(prev => {
      const updated = { ...prev, [key]: value };
      
      // Call onChange to update parent component
      if (onChange && selectedAlgorithmId) {
        const selectedAlgorithm = optimizers.find(alg => alg.id === selectedAlgorithmId);
        onChange({
          algorithmId: selectedAlgorithmId,
          algorithmType: selectedAlgorithm?.name || '',
          parameters: updated
        });
      }
      
      return updated;
    });
  };
  
  // When algorithm selection changes
  useEffect(() => {
    if (selectedAlgorithmId) {
      getAlgorithmDetails(selectedAlgorithmId);
    }
  }, [selectedAlgorithmId, getAlgorithmDetails]);
  
  // When auto-config changes
  useEffect(() => {
    if (selectedAlgorithmId) {
      getAlgorithmDetails(selectedAlgorithmId);
    }
  }, [useAutoConfig, selectedAlgorithmId, getAlgorithmDetails]);
  
  // Handle algorithm selection
  const handleAlgorithmSelect = (algorithmId: string) => {
    setSelectedAlgorithmId(algorithmId);
    
    // Find the category of this algorithm
    for (const [category, algorithmList] of Object.entries(categories)) {
      if (algorithmList.some(alg => alg.id === algorithmId)) {
        setSelectedCategory(category);
        break;
      }
    }
  };
  
  // Apply parameter preset
  const applyPreset = (preset: ParameterPreset) => {
    setAlgorithmParameters(preset.parameters);
    
    // Call onChange to update parent component
    if (onChange && selectedAlgorithmId) {
      const selectedAlgorithm = optimizers.find(alg => alg.id === selectedAlgorithmId);
      onChange({
        algorithmId: selectedAlgorithmId,
        algorithmType: selectedAlgorithm?.name || '',
        parameters: preset.parameters
      });
    }
  };
  
  // Filter parameters based on current view
  const getVisibleParameters = () => {
    if (parameterView === 'all') {
      return Object.keys(parameterMetadata);
    }
    
    return Object.entries(parameterMetadata)
      .filter(([_, meta]) => meta.category === parameterView || meta.category === 'basic')
      .map(([key]) => key);
  };
  
  // Render parameter input based on type
  const renderParameterInput = (key: string, metadata: ParameterMetadata) => {
    const value = algorithmParameters[key] !== undefined 
      ? algorithmParameters[key] 
      : metadata.default;
    
    switch (metadata.type) {
      case 'boolean':
        return (
          <Switch 
            checked={!!value} 
            onCheckedChange={checked => updateParameter(key, checked)} 
          />
        );
      case 'integer':
      case 'number':
        if (metadata.min !== undefined && metadata.max !== undefined && metadata.step !== undefined) {
          return (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{metadata.min}</span>
                <span>{metadata.max}</span>
              </div>
              <div className="flex items-center gap-2">
                <Slider 
                  value={[typeof value === 'number' ? value : metadata.default]} 
                  min={metadata.min} 
                  max={metadata.max} 
                  step={metadata.step}
                  onValueChange={([val]) => updateParameter(key, val)}
                  className="flex-1"
                />
                <Input 
                  type="number"
                  value={value}
                  onChange={e => {
                    const val = metadata.type === 'integer' 
                      ? parseInt(e.target.value) 
                      : parseFloat(e.target.value);
                    if (!isNaN(val)) {
                      updateParameter(key, val);
                    }
                  }}
                  min={metadata.min}
                  max={metadata.max}
                  step={metadata.step}
                  className="w-20"
                />
              </div>
            </div>
          );
        }
        return (
          <Input 
            type="number"
            value={value}
            onChange={e => {
              const val = metadata.type === 'integer' 
                ? parseInt(e.target.value) 
                : parseFloat(e.target.value);
              if (!isNaN(val)) {
                updateParameter(key, val);
              }
            }}
            className="w-full"
          />
        );
      case 'select':
        return (
          <Select 
            value={String(value)} 
            onValueChange={val => updateParameter(key, val)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select option" />
            </SelectTrigger>
            <SelectContent>
              {metadata.options?.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      default:
        return (
          <Input 
            value={value?.toString()} 
            onChange={e => updateParameter(key, e.target.value)}
            className="w-full" 
          />
        );
    }
  };
  
  const visibleParameters = getVisibleParameters();
  const selectedAlgorithm = optimizers.find(alg => alg.id === selectedAlgorithmId);
  
  // Loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center p-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2">Loading optimization algorithms...</span>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <div className="p-4 border border-red-200 bg-red-50 rounded-md">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <AlertCircle className="h-5 w-5" />
          <span className="font-medium">Error</span>
        </div>
        <p>{error}</p>
        <Button 
          variant="outline" 
          className="mt-4"
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </div>
    );
  }
  
  // Empty state
  if (optimizers.length === 0) {
    return (
      <div className="p-4 border border-yellow-200 bg-yellow-50 rounded-md">
        <div className="flex items-center gap-2 text-yellow-600 mb-2">
          <AlertTriangle className="h-5 w-5" />
          <span className="font-medium">No Algorithms Available</span>
        </div>
        <p>No optimization algorithms are currently available. Please try again later.</p>
      </div>
    );
  }
  
  return (
    <div className={className}>
      <div className="space-y-6">
        {/* Algorithm Selection */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Select Optimization Algorithm</CardTitle>
            <CardDescription>
              Choose an algorithm based on your optimization problem characteristics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs 
              defaultValue={selectedCategory || Object.keys(categories)[0]}
              value={selectedCategory}
              onValueChange={setSelectedCategory}
            >
              <TabsList className="mb-4 flex flex-wrap">
                {Object.entries(categories).map(([category, algorithms]) => (
                  <TabsTrigger 
                    key={category} 
                    value={category}
                    className="flex items-center gap-1"
                  >
                    {CATEGORY_LABELS[category] || category}
                    <Badge variant="secondary" className="ml-1 rounded-full px-2 py-0 text-xs">
                      {algorithms.length}
                    </Badge>
                  </TabsTrigger>
                ))}
              </TabsList>
              
              {Object.entries(categories).map(([category, algorithms]) => (
                <TabsContent key={category} value={category} className="mt-0">
                  <RadioGroup 
                    value={selectedAlgorithmId} 
                    onValueChange={handleAlgorithmSelect}
                    className="grid gap-2 md:grid-cols-2 lg:grid-cols-3"
                  >
                    {algorithms.map(algorithm => (
                      <div
                        key={algorithm.id}
                        className={`border rounded-lg p-4 transition-colors cursor-pointer ${
                          selectedAlgorithmId === algorithm.id 
                            ? "border-primary bg-primary/5" 
                            : "border-border hover:border-primary/50"
                        }`}
                        onClick={() => handleAlgorithmSelect(algorithm.id)}
                      >
                        <RadioGroupItem 
                          value={algorithm.id} 
                          id={algorithm.id} 
                          className="sr-only" 
                        />
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h3 className="font-medium">{algorithm.name}</h3>
                            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                              {algorithm.description || 'No description available'}
                            </p>
                          </div>
                          {algorithm.category === 'meta' && (
                            <Badge variant="secondary" className="ml-2">Meta</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </RadioGroup>
                </TabsContent>
              ))}
            </Tabs>
          </CardContent>
        </Card>
        
        {/* Algorithm Configuration */}
        {selectedAlgorithmId && selectedAlgorithm && (
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Configure {selectedAlgorithm.name}</CardTitle>
                  <CardDescription>
                    Set parameters for the selected optimization algorithm
                  </CardDescription>
                </div>
                
                {enableAutoConfig && (
                  <div className="flex items-center gap-2">
                    <Label htmlFor="auto-config" className="text-sm cursor-pointer">
                      Auto-configure
                    </Label>
                    <Switch 
                      id="auto-config" 
                      checked={useAutoConfig} 
                      onCheckedChange={setUseAutoConfig} 
                    />
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {enablePresets && ALGORITHM_PRESETS[selectedAlgorithmId] && (
                <div className="mb-6">
                  <h3 className="text-sm font-medium mb-2">Parameter Presets</h3>
                  <div className="flex flex-wrap gap-2">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        const algorithm = optimizers.find(alg => alg.id === selectedAlgorithmId);
                        if (algorithm) {
                          setAlgorithmParameters(algorithm.defaultParameters);
                        }
                      }}
                    >
                      Default
                    </Button>
                    {ALGORITHM_PRESETS[selectedAlgorithmId].map(preset => (
                      <TooltipProvider key={preset.id}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => applyPreset(preset)}
                            >
                              {preset.name}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="bottom">
                            <p className="max-w-xs">{preset.description}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-sm font-medium">Parameters</h3>
                  
                  <Select 
                    value={parameterView} 
                    onValueChange={(value: 'basic' | 'advanced' | 'all') => setParameterView(value)}
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue placeholder="View" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="basic">Basic</SelectItem>
                      <SelectItem value="advanced">Advanced</SelectItem>
                      <SelectItem value="all">All Parameters</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  {visibleParameters.length > 0 ? (
                    visibleParameters.map(key => (
                      <div key={key} className="space-y-1">
                        <div className="flex justify-between">
                          <Label htmlFor={key} className="text-sm">
                            {parameterMetadata[key].name}
                            {parameterMetadata[key].units && (
                              <span className="text-muted-foreground ml-1">
                                ({parameterMetadata[key].units})
                              </span>
                            )}
                          </Label>
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Info className="h-4 w-4 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent side="top">
                                <p className="max-w-xs">{parameterMetadata[key].description}</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
                        {renderParameterInput(key, parameterMetadata[key])}
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-4 text-muted-foreground">
                      <Settings className="h-12 w-12 mx-auto mb-2 opacity-20" />
                      <p>No configurable parameters available for this algorithm.</p>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
            
            <CardFooter className="justify-between border-t px-6 py-4">
              <div className="text-sm text-muted-foreground">
                {useAutoConfig ? 
                  "Parameters are auto-configured based on your problem characteristics." :
                  "Manual configuration mode enabled."
                }
              </div>
              
              <Button onClick={() => {
                if (onChange && selectedAlgorithmId) {
                  onChange({
                    algorithmId: selectedAlgorithmId,
                    algorithmType: selectedAlgorithm.name,
                    parameters: algorithmParameters,
                    validationConfig: {
                      autoConfigured: useAutoConfig
                    }
                  });
                }
              }}>
                Apply Configuration
              </Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </div>
  );
} 