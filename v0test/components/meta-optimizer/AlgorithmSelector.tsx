import React, { useState, useEffect, useCallback } from 'react';

// Define types for algorithm objects
interface Algorithm {
  id: string;
  name: string;
  description: string;
}

// Define type for algorithm categories
type AlgorithmCategories = {
  [key: string]: Algorithm[];
};

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Info, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { 
  getAlgorithmRecommendations, 
  getHyperparameterRecommendations,
  AlgorithmRecommendation
} from '@/lib/api/meta-optimizer/algorithm-selection';
import { ProblemCharacteristics } from '@/lib/utils/meta-optimizer/problem-classification';
// Import the new OptimizerClient functions
import {
  fetchOptimizers,
  getOptimizerDetails,
  getRecommendedParameters,
  Optimizer,
  OptimizerParameters
} from '@/lib/api/optimizers';

// Algorithms grouped by category
const ALGORITHMS: AlgorithmCategories = {
  evolutionary: [
    { id: 'ga', name: 'Genetic Algorithm', description: 'Good for discrete optimization with many local optima' },
    { id: 'de', name: 'Differential Evolution', description: 'Effective for continuous parameter optimization' },
    { id: 'es', name: 'Evolution Strategy', description: 'Suitable for noisy fitness functions' },
    { id: 'pso', name: 'Particle Swarm Optimization', description: 'Efficient for continuous optimization problems' }
  ],
  classical: [
    { id: 'nm', name: 'Nelder-Mead', description: 'Simplex-based direct search method' },
    { id: 'bfgs', name: 'BFGS', description: 'Quasi-Newton method for unconstrained optimization' },
    { id: 'cg', name: 'Conjugate Gradient', description: 'Effective for large-scale optimization' }
  ],
  bayesian: [
    { id: 'gp', name: 'Gaussian Process', description: 'Sample-efficient optimization for expensive functions' },
    { id: 'tpe', name: 'Tree-structured Parzen Estimator', description: 'Handles conditional parameter spaces' }
  ],
  hybrid: [
    { id: 'ga-de', name: 'GA-DE Hybrid', description: 'Combines discrete and continuous optimization' },
    { id: 'ga-es', name: 'GA-ES Hybrid', description: 'Robust to noise with good exploration' }
  ],
};

interface AlgorithmSuitabilityResult {
  score: number;
  reasons: string[];
}

// Algorithm suitability scoring based on problem characteristics
const getAlgorithmSuitability = (algorithm: Algorithm, problemCharacteristics: ProblemCharacteristics | undefined): AlgorithmSuitabilityResult => {
  if (!problemCharacteristics) return { score: 0, reasons: [] };
  
  let score = 0;
  const reasons: string[] = [];
  
  // Examples of suitability rules
  if (algorithm.id === 'de' && problemCharacteristics.modality === 'multimodal') {
    score += 2;
    reasons.push('Good for multimodal problems');
  }
  
  if (algorithm.id === 'ga' && problemCharacteristics.dimensionality === 'high') {
    score += 1;
    reasons.push('Can handle high-dimensional spaces');
  }
  
  if (algorithm.id === 'gp' && problemCharacteristics.evaluationCost === 'expensive') {
    score += 3;
    reasons.push('Sample-efficient for expensive evaluations');
  }
  
  // Add more complex rules based on algorithm strengths
  
  return { score, reasons };
};

interface AlgorithmSelectorProps {
  problemCharacteristics?: ProblemCharacteristics; 
  onAlgorithmSelect?: (algorithm: { 
    id: string; 
    name: string; 
    parameters: Record<string, unknown>;
  }) => void;
  preselectedAlgorithm?: string | null;
}

interface AlgorithmWithSuitability extends Algorithm {
  score: number;
  reasons: string[];
}

export default function AlgorithmSelector({ 
  problemCharacteristics, 
  onAlgorithmSelect, 
  preselectedAlgorithm = null 
}: AlgorithmSelectorProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('evolutionary');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>(preselectedAlgorithm || ALGORITHMS.evolutionary[0].id);
  const [autoConfig, setAutoConfig] = useState<boolean>(true);
  const [algorithmParams, setAlgorithmParams] = useState<Record<string, unknown>>({});
  const [apiRecommendations, setApiRecommendations] = useState<AlgorithmRecommendation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hyperparameterExplanations, setHyperparameterExplanations] = useState<Record<string, string>>({});
  // Add state for optimizers from API
  const [availableOptimizers, setAvailableOptimizers] = useState<Optimizer[]>([]);
  const [optimizerCategories, setOptimizerCategories] = useState<Record<string, Optimizer[]>>({});
  
  // Fetch available optimizers from API
  useEffect(() => {
    const loadOptimizers = async () => {
      setIsLoading(true);
      try {
        const optimizers = await fetchOptimizers();
        setAvailableOptimizers(optimizers);
        
        // Group optimizers by category
        const categories: Record<string, Optimizer[]> = {};
        optimizers.forEach(optimizer => {
          const category = optimizer.category || 'other';
          if (!categories[category]) {
            categories[category] = [];
          }
          categories[category].push(optimizer);
        });
        
        setOptimizerCategories(categories);
        
        // If we have optimizers and no algorithm is selected yet, select the first available one
        if (optimizers.length > 0 && !preselectedAlgorithm && !selectedAlgorithm) {
          setSelectedAlgorithm(optimizers[0].id);
        }
      } catch (error) {
        console.error('Error fetching optimizers:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadOptimizers();
  }, [preselectedAlgorithm, selectedAlgorithm]);
  
  // Fetch algorithm recommendations from API
  useEffect(() => {
    if (!problemCharacteristics) return;
    
    const fetchRecommendations = async () => {
      setIsLoading(true);
      try {
        const recommendations = await getAlgorithmRecommendations(problemCharacteristics);
        setApiRecommendations(recommendations);
        
        // If we have recommendations and no algorithm is selected yet, select the top one
        if (recommendations.length > 0 && !preselectedAlgorithm) {
          setSelectedAlgorithm(recommendations[0].algorithmId);
        }
      } catch (error) {
        console.error('Error fetching algorithm recommendations:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchRecommendations();
  }, [problemCharacteristics, preselectedAlgorithm]);
  
  // Generate recommendations based on problem characteristics
  const getRecommendations = useCallback((): AlgorithmWithSuitability[] => {
    // If we have API recommendations, use those
    if (apiRecommendations.length > 0) {
      return apiRecommendations.map(rec => ({
        id: rec.algorithmId,
        name: rec.algorithmName,
        description: rec.suitableFor.join(', '),
        score: rec.confidence * 10, // Scale to match our UI expectations
        reasons: rec.reasons
      }));
    }
    
    // Otherwise fall back to client-side recommendations
    if (!problemCharacteristics) return [];
    
    // Use available optimizers from API if available, otherwise fall back to hardcoded ones
    const allAlgorithms = availableOptimizers.length > 0 
      ? availableOptimizers.map(opt => ({
          id: opt.id,
          name: opt.name,
          description: opt.description || ''
        }))
      : Object.values(ALGORITHMS).flat();
      
    return allAlgorithms
      .map(alg => ({
        ...alg,
        ...getAlgorithmSuitability(alg, problemCharacteristics)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);
  }, [apiRecommendations, problemCharacteristics, availableOptimizers]);
  
  // Get default parameters for a given algorithm using the API or fallback
  const getDefaultParameters = useCallback(async (algorithmId: string): Promise<Record<string, unknown>> => {
    if (problemCharacteristics && autoConfig) {
      try {
        // First try to get optimizer-specific recommended parameters
        const optimizerParams = await getRecommendedParameters(
          algorithmId, 
          problemCharacteristics
        );
        
        if (Object.keys(optimizerParams).length > 0) {
          return optimizerParams;
        }
        
        // Fall back to hyperparameter recommendations from meta-optimizer API
        const hyperParams = await getHyperparameterRecommendations(algorithmId, problemCharacteristics);
        
        // Store explanations for the UI
        setHyperparameterExplanations(hyperParams.explanations || {});
        
        return hyperParams.parameters;
      } catch (error) {
        console.error('Error fetching parameter recommendations:', error);
      }
    }
    
    // Try to get default parameters from the optimizer details
    try {
      const optimizerDetails = await getOptimizerDetails(algorithmId);
      if (optimizerDetails && optimizerDetails.defaultParameters) {
        return optimizerDetails.defaultParameters;
      }
    } catch (error) {
      console.error('Error fetching optimizer details:', error);
    }
    
    // Default parameters as fallback
    const defaults: Record<string, Record<string, unknown>> = {
      ga: { populationSize: 100, generations: 100, crossoverRate: 0.8, mutationRate: 0.1 },
      de: { populationSize: 50, generations: 100, F: 0.8, CR: 0.9 },
      es: { populationSize: 100, generations: 100, sigma: 0.1 },
      pso: { particles: 30, iterations: 100, c1: 2.0, c2: 2.0 },
      nm: { maxIterations: 200, xatol: 0.0001, fatol: 0.0001 },
      bfgs: { maxIterations: 100, gtol: 1e-5 },
      cg: { maxIterations: 100, gtol: 1e-5 },
      gp: { initPoints: 5, nIterations: 50, acqFunc: 'ei' },
      tpe: { iterations: 100, prior: 'uniform' },
    };
    
    // Handle hybrid algorithms
    if (algorithmId === 'ga-de') {
      return {
        ...await getDefaultParameters('ga'), 
        ...await getDefaultParameters('de'), 
        switchGen: 50
      };
    }
    
    if (algorithmId === 'ga-es') {
      return {
        ...await getDefaultParameters('ga'), 
        ...await getDefaultParameters('es'), 
        switchGen: 50
      };
    }
    
    return defaults[algorithmId] || {};
  }, [problemCharacteristics, autoConfig]);

  // Update parameters when algorithm changes
  useEffect(() => {
    const updateAlgorithmParameters = async () => {
      const algorithm = Object.values(ALGORITHMS)
        .flat()
        .find(alg => alg.id === selectedAlgorithm);
        
      if (algorithm) {
        setIsLoading(true);
        try {
          // Get default or recommended parameters
          const defaultParams = await getDefaultParameters(selectedAlgorithm);
          setAlgorithmParams(defaultParams);
          
          // Notify parent component
          if (onAlgorithmSelect) {
            onAlgorithmSelect({
              id: selectedAlgorithm,
              name: algorithm.name,
              parameters: defaultParams
            });
          }
        } catch (error) {
          console.error('Error setting algorithm parameters:', error);
        } finally {
          setIsLoading(false);
        }
      }
    };
    
    updateAlgorithmParameters();
  }, [selectedAlgorithm, getDefaultParameters, onAlgorithmSelect]);
  
  // Update a specific parameter value
  const updateParameter = (key: string, value: unknown) => {
    setAlgorithmParams(prev => {
      const updated = { ...prev, [key]: value };
      
      // Notify parent component about parameter change
      if (onAlgorithmSelect) {
        const algorithm = Object.values(ALGORITHMS)
          .flat()
          .find(alg => alg.id === selectedAlgorithm);
          
        if (algorithm) {
          onAlgorithmSelect({
            id: selectedAlgorithm,
            name: algorithm.name,
            parameters: updated
          });
        }
      }
      
      return updated;
    });
  };
  
  const recommendations = getRecommendations();
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Algorithm Selection</CardTitle>
        <CardDescription>
          Select and configure optimization algorithms for your problem
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="category" className="w-full">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="category">By Category</TabsTrigger>
            <TabsTrigger value="recommended">Recommended</TabsTrigger>
            <TabsTrigger value="configuration">Configuration</TabsTrigger>
          </TabsList>
          
          <TabsContent value="category" className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              {Object.keys(optimizerCategories).length > 0 ? (
                Object.keys(optimizerCategories).map(category => (
                  <Button 
                    key={category}
                    variant={selectedCategory === category ? "default" : "outline"}
                    onClick={() => setSelectedCategory(category)}
                    className="justify-start"
                  >
                    {category.charAt(0).toUpperCase() + category.slice(1)}
                  </Button>
                ))
              ) : (
                Object.keys(ALGORITHMS).map(category => (
                  <Button 
                    key={category}
                    variant={selectedCategory === category ? "default" : "outline"}
                    onClick={() => setSelectedCategory(category)}
                    className="justify-start"
                  >
                    {category.charAt(0).toUpperCase() + category.slice(1)}
                  </Button>
                ))
              )}
            </div>
            
            <div className="space-y-2 mt-4">
              <Label>Select Algorithm</Label>
              <div className="grid gap-2">
                {isLoading ? (
                  <div className="flex items-center justify-center p-6">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : optimizerCategories[selectedCategory] ? (
                  optimizerCategories[selectedCategory].map((optimizer) => (
                    <div 
                      key={optimizer.id}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedAlgorithm === optimizer.id ? 
                        'border-primary bg-primary/10' : 
                        'border-border hover:border-primary/50'
                      }`}
                      onClick={() => setSelectedAlgorithm(optimizer.id)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium">{optimizer.name}</h4>
                          <p className="text-sm text-muted-foreground">{optimizer.description || 'No description available'}</p>
                        </div>
                        {!optimizer.available && (
                          <Badge variant="outline" className="text-yellow-500 border-yellow-500">
                            Not Available
                          </Badge>
                        )}
                        {selectedAlgorithm === optimizer.id && (
                          <CheckCircle2 className="h-5 w-5 text-primary" />
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  ALGORITHMS[selectedCategory as keyof typeof ALGORITHMS]?.map((algorithm: Algorithm) => (
                    <div 
                      key={algorithm.id}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedAlgorithm === algorithm.id ? 
                        'border-primary bg-primary/10' : 
                        'border-border hover:border-primary/50'
                      }`}
                      onClick={() => setSelectedAlgorithm(algorithm.id)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium">{algorithm.name}</h4>
                          <p className="text-sm text-muted-foreground">{algorithm.description}</p>
                        </div>
                        {selectedAlgorithm === algorithm.id && (
                          <CheckCircle2 className="h-5 w-5 text-primary" />
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="recommended">
            <div className="space-y-4">
              <div className="bg-muted p-3 rounded-md">
                <div className="flex items-center gap-2">
                  {isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-muted-foreground" />
                  )}
                  <p className="text-sm">
                    {isLoading && 'Analyzing problem characteristics...'}
                    {!isLoading && problemCharacteristics ? 
                      'Recommendations based on your problem characteristics:' : 
                      'Upload or select a dataset to get algorithm recommendations'}
                  </p>
                </div>
              </div>
              
              {recommendations.length > 0 ? (
                <div className="space-y-2">
                  {recommendations.map(algorithm => (
                    <div 
                      key={algorithm.id}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedAlgorithm === algorithm.id ? 
                        'border-primary bg-primary/10' : 
                        'border-border hover:border-primary/50'
                      }`}
                      onClick={() => setSelectedAlgorithm(algorithm.id)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium">{algorithm.name}</h4>
                            <Badge variant="outline" className="ml-2">
                              Score: {typeof algorithm.score === 'number' ? algorithm.score.toFixed(1) : 'N/A'}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">{algorithm.description}</p>
                          <div className="mt-2 flex flex-wrap gap-1">
                            {algorithm.reasons?.map((reason, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
                                {reason}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        {selectedAlgorithm === algorithm.id && (
                          <CheckCircle2 className="h-5 w-5 text-primary" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : !isLoading && (
                <div className="text-center py-6 text-muted-foreground">
                  <p>No recommendations available.</p>
                  <p className="text-sm mt-1">Try selecting a dataset or modifying your problem characteristics.</p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="configuration" className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="auto-config">Automatic configuration</Label>
              <Switch 
                id="auto-config" 
                checked={autoConfig}
                onCheckedChange={setAutoConfig}
              />
            </div>
            
            <div className="space-y-3">
              {isLoading ? (
                <div className="flex justify-center py-6">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                Object.entries(algorithmParams).map(([key, value]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Label htmlFor={key} className="capitalize">{key.replace(/([A-Z])/g, ' $1')}</Label>
                        {hyperparameterExplanations[key] && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="w-60">{hyperparameterExplanations[key]}</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                      </div>
                      {typeof value === 'number' && (
                        <span className="text-sm font-mono">{value}</span>
                      )}
                    </div>
                    
                    {typeof value === 'number' && (
                      <div className="flex gap-2 items-center">
                        <Slider
                          id={key}
                          disabled={autoConfig}
                          value={[value as number]}
                          max={key.includes('Size') || key.includes('iterations') || key.includes('generations') ? 500 : key.includes('Rate') ? 1 : 10}
                          min={key.includes('Size') || key.includes('iterations') || key.includes('generations') ? 10 : 0.01}
                          step={key.includes('Size') || key.includes('iterations') || key.includes('generations') ? 5 : 0.01}
                          onValueChange={(vals) => updateParameter(key, vals[0])}
                          className="flex-1"
                        />
                        <Input
                          type="number"
                          disabled={autoConfig}
                          value={value as number}
                          onChange={(e) => updateParameter(key, parseFloat(e.target.value))}
                          className="w-20"
                        />
                      </div>
                    )}
                    
                    {typeof value === 'string' && (
                      <Input
                        id={key}
                        disabled={autoConfig}
                        value={value as string}
                        onChange={(e) => updateParameter(key, e.target.value)}
                      />
                    )}
                    
                    {typeof value === 'boolean' && (
                      <Switch
                        id={key}
                        disabled={autoConfig}
                        checked={value as boolean}
                        onCheckedChange={(checked) => updateParameter(key, checked)}
                      />
                    )}
                  </div>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 