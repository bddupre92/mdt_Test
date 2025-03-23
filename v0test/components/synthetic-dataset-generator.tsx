"use client"

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2, Database, BarChart, LineChart, PieChart, ScatterChart } from 'lucide-react';
import { generateSyntheticDataset } from '@/lib/api/datasets';

// Types for synthetic dataset parameters
interface SyntheticDatasetParams {
  name: string;
  description: string;
  category: string;
  type: 'regression' | 'classification' | 'time-series' | 'clustering';
  features: number;
  samples: number;
  noise: number;
  complexity: 'low' | 'medium' | 'high';
  missingValues: number;
  outlierPercentage: number;
  classImbalance?: number; // Only for classification
  timeSteps?: number; // Only for time-series
  seasonality?: boolean; // Only for time-series
  trend?: boolean; // Only for time-series
  clusters?: number; // Only for clustering
  correlatedFeatures?: boolean; // Whether to include correlated features
}

interface SyntheticDatasetGeneratorProps {
  onDatasetGenerated?: (datasetId: string) => void;
}

export function SyntheticDatasetGenerator({ onDatasetGenerated }: SyntheticDatasetGeneratorProps) {
  const [activeTab, setActiveTab] = useState('basic');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Default parameters
  const [params, setParams] = useState<SyntheticDatasetParams>({
    name: 'Synthetic Dataset',
    description: 'Automatically generated synthetic dataset',
    category: 'tabular',
    type: 'classification',
    features: 10,
    samples: 1000,
    noise: 0.1,
    complexity: 'medium',
    missingValues: 0,
    outlierPercentage: 0,
    classImbalance: 0.5, // balanced
    correlatedFeatures: false
  });
  
  const handleInputChange = (field: keyof SyntheticDatasetParams, value: any) => {
    setParams(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  const handleTypeChange = (type: 'regression' | 'classification' | 'time-series' | 'clustering') => {
    // Reset specific parameters when type changes
    const newParams: Partial<SyntheticDatasetParams> = { type };
    
    if (type === 'time-series') {
      newParams.timeSteps = 100;
      newParams.seasonality = true;
      newParams.trend = true;
    } else if (type === 'classification') {
      newParams.classImbalance = 0.5;
    } else if (type === 'clustering') {
      newParams.clusters = 3;
    }
    
    setParams(prev => ({
      ...prev,
      ...newParams
    }));
  };
  
  const handleGenerateDataset = async () => {
    setIsGenerating(true);
    setError(null);
    
    try {
      const dataset = await generateSyntheticDataset(params);
      
      if (dataset && dataset.id) {
        console.log('Dataset generated successfully:', dataset);
        
        // Call the onDatasetGenerated callback with the new dataset ID
        if (onDatasetGenerated) {
          onDatasetGenerated(dataset.id);
        }
        
        // Reset form or show success
        setParams({
          ...params,
          name: `Synthetic Dataset ${Math.floor(Math.random() * 1000)}`, // Generate a new default name
        });
      } else {
        throw new Error('Failed to generate dataset');
      }
    } catch (err) {
      console.error('Error generating synthetic dataset:', err);
      setError('Failed to generate dataset. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };
  
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Synthetic Dataset Generator</CardTitle>
        <CardDescription>
          Create custom synthetic datasets with specific characteristics for algorithm testing
        </CardDescription>
      </CardHeader>
      
      <Tabs defaultValue="basic" value={activeTab} onValueChange={setActiveTab}>
        <div className="px-6 border-b">
          <TabsList>
            <TabsTrigger value="basic">Basic Settings</TabsTrigger>
            <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
            <TabsTrigger value="specific">Type-specific Settings</TabsTrigger>
          </TabsList>
        </div>
        
        <CardContent className="p-6">
          <TabsContent value="basic" className="mt-0 space-y-4">
            {/* Name and Description */}
            <div className="space-y-4">
              <div>
                <Label htmlFor="dataset-name">Dataset Name</Label>
                <Input 
                  id="dataset-name" 
                  value={params.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                  placeholder="Give your dataset a name"
                />
              </div>
              
              <div>
                <Label htmlFor="dataset-description">Description</Label>
                <Textarea 
                  id="dataset-description"
                  value={params.description}
                  onChange={(e) => handleInputChange('description', e.target.value)}
                  placeholder="Briefly describe this dataset"
                  rows={3}
                />
              </div>
            </div>
            
            {/* Category and Type */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="dataset-category">Category</Label>
                <Select 
                  value={params.category}
                  onValueChange={(value) => handleInputChange('category', value)}
                >
                  <SelectTrigger id="dataset-category">
                    <SelectValue placeholder="Select category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tabular">Tabular</SelectItem>
                    <SelectItem value="time-series">Time Series</SelectItem>
                    <SelectItem value="spatial">Spatial</SelectItem>
                    <SelectItem value="text">Text</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="dataset-type">Type</Label>
                <Select 
                  value={params.type}
                  onValueChange={(value: any) => handleTypeChange(value)}
                >
                  <SelectTrigger id="dataset-type">
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="regression">Regression</SelectItem>
                    <SelectItem value="time-series">Time Series</SelectItem>
                    <SelectItem value="clustering">Clustering</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            {/* Size Parameters */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between">
                  <Label htmlFor="features-slider">Number of Features: {params.features}</Label>
                  <Input 
                    type="number" 
                    className="w-20" 
                    min={2} 
                    max={100}
                    value={params.features}
                    onChange={(e) => handleInputChange('features', parseInt(e.target.value) || 2)}
                  />
                </div>
                <Slider 
                  id="features-slider"
                  min={2}
                  max={100}
                  step={1}
                  value={[params.features]}
                  onValueChange={(value) => handleInputChange('features', value[0])}
                  className="mt-2"
                />
              </div>
              
              <div>
                <div className="flex justify-between">
                  <Label htmlFor="samples-slider">Number of Samples: {params.samples}</Label>
                  <Input 
                    type="number" 
                    className="w-20" 
                    min={100} 
                    max={10000}
                    value={params.samples}
                    onChange={(e) => handleInputChange('samples', parseInt(e.target.value) || 100)}
                  />
                </div>
                <Slider 
                  id="samples-slider"
                  min={100}
                  max={10000}
                  step={100}
                  value={[params.samples]}
                  onValueChange={(value) => handleInputChange('samples', value[0])}
                  className="mt-2"
                />
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="advanced" className="mt-0 space-y-4">
            {/* Complexity */}
            <div>
              <Label htmlFor="complexity">Complexity</Label>
              <RadioGroup 
                id="complexity" 
                value={params.complexity}
                onValueChange={(value: any) => handleInputChange('complexity', value)}
                className="flex space-x-4 mt-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="low" id="complexity-low" />
                  <Label htmlFor="complexity-low">Low</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="medium" id="complexity-medium" />
                  <Label htmlFor="complexity-medium">Medium</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="high" id="complexity-high" />
                  <Label htmlFor="complexity-high">High</Label>
                </div>
              </RadioGroup>
              <p className="text-xs text-gray-500 mt-1">
                Determines how complex the relationships between features will be
              </p>
            </div>
            
            {/* Noise Slider */}
            <div>
              <div className="flex justify-between">
                <Label htmlFor="noise-slider">Noise Level: {(params.noise * 100).toFixed(0)}%</Label>
              </div>
              <Slider 
                id="noise-slider"
                min={0}
                max={0.5}
                step={0.01}
                value={[params.noise]}
                onValueChange={(value) => handleInputChange('noise', value[0])}
                className="mt-2"
              />
              <p className="text-xs text-gray-500 mt-1">
                Adds random noise to the dataset to simulate real-world imperfections
              </p>
            </div>
            
            {/* Missing Values */}
            <div>
              <div className="flex justify-between">
                <Label htmlFor="missing-slider">Missing Values: {(params.missingValues * 100).toFixed(0)}%</Label>
              </div>
              <Slider 
                id="missing-slider"
                min={0}
                max={0.3}
                step={0.01}
                value={[params.missingValues]}
                onValueChange={(value) => handleInputChange('missingValues', value[0])}
                className="mt-2"
              />
              <p className="text-xs text-gray-500 mt-1">
                Percentage of values that will be set as missing/null
              </p>
            </div>
            
            {/* Outliers */}
            <div>
              <div className="flex justify-between">
                <Label htmlFor="outlier-slider">Outliers: {(params.outlierPercentage * 100).toFixed(0)}%</Label>
              </div>
              <Slider 
                id="outlier-slider"
                min={0}
                max={0.1}
                step={0.01}
                value={[params.outlierPercentage]}
                onValueChange={(value) => handleInputChange('outlierPercentage', value[0])}
                className="mt-2"
              />
              <p className="text-xs text-gray-500 mt-1">
                Percentage of data points that will be outliers
              </p>
            </div>
            
            {/* Correlated Features */}
            <div className="flex items-start space-x-2 pt-2">
              <Checkbox 
                id="correlated-features" 
                checked={params.correlatedFeatures}
                onCheckedChange={(checked) => handleInputChange('correlatedFeatures', checked)}
              />
              <div className="space-y-1">
                <Label 
                  htmlFor="correlated-features" 
                  className="font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Include Correlated Features
                </Label>
                <p className="text-xs text-gray-500">
                  Generates some features that are correlated with each other
                </p>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="specific" className="mt-0 space-y-4">
            {/* Type-specific controls based on the selected type */}
            {params.type === 'classification' && (
              <div>
                <div className="flex justify-between">
                  <Label htmlFor="balance-slider">Class Balance: {(params.classImbalance || 0.5) * 100}%</Label>
                </div>
                <Slider 
                  id="balance-slider"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={[params.classImbalance || 0.5]}
                  onValueChange={(value) => handleInputChange('classImbalance', value[0])}
                  className="mt-2"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Controls how balanced the classes are. 100% = perfectly balanced, lower values = more imbalanced
                </p>
              </div>
            )}
            
            {params.type === 'clustering' && (
              <div>
                <div className="flex justify-between">
                  <Label htmlFor="clusters-input">Number of Clusters</Label>
                  <Input 
                    id="clusters-input"
                    type="number" 
                    className="w-20" 
                    min={2} 
                    max={10}
                    value={params.clusters || 3}
                    onChange={(e) => handleInputChange('clusters', parseInt(e.target.value) || 2)}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Number of distinct clusters to generate in the dataset
                </p>
              </div>
            )}
            
            {params.type === 'time-series' && (
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between">
                    <Label htmlFor="timesteps-input">Number of Time Steps</Label>
                    <Input 
                      id="timesteps-input"
                      type="number" 
                      className="w-20" 
                      min={50} 
                      max={1000}
                      value={params.timeSteps || 100}
                      onChange={(e) => handleInputChange('timeSteps', parseInt(e.target.value) || 50)}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Number of time points in the series
                  </p>
                </div>
                
                <div className="flex items-start space-x-2">
                  <Checkbox 
                    id="seasonality" 
                    checked={params.seasonality}
                    onCheckedChange={(checked) => handleInputChange('seasonality', checked)}
                  />
                  <div className="space-y-1">
                    <Label 
                      htmlFor="seasonality" 
                      className="font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Include Seasonality
                    </Label>
                    <p className="text-xs text-gray-500">
                      Adds seasonal patterns to the time series data
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-2">
                  <Checkbox 
                    id="trend" 
                    checked={params.trend}
                    onCheckedChange={(checked) => handleInputChange('trend', checked)}
                  />
                  <div className="space-y-1">
                    <Label 
                      htmlFor="trend" 
                      className="font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Include Trend
                    </Label>
                    <p className="text-xs text-gray-500">
                      Adds upward or downward trends to the time series data
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            {/* Display visualization of what will be generated */}
            <div className="border rounded p-4 mt-6">
              <h3 className="text-sm font-medium mb-2">Data Preview (Simulated)</h3>
              <div className="h-40 flex items-center justify-center">
                {params.type === 'classification' && <PieChart className="h-32 w-32 text-blue-500 opacity-70" />}
                {params.type === 'regression' && <ScatterChart className="h-32 w-32 text-blue-500 opacity-70" />}
                {params.type === 'time-series' && <LineChart className="h-32 w-32 text-blue-500 opacity-70" />}
                {params.type === 'clustering' && <BarChart className="h-32 w-32 text-blue-500 opacity-70" />}
              </div>
              <p className="text-xs text-gray-500 text-center mt-2">
                This preview shows the expected shape of your generated dataset
              </p>
            </div>
          </TabsContent>
        </CardContent>
      </Tabs>
      
      <CardFooter className="px-6 py-4 bg-gray-50 flex justify-between">
        {error && (
          <div className="text-red-600 text-sm">{error}</div>
        )}
        <div className="flex-grow"></div>
        <Button 
          onClick={handleGenerateDataset}
          disabled={isGenerating || !params.name || !params.description}
        >
          {isGenerating ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Database className="mr-2 h-4 w-4" />
              Generate Dataset
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
} 