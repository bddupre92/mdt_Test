import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { BarChart } from "./charts";
import { Button } from "./ui/button";
import { Spinner } from "./ui/spinner";

// Define interfaces for the SATZilla prediction component
interface PredictionItem {
  optimizerId: string;
  confidence: number;
}

interface ProblemFeature {
  [key: string]: string | number;
}

interface PredictionQualityItem {
  algorithm: string;
  predictedPerformance: number;
  actualPerformance: number;
}

interface SATZillaPredictionProps {
  predictions: PredictionItem[];
  problemFeatures?: ProblemFeature;
  predictionQuality: PredictionQualityItem[];
  selectedOptimizer: string;
  algorithmNames: Record<string, string>;
  // Add datasets and models for page.tsx compatibility
  datasets?: any[];
  models?: any[];
  onExecute?: (datasetId: string, optimizerIds: string[], benchmarkId: string) => void;
  results?: any;
}

const SATZillaPrediction: React.FC<SATZillaPredictionProps> = ({
  predictions = [],
  problemFeatures = {},
  predictionQuality = [],
  selectedOptimizer = "",
  algorithmNames = {},
  datasets = [],
  models = [],
  onExecute,
  results,
}) => {
  // Helper function to get algorithm name
  const getAlgorithmName = (id: string) => {
    return algorithmNames[id] || id;
  };
  
  // Add state for form controls
  const [selectedDatasetId, setSelectedDatasetId] = React.useState<string>("");
  const [selectedOptimizers, setSelectedOptimizers] = React.useState<string[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = React.useState<string>("");
  const [isSubmitting, setIsSubmitting] = React.useState<boolean>(false);
  
  // Benchmark function options
  const benchmarkFunctions = [
    { id: "sphere", name: "Sphere Function" },
    { id: "rastrigin", name: "Rastrigin Function" },
    { id: "rosenbrock", name: "Rosenbrock Function" },
    { id: "ackley", name: "Ackley Function" },
    { id: "griewank", name: "Griewank Function" }
  ];
  
  // Organize problem features by type - with null checks
  const numericFeatures = Object.entries(problemFeatures || {})
    .filter(([_, value]) => typeof value === "number")
    .map(([key, value]) => ({
      x: key,
      y: value as number,
    }));
    
  const stringFeatures = Object.entries(problemFeatures || {})
    .filter(([_, value]) => typeof value === "string");

  // If we don't have problemFeatures or predictions but have datasets and models,
  // we should show a form to select dataset and optimizer for analysis
  const showSelectionForm = (!predictions || predictions.length === 0) && 
    datasets && datasets.length > 0 && models && models.length > 0;
    
  return (
    <div className="space-y-4">
      {showSelectionForm ? (
        <Card>
          <CardHeader>
            <CardTitle>SATZilla Meta-Learning Analysis</CardTitle>
            <CardDescription>
              Select a dataset and optimizers to analyze which algorithm would perform best
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              SATZilla uses machine learning to predict the best optimization algorithm 
              for specific problem types based on features extracted from the problem.
            </p>
            
            {/* Dataset and optimizer selection form */}
            <form 
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (onExecute && selectedDatasetId && selectedOptimizers.length > 0 && selectedBenchmark) {
                  setIsSubmitting(true);
                  onExecute(selectedDatasetId, selectedOptimizers, selectedBenchmark);
                  // The isSubmitting state would be reset when the predictions are loaded
                }
              }}
            >
              {/* Dataset selection */}
              <div className="space-y-2">
                <label className="block text-sm font-medium">Select Dataset</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                >
                  <option value="">-- Select a dataset --</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} {dataset.type && `(${dataset.type})`}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-muted-foreground">
                  Select any dataset to analyze which algorithm would perform best on it.
                </p>
              </div>

              {/* Benchmark function selection */}
              <div className="space-y-2">
                <label className="block text-sm font-medium">Benchmark Function</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={selectedBenchmark}
                  onChange={(e) => setSelectedBenchmark(e.target.value)}
                >
                  <option value="">-- Select benchmark --</option>
                  {benchmarkFunctions.map((bench) => (
                    <option key={bench.id} value={bench.id}>
                      {bench.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Optimizer multi-selection */}
              <div className="space-y-2">
                <label className="block text-sm font-medium">Select Optimizers to Compare</label>
                <div className="grid grid-cols-2 gap-2">
                  {models.filter(m => m.type === 'optimization').map((model) => (
                    <div key={model.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id={model.id}
                        checked={selectedOptimizers.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedOptimizers([...selectedOptimizers, model.id]);
                          } else {
                            setSelectedOptimizers(selectedOptimizers.filter(id => id !== model.id));
                          }
                        }}
                        className="h-4 w-4 rounded border-gray-300"
                      />
                      <label htmlFor={model.id} className="text-sm">
                        {model.name}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              <Button 
                type="submit" 
                className="w-full md:w-auto" 
                disabled={isSubmitting} 
              >
                {isSubmitting ? (
                  <div className="flex items-center gap-2">
                    <Spinner size="small" /> 
                    <span>Running Analysis...</span>
                  </div>
                ) : (
                  "Run SATZilla Analysis"
                )}
              </Button>
              
              {/* Add a loading progress bar when analysis is running */}
              {isSubmitting && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm text-muted-foreground">Analysis in progress. Please wait...</p>
                  <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                    <div className="bg-primary h-2.5 animate-pulse" style={{ width: "100%" }}></div>
                  </div>
                </div>
              )}
            </form>
            
            <p className="text-sm text-muted-foreground mt-4">
              Please select options from the form above to perform meta-learning analysis.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>SATZilla Algorithm Selection</CardTitle>
              <CardDescription>
                Machine learning prediction of the best algorithm for this problem based on 
                extracted features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Algorithm Selection and Confidence */}
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-muted rounded-lg p-3">
                    <p className="text-xs text-muted-foreground">Selected Algorithm</p>
                    <div className="text-lg font-semibold flex items-center">
                      {getAlgorithmName(selectedOptimizer)}
                      <Badge className="ml-2" variant="outline">Best</Badge>
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <p className="text-xs text-muted-foreground">Selection Confidence</p>
                    <p className="text-lg font-semibold">
                      {(predictions.find(p => p.optimizerId === selectedOptimizer)?.confidence || 0) * 100}%
                    </p>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium mb-2">Algorithm Predictions</h3>
                  <div className="space-y-3">
                    {predictions.map((pred) => {
                      const isSelected = selectedOptimizer === pred.optimizerId;
                      
                      return (
                        <div key={pred.optimizerId} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-sm flex items-center">
                              {getAlgorithmName(pred.optimizerId)}
                              {isSelected && (
                                <Badge className="ml-2" variant="secondary">Selected</Badge>
                              )}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              {(pred.confidence * 100).toFixed(1)}% confidence
                            </span>
                          </div>
                          <Progress 
                            value={pred.confidence * 100} 
                            className={isSelected ? "bg-primary/20" : ""} 
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
              
              {/* Problem Features */}
              {(stringFeatures.length > 0 || numericFeatures.length > 0) && (
                <div className="space-y-4">
                  <h3 className="text-sm font-medium">Problem Characteristics</h3>
                  {stringFeatures.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {stringFeatures.map(([key, value]) => (
                        <div key={key} className="flex justify-between p-2 bg-muted/50 rounded-md">
                          <span className="text-sm">{key}</span>
                          <Badge variant="outline">{String(value)}</Badge>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {numericFeatures.length > 0 && (
                    <div className="h-[200px] bg-muted/50 rounded-md p-2">
                      <BarChart
                        data={numericFeatures}
                        xLabel="Feature"
                        yLabel="Value"
                      />
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
          
          {predictionQuality && predictionQuality.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Prediction Quality</CardTitle>
                <CardDescription>
                  Analyzing how well the algorithm selection predictions match actual performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    SATZilla uses landscape analysis to build a machine learning model
                    that can predict which algorithm will perform best on a given problem.
                    This visualization shows how well the predictions align with actual performance.
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {predictionQuality.map((item) => {
                      // Normalize values to ensure they're in the 0-1 range
                      const normalizedPredicted = Math.min(Math.max(item.predictedPerformance, 0), 1);
                      const normalizedActual = Math.min(Math.max(item.actualPerformance, 0), 1);
                      const error = Math.abs(normalizedPredicted - normalizedActual);
                      const errorPercentage = (error * 100).toFixed(1);
                      const errorClass = 
                        error < 0.1 ? "text-green-600" : 
                        error < 0.2 ? "text-yellow-600" : 
                        "text-red-600";
                        
                      return (
                        <div key={item.algorithm} className="p-3 bg-muted/50 rounded-md">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium">{getAlgorithmName(item.algorithm)}</span>
                            <span className={`text-xs font-medium ${errorClass}`}>
                              Error: {errorPercentage}%
                            </span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs w-16">Predicted:</span>
                              <Progress 
                                value={normalizedPredicted * 100} 
                                className="bg-blue-100 h-2" 
                              />
                              <span className="text-xs w-12 text-right">
                                {(normalizedPredicted * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs w-16">Actual:</span>
                              <Progress 
                                value={normalizedActual * 100} 
                                className="bg-green-100 h-2" 
                              />
                              <span className="text-xs w-12 text-right">
                                {(normalizedActual * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="mt-2 pt-2 border-t border-muted">
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>
                                Accuracy: {Math.max(0, 100 - parseFloat(errorPercentage)).toFixed(1)}%
                              </span>
                              <span>{normalizedActual > normalizedPredicted ? "Underestimated" : "Overestimated"}</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Add detailed algorithm performance metrics comparison */}
          {predictionQuality && predictionQuality.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Algorithm Performance Comparison</CardTitle>
                <CardDescription>
                  Detailed comparison of predicted vs. actual algorithm performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="border rounded-md overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-muted">
                      <tr>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Algorithm
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Predicted
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actual
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Error
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {predictionQuality
                        .sort((a, b) => b.actualPerformance - a.actualPerformance) // Sort by actual performance
                        .map((item) => {
                          // Normalize values to ensure they're in the 0-1 range
                          const normalizedPredicted = Math.min(Math.max(item.predictedPerformance, 0), 1);
                          const normalizedActual = Math.min(Math.max(item.actualPerformance, 0), 1);
                          const error = Math.abs(normalizedPredicted - normalizedActual);
                          const errorPercentage = (error * 100).toFixed(1);
                          const errorClass = 
                            error < 0.1 ? "text-green-600" : 
                            error < 0.2 ? "text-yellow-600" : 
                            "text-red-600";
                          const status = normalizedActual > normalizedPredicted 
                            ? "Underestimated" 
                            : "Overestimated";
                            
                          return (
                            <tr key={item.algorithm}>
                              <td className="px-4 py-2 whitespace-nowrap text-sm font-medium">
                                {getAlgorithmName(item.algorithm)}
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap text-sm">
                                {(normalizedPredicted * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap text-sm">
                                {(normalizedActual * 100).toFixed(1)}%
                              </td>
                              <td className={`px-4 py-2 whitespace-nowrap text-sm ${errorClass} font-medium`}>
                                {(error * 100).toFixed(1)}%
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap text-xs text-muted-foreground">
                                {status}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4">
                  <p className="text-sm text-muted-foreground">
                    This table shows the comparison between predicted and actual performance for each algorithm.
                    The "Error" column shows the absolute difference between predicted and actual values.
                    Lower error values indicate more accurate predictions by the SATZilla meta-learning system.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
      
      {/* Add detailed performance metrics section when predictions are available */}
      {predictions.length > 0 && predictionQuality.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Algorithm Performance Metrics</CardTitle>
            <CardDescription>
              Visualization of algorithm performance across different metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Algorithm Comparison Table */}
              <div>
                <h3 className="text-sm font-medium mb-2">Performance Comparison</h3>
                <div className="border rounded-md overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-muted">
                      <tr>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Algorithm
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Predicted
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actual
                        </th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Error
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {predictionQuality.map((item) => {
                        // Normalize values to ensure they're in the 0-1 range
                        const normalizedPredicted = Math.min(Math.max(item.predictedPerformance, 0), 1);
                        const normalizedActual = Math.min(Math.max(item.actualPerformance, 0), 1);
                        const error = Math.abs(normalizedPredicted - normalizedActual);
                        const errorClass = 
                          error < 0.1 ? "text-green-600" : 
                          error < 0.2 ? "text-yellow-600" : 
                          "text-red-600";
                          
                        return (
                          <tr key={item.algorithm}>
                            <td className="px-4 py-2 whitespace-nowrap text-sm font-medium">
                              {getAlgorithmName(item.algorithm)}
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap text-sm">
                              {(normalizedPredicted * 100).toFixed(1)}%
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap text-sm">
                              {(normalizedActual * 100).toFixed(1)}%
                            </td>
                            <td className={`px-4 py-2 whitespace-nowrap text-sm ${errorClass} font-medium`}>
                              {(error * 100).toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* Prediction Accuracy Visualization */}
              <div>
                <h3 className="text-sm font-medium mb-2">Prediction Accuracy</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {predictionQuality.map((item) => (
                    <div key={item.algorithm} className="bg-muted/30 p-3 rounded-md">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">{getAlgorithmName(item.algorithm)}</span>
                        <span className="text-xs">
                          Error: {(Math.abs(item.predictedPerformance - item.actualPerformance) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs w-16">Predicted:</span>
                          <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-blue-600 h-2.5 rounded-full" 
                              style={{ width: `${item.predictedPerformance * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs w-10 text-right">{(item.predictedPerformance * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs w-16">Actual:</span>
                          <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-green-600 h-2.5 rounded-full" 
                              style={{ width: `${item.actualPerformance * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs w-10 text-right">{(item.actualPerformance * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SATZillaPrediction; 