"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { LineChart, BarChart, ScatterChart } from "@/components/charts"
import { FileUploader } from "@/components/file-uploader"
import { AlertCircle, CheckCircle2, Info, Download, ExternalLink } from "lucide-react"
import { ConfusionMatrix } from "@/components/confusion-matrix"
import { FeatureImportanceVisualization } from "@/components/feature-importance-visualization"
import { HyperparameterTuning } from "@/components/hyperparameter-tuning"
import { BenchmarkComparison } from "@/components/benchmark-comparison"
import { exportChartAsImage, generateExportFileName } from "@/lib/utils/export-chart"
import SATZillaPrediction from "../components/satzilla-prediction"
import { FrameworkRunner } from "@/components/framework-runner"
import { Spinner } from "@/components/ui/spinner"
import Link from "next/link"

// Add this interface at the top of the file, outside any functions
interface SATZillaMetrics {
  avgBestFitness: number;
  avgExecutionTime: number;
  bestOptimizer: string;
  bestFitness: number;
  selectionAccuracy: number;
}

export default function DataVisualizationHub() {
  // State for data management
  const [datasets, setDatasets] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [benchmarks, setBenchmarks] = useState<any[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string | undefined>(undefined)
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined)
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>("rastrigin")
  const [currentData, setCurrentData] = useState<any>(null)
  const [tuningConfig, setTuningConfig] = useState<any>(null)
  const [results, setResults] = useState<any>(null)

  // State for model execution
  const [isExecuting, setIsExecuting] = useState<boolean>(false)
  const [executionProgress, setExecutionProgress] = useState<number>(0)

  // Add visualization config state
  const [visualizationConfig, setVisualizationConfig] = useState<{
    type: string;
    features: string[];
    target: string;
    showConfidenceIntervals: boolean;
  }>({
    type: "line",
    features: [],
    target: "",
    showConfidenceIntervals: true,
  })

  // State for notifications
  const [notification, setNotification] = useState<{
    type: "success" | "error" | "info" | "warning"
    message: string
  } | null>(null)

  // State for active tab
  const [activeTab, setActiveTab] = useState("visualization")

  // Refs for charts to enable exporting
  const chartRefs = {
    lineChart: useRef<HTMLDivElement>(null),
    barChart: useRef<HTMLDivElement>(null),
    scatterChart: useRef<HTMLDivElement>(null),
  }

  // Add this with the other state variables
  const [isBenchmarkMode, setIsBenchmarkMode] = useState<boolean>(false)
  const [selectedTab, setSelectedTab] = useState<string>("visualization")
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null)

  // State for confusion matrix and hyperparameter tuning
  const [confusionMatrix, setConfusionMatrix] = useState<any>(null)
  const [hyperparameterResults, setHyperparameterResults] = useState<any>(null)

  // Add refs for chart containers
  const visualizationRef = useRef<HTMLDivElement>(null)

  // After the model state in the Dashboard component
  const [progress, setProgress] = useState(0)
  const [progressStatus, setProgressStatus] = useState("")
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [executionState, setExecutionState] = useState<"idle" | "running" | "completed">("idle")

  // State for model parameters
  const [modelParams, setModelParams] = useState({
    // Linear regression parameters
    regularization: 0.01,
    learningRate: 0.01,
    
    // Random forest parameters
    nEstimators: 100,
    maxDepth: 10,
    
    // Differential evolution parameters
    populationSize: 50,
    F: 0.8,
    CR: 0.9,
    
    // Evolution strategy parameters
    sigma: 1.0,
    adaptSigma: true,
    
    // Ant colony parameters
    numAnts: 20,
    alpha: 1.0,
    beta: 2.0,
    evaporationRate: 0.1,
    
    // Grey wolf parameters
    numWolves: 30,
    a: 2.0,
    
    // Meta-optimizer parameters
    includeDe: true,
    includeEs: true,
    includeAco: true,
    includeGwo: true,
    selectionBudget: 5,
    selectionCriteria: "performance" as "performance" | "efficiency" | "robustness" | "adaptive",
    
    // Surrogate optimizer parameters
    initialPoints: 20,
    exploitationRatio: 0.5,
    
    // General parameters
    normalize: true,
    featureSelection: false
  });

  // Add missing state for benchmark function
  const [benchmarkFunction, setBenchmarkFunction] = useState("rastrigin");

  // Add state for the benchmark comparison
  const [compareAlgorithms, setCompareAlgorithms] = useState(true);

  // Add a new state variable for tracking whether to show the ML selection tab
  const [showMlSelection, setShowMlSelection] = useState(false);

  // First, add a new state to track multiple selected models
  const [selectedModelsForTuning, setSelectedModelsForTuning] = useState<string[]>([]);

  // Add new state for paper visualizations
  const [paperVisualizations, setPaperVisualizations] = useState<any[]>([])
  const [isLoadingVisualizations, setIsLoadingVisualizations] = useState(false)

  // Load initial data
  useEffect(() => {
    // Load available datasets and models
    const loadInitialData = async () => {
      try {
        // Fetch datasets from the API
        const apiDatasets = await fetch('/api/datasets').then(res => res.json());
        
        // Combine API datasets with benchmark datasets
        const allDatasets = [
          ...apiDatasets.datasets, // Real datasets from the data explorer
          // Benchmark datasets for optimization testing
          { id: "benchmark-sphere", name: "Sphere Function", type: "benchmark" },
          { id: "benchmark-rastrigin", name: "Rastrigin Function", type: "benchmark" },
          { id: "benchmark-rosenbrock", name: "Rosenbrock Function", type: "benchmark" },
          { id: "benchmark-ackley", name: "Ackley Function", type: "benchmark" },
          { id: "benchmark-griewank", name: "Griewank Function", type: "benchmark" },
        ];
        
        setDatasets(allDatasets);

        // Set optimizer models
        setModels([
          { id: "linear-regression", name: "Linear Regression", type: "regression" },
          { id: "random-forest", name: "Random Forest", type: "classification" },
          { id: "differential-evolution", name: "Differential Evolution (DE)", type: "optimization" },
          { id: "evolution-strategy", name: "Evolution Strategy (ES)", type: "optimization" },
          { id: "ant-colony", name: "Ant Colony Optimization (ACO)", type: "optimization" },
          { id: "grey-wolf", name: "Grey Wolf Optimizer (GWO)", type: "optimization" },
          { id: "meta-optimizer", name: "Meta-Optimizer", type: "optimization" },
          { id: "surrogate-optimizer", name: "Surrogate ML Optimizer (SATZilla)", type: "optimization" },
        ])

        setNotification({
          type: "info",
          message: "System initialized. Please select a dataset and model to begin.",
        })
      } catch (error) {
        console.error("Failed to load initial data:", error);
        setNotification({
          type: "error",
          message: "Failed to load initial data. Please refresh the page.",
        })
      }
    }

    loadInitialData()
  }, [])

  // Add a new useEffect to load paper visualizations
  useEffect(() => {
    const loadPaperVisualizations = async () => {
      try {
        setIsLoadingVisualizations(true)
        // In a real implementation, this would fetch from the API
        // For now, we'll mock the data
        setTimeout(() => {
          setPaperVisualizations([
            {
              id: 'performance_comparison',
              title: 'Algorithm Performance Comparison',
              description: 'Comparison of evolutionary algorithms on benchmark functions',
              image: '/api/visualizations?file=performance_comparison_best_fitness_avg.png'
            },
            {
              id: 'algorithm_selection',
              title: 'Algorithm Selection Analysis',
              description: 'Frequency and feature importance for algorithm selection',
              image: '/api/visualizations?file=algorithm_selection_frequency.png'
            }
          ])
          setIsLoadingVisualizations(false)
        }, 1000)
      } catch (error) {
        console.error('Error loading paper visualizations:', error)
        setIsLoadingVisualizations(false)
      }
    }

    loadPaperVisualizations()
  }, [])

  // Handle dataset selection
  const handleDatasetSelect = async (datasetId: string) => {
    setSelectedDataset(datasetId)
    setIsExecuting(true)
    setExecutionProgress(30)

    try {
      // Simulate loading dataset
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Generate synthetic data based on selected dataset
      const syntheticData = generateSyntheticData(datasetId)
      setCurrentData(syntheticData)

      setExecutionProgress(100)
      setNotification({
        type: "success",
        message: `Dataset "${datasetId}" loaded successfully.`,
      })
    } catch (error) {
      setNotification({
        type: "error",
        message: `Failed to load dataset "${datasetId}".`,
      })
    } finally {
      setIsExecuting(false)
      setExecutionProgress(0)
    }
  }

  // Handle model execution
  const handleExecuteModel = async () => {
    // Make sure currentData exists and has the minimum required properties
    if (!currentData || !selectedModel) {
      setNotification({
        type: "error",
        message: "Please select both a dataset and a model before execution."
      });
      return;
    }
    
    // Validate that currentData has the expected structure
    if (!currentData.id || !currentData.name) {
      setNotification({
        type: "error",
        message: "Selected dataset is missing required properties."
      });
      return;
    }
    
    setExecutionProgress(0);
    setIsExecuting(true);
    
    // Clear any previous results
    setResults(null);
    
    try {
      // Here we would normally send the request to an API endpoint
      // For the demo, we'll simulate the execution
      
      // Simulate progress
      const progressInterval = setInterval(() => {
        setExecutionProgress((prev: number) => {
          if (prev >= 99) {
            clearInterval(progressInterval);
            return 100;
          }
          return prev + Math.floor(Math.random() * 10);
        });
      }, 200);
      
      // Simulate delay for processing
      await new Promise((resolve) => setTimeout(resolve, 2500));
      
      let resultData: any = {
        id: `exec-${Date.now()}`,
        timestamp: new Date().toISOString(),
        datasetId: currentData.id,
        datasetName: currentData.name,
        modelId: selectedModel,
        modelName: models.find(m => m.id === selectedModel)?.name,
        metrics: {
          accuracy: 0.85 + Math.random() * 0.1,
          f1Score: 0.82 + Math.random() * 0.1,
          precision: 0.84 + Math.random() * 0.1,
          recall: 0.86 + Math.random() * 0.1,
          auc: 0.89 + Math.random() * 0.1,
          executionTime: Math.floor(1500 + Math.random() * 1000),
        },
        visualizations: [
          {
            type: "scatter",
            x: Array.from({ length: 100 }, (_, i) => i / 100),
            y: Array.from({ length: 100 }, () => Math.random()),
            xLabel: "Time",
            yLabel: "Progress",
          },
          {
            type: "convergence",
            iterations: Array.from({ length: 30 }, (_, i) => i + 1),
            fitness: Array.from({ length: 30 }, () => Math.random() * 0.5 + 0.5 - Math.random() * 0.1 * Math.exp(-Math.random())),
          }
        ],
        featureImportance: Array.isArray(currentData.features) 
          ? currentData.features.map((feature: string, index: number) => ({
              feature,
              importance: Math.random()
            })).sort((a: {feature: string, importance: number}, b: {feature: string, importance: number}) => b.importance - a.importance)
          : // Handle cases where features is an object or missing entirely
            Object.keys(currentData.features || {}).length > 0
              ? Object.keys(currentData.features || {}).map(feature => ({
                  feature,
                  importance: Math.random()
                })).sort((a, b) => b.importance - a.importance)
              : // Create default features if none exist
                [
                  { feature: "Feature 1", importance: 0.8 },
                  { feature: "Feature 2", importance: 0.6 },
                  { feature: "Feature 3", importance: 0.4 },
                  { feature: "Feature 4", importance: 0.3 },
                  { feature: "Feature 5", importance: 0.2 }
                ],
      };
      
      // Add SATZilla-specific data for algorithm selection if using surrogate model
      if (selectedModel.includes("surrogate")) {
        // Set flag to show ML tab
        setShowMlSelection(true);
        
        // Generate list of optimizer algorithms
        const optimizers = [
          { id: "genetic", name: "Genetic Algorithm" },
          { id: "particle_swarm", name: "Particle Swarm" },
          { id: "simulated_annealing", name: "Simulated Annealing" },
          { id: "differential_evolution", name: "Differential Evolution" },
          { id: "cmaes", name: "CMA-ES" }
        ];
        
        // Generate synthetic ML predictions (confidences) for each algorithm
        let algorithmConfidences = optimizers.map(opt => ({
          optimizerId: opt.id,
          confidence: Math.random() * 0.4 + 0.3 // Between 0.3 and 0.7
        }));
        
        // Normalize confidences so they sum to 1
        const totalConfidence = algorithmConfidences.reduce((sum, alg) => sum + alg.confidence, 0);
        algorithmConfidences = algorithmConfidences.map(alg => ({
          ...alg,
          confidence: alg.confidence / totalConfidence
        }));
        
        // Sort by confidence (higher is better)
        algorithmConfidences.sort((a, b) => b.confidence - a.confidence);
        
        // Randomly select one as the "best"
        const bestIndex = Math.floor(Math.random() * optimizers.length);
        algorithmConfidences[bestIndex].confidence = Math.random() * 0.2 + 0.8; // Between 0.8 and 1.0
        
        // Problem features that SATZilla would analyze
        const problemFeatures = {
          "dimensionality": Array.isArray(currentData.features)
            ? currentData.features.length
            : (currentData.features ? Object.keys(currentData.features).length : 5),
          "datasetSize": currentData.rows || 100,
          "featureEntropy": (0.4 + Math.random() * 0.5).toFixed(2),
          "problemType": selectedModel.includes("classification") ? "Classification" : "Regression",
          "featureCorrelation": (0.2 + Math.random() * 0.6).toFixed(2),
          "landscapeRuggedness": (0.1 + Math.random() * 0.9).toFixed(2),
          "modalityEstimate": Math.floor(1 + Math.random() * 4),
          "gradientStructure": Math.random() > 0.5 ? "Smooth" : "Discontinuous",
          "optimizationComplexity": Math.random() > 0.7 ? "High" : Math.random() > 0.4 ? "Medium" : "Low",
        };
        
        // Generate comparison data between predicted and actual performance
        const predictionQuality = optimizers.map(opt => {
          const predicted = algorithmConfidences.find(c => c.optimizerId === opt.id)?.confidence || 0;
          // Add some noise to actual vs predicted
          const actual = predicted * (0.7 + Math.random() * 0.6);
          
          return {
            algorithm: opt.name,
            predictedPerformance: predicted,
            actualPerformance: actual
          };
        });
        
        // Add to the result data
        resultData = {
          ...resultData,
          mlPredictions: algorithmConfidences,
          problemFeatures,
          predictionQuality,
          metrics: {
            ...resultData.metrics,
            // Custom properties for SATZilla results
            selectedOptimizer: algorithmConfidences[0].optimizerId,
            selectionAccuracy: 0.7 + Math.random() * 0.3,
          }
        };
      } else {
        // Reset the flag when not using a surrogate model
        setShowMlSelection(false);
      }
      
      clearInterval(progressInterval);
      setExecutionProgress(100);
      setResults(resultData);
      
      setNotification({
        type: "success",
        message: `Model ${models.find(m => m.id === selectedModel)?.name || selectedModel} executed successfully.`,
      });
    } catch (error: any) {
      setIsExecuting(false);
      setNotification({
        type: "error",
        message: `Error executing model: ${error.message}`,
      });
    } finally {
      setIsExecuting(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setIsExecuting(true)
    setExecutionProgress(20)

    try {
      // Simulate processing uploaded file
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // In a real application, you would parse the file here
      const fileData = await readFileAsJson(file)

      // Add the uploaded dataset to the list
      const newDataset = {
        id: `uploaded-${Date.now()}`,
        name: file.name,
        type: "custom",
        source: "upload",
      }

      setDatasets((prev) => [...prev, newDataset])
      setSelectedDataset(newDataset.id)
      setCurrentData(fileData)

      setExecutionProgress(100)
      setNotification({
        type: "success",
        message: `File "${file.name}" uploaded and processed successfully.`,
      })
    } catch (error) {
      setNotification({
        type: "error",
        message: `Failed to process uploaded file. Please ensure it's in the correct format.`,
      })
    } finally {
      setTimeout(() => {
        setIsExecuting(false)
        setExecutionProgress(0)
      }, 500)
    }
  }

  // Helper function to read file as JSON
  const readFileAsJson = async (file: File) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target?.result as string)
          resolve(json)
        } catch (error) {
          reject(error)
        }
      }
      reader.onerror = (e) => reject(e)
      reader.readAsText(file)
    })
  }

  // Helper function to generate synthetic data for SATZilla analysis
  const generateSyntheticData = (datasetId: string) => {
    // Find the dataset in our list
    const dataset = datasets.find(d => d.id === datasetId);
    
    if (!dataset) {
      // If dataset doesn't exist, generate a basic placeholder
      return {
        id: datasetId,
        name: `Dataset ${datasetId}`,
        type: 'synthetic',
        features: ['feature_1', 'feature_2', 'feature_3'],
        dimensions: 3,
        samples: 100,
        data: Array.from({ length: 100 }, () => 
          Array.from({ length: 3 }, () => Math.random() * 10)
        ),
        metadata: {
          description: "Synthetic data generated for analysis",
          type: "synthetic"
        }
      };
    }
    
    // If dataset exists but has no data, generate it based on the dataset type
    if (!dataset.data) {
      return generateSyntheticDataset(dataset.type || 'regression', 100, 3);
    }
    
    // Return the existing dataset
    return dataset;
  };

  // Helper function to generate model results
  const generateModelResults = (modelId: string, data: any) => {
    const results: any = {
      predictions: [],
      metrics: {},
      featureImportance: {},
      executionTime: Math.random() * 2 + 0.5, // Random execution time between 0.5 and 2.5 seconds
    }

    // Check if data and features exist
    if (!data || !data.features) {
      console.warn(`Missing data or features for model ${modelId}`);
      // Return default results
      results.metrics = {
        mse: 0,
        rmse: 0,
        r2: 0,
        mae: 0,
      };
      results.featureImportance = {
        x1: 0,
        x2: 0,
      };
      return results;
    }

    // Generate predictions based on model type
    if (modelId.includes("linear-regression")) {
      // Simple linear regression predictions
      const x1Values = data.features.x1 || []
      const x2Values = data.features.x2 || []

      results.predictions = x1Values.map((x1: number, i: number) => {
        const x2 = x2Values[i] || 0
        return 2.1 * x1 + 3.2 * x2 + Math.random() * 0.5
      })

      results.metrics = {
        mse: 2.34,
        rmse: 1.53,
        r2: 0.87,
        mae: 1.21,
      }

      results.featureImportance = {
        x1: 0.65,
        x2: 0.35,
      }
    } else if (modelId.includes("random-forest")) {
      // Classification predictions
      const x1Values = data.features.x1 || []
      const x2Values = data.features.x2 || []

      results.predictions = x1Values.map((x1: number, i: number) => {
        const x2 = x2Values[i] || 0
        return x1 + x2 > 10 ? 1 : 0
      })

      results.probabilities = x1Values.map((x1: number, i: number) => {
        const x2 = x2Values[i] || 0
        const prob = (x1 + x2) / 20
        return Math.min(Math.max(prob, 0), 1)
      })

      results.metrics = {
        accuracy: 0.92,
        precision: 0.89,
        recall: 0.94,
        f1: 0.91,
      }

      results.featureImportance = {
        x1: 0.48,
        x2: 0.52,
      }
    } else if (modelId.includes("differential-evolution") || modelId.includes("evolution-strategy")) {
      // Optimization results
      results.bestSolution = [2.34, 1.56, 3.78, 0.92, 4.21]
      results.bestFitness = 0.0023
      results.convergence = Array.from({ length: 50 }, (_, i) => ({
        iteration: i,
        fitness: 1.0 / (i + 1) + 0.001,
      }))

      results.metrics = {
        evaluations: 5000,
        iterations: 50,
        convergenceRate: 0.95,
        diversityFinal: 0.12,
      }

      results.parameterImportance = {
        mutationRate: 0.72,
        populationSize: 0.45,
        crossoverRate: 0.38,
      }
    }

    return results
  }

  // Helper function to generate synthetic datasets
  const generateSyntheticDataset = (type: string, samples = 100, dimensions = 2) => {
    const id = `synthetic_${type}_${Date.now()}`;
    const name = `Synthetic ${type.charAt(0).toUpperCase() + type.slice(1)} Dataset`;
    
    // Generate synthetic data based on the type
    let data: any = {};
    let features: string[] = [];
    
    // Create feature names
    for (let i = 0; i < dimensions; i++) {
      features.push(`feature_${i + 1}`);
    }
    
    if (type === 'regression') {
      // Regression dataset
      const X = Array.from({ length: samples }, () => 
        Array.from({ length: dimensions }, () => Math.random() * 10 - 5)
      );
      const y = X.map(x => Math.sin(x[0]) + Math.cos(x[1]) + Math.random() * 0.2);
      
      data = {
        id,
        name,
        type: 'regression',
        rows: samples,
        features: features, // Array of feature names
        data: X.map((x, i) => ({ ...Object.fromEntries(x.map((v, j) => [features[j], v])), target: y[i] })),
        target: 'target',
        description: `Synthetic regression dataset with ${samples} samples and ${dimensions} features.`
      };
    } else if (type === 'classification') {
      // Classification dataset
      const X = Array.from({ length: samples }, () => 
        Array.from({ length: dimensions }, () => Math.random() * 10 - 5)
      );
      const y = X.map(x => (Math.sin(x[0]) + Math.cos(x[1]) > 0 ? 1 : 0));
      
      data = {
        id,
        name,
        type: 'classification',
        rows: samples,
        features: features, // Array of feature names
        data: X.map((x, i) => ({ ...Object.fromEntries(x.map((v, j) => [features[j], v])), target: y[i] })),
        target: 'target',
        classes: [0, 1],
        description: `Synthetic classification dataset with ${samples} samples and ${dimensions} features.`
      };
    } else if (type === 'timeseries') {
      // Time series dataset
      const timepoints = Array.from({ length: samples }, (_, i) => i);
      const values = timepoints.map(t => Math.sin(t / 10) + Math.random() * 0.5);
      
      data = {
        id,
        name,
        type: 'timeseries',
        rows: samples,
        features: ['time', 'value'],
        data: timepoints.map((t, i) => ({ time: t, value: values[i] })),
        timeColumn: 'time',
        valueColumn: 'value',
        description: `Synthetic time series dataset with ${samples} timepoints.`
      };
    } else if (type === 'optimization') {
      // Optimization problem dataset
      const dimensions = 10;
      features = Array.from({ length: dimensions }, (_, i) => `x${i + 1}`);
      
      data = {
        id,
        name,
        type: 'optimization',
        rows: 1, // Just configuration for optimization
        features: features, // Array of feature names
        bounds: Array(dimensions).fill([-5, 5]),
        objective: 'minimize',
        function: 'rastrigin',
        description: `Synthetic optimization problem with ${dimensions} dimensions.`
      };
    }
    
    return data;
  };

  // Function to run benchmarks on multiple models
  const runBenchmarks = async () => {
    setIsExecuting(true);
    setExecutionProgress(10);
    setNotification({
      type: "info",
      message: "Running benchmarks on all optimization algorithms. This may take a moment...",
    });

    // Inside the runBenchmarks function, before the try block
    setIsBenchmarkMode(true);

    try {
      // Create synthetic datasets for benchmarking
      const regressionData = generateSyntheticDataset("regression", 200, 5);
      const classificationData = generateSyntheticDataset("classification", 200, 5);

      setExecutionProgress(20);

      // Create optimization benchmarks
      const benchmarkFunctions = [
        { id: "sphere", name: "Sphere Function", type: "benchmark" },
        { id: "rastrigin", name: "Rastrigin Function", type: "benchmark" },
        { id: "rosenbrock", name: "Rosenbrock Function", type: "benchmark" },
        { id: "ackley", name: "Ackley Function", type: "benchmark" },
        { id: "griewank", name: "Griewank Function", type: "benchmark" },
      ];

      setExecutionProgress(30);

      // Run all models on appropriate datasets
      const benchmarkResults: any = {
        regression: {},
        classification: {},
        optimization: {}
      };

      // Benchmark regression models
      for (const model of models.filter((m) => m.id === "linear-regression")) {
        await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate processing time
        benchmarkResults.regression[model.id] = generateModelResults(model.id, regressionData);
      }

      setExecutionProgress(40);

      // Benchmark classification models
      for (const model of models.filter((m) => m.id === "logistic-regression")) {
        await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate processing time
        benchmarkResults.classification[model.id] = generateModelResults(model.id, classificationData);
      }

      setExecutionProgress(50);

      // Benchmark optimization models on each benchmark function
      const optimizationResults: any = {};
      const optimizationModels = models.filter((m) => m.type === "optimization");
      
      for (const benchmarkFunc of benchmarkFunctions) {
        const benchmarkData = generateSyntheticDataset("optimization", 100, 10);
        optimizationResults[benchmarkFunc.id] = {};
        
        for (const model of optimizationModels) {
          await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate processing time
          
          // Special handling for SATZilla meta-optimizer
          if (model.id === "surrogate-optimizer") {
            optimizationResults[benchmarkFunc.id][model.id] = generateSATZillaResults(
              model.id, 
              benchmarkData, 
              optimizationModels.filter(m => m.id !== "surrogate-optimizer").map(m => m.id),
              benchmarkFunc.id
            );
          } else {
            const result = generateModelResults(model.id, benchmarkData);
            
            optimizationResults[benchmarkFunc.id][model.id] = {
              modelId: model.id,
              modelName: model.name,
              benchmarkId: benchmarkFunc.id,
              benchmarkName: benchmarkFunc.name,
              metrics: result.metrics || {},
              executionTime: result.executionTime || 0,
              convergence: result.convergence || [],
            };
          }
        }
      }

      setExecutionProgress(80);

      // Process benchmark results for visualization
      const processedResults = {
        metrics: {
          regression: Object.entries(benchmarkResults.regression).map(([id, result]: [string, any]) => ({
            name: models.find((m) => m.id === id)?.name || id,
            accuracy: result.metrics?.accuracy || 0,
            r2: result.metrics?.r2 || 0,
            executionTime: result.executionTime || 0,
          })),
          classification: Object.entries(benchmarkResults.classification).map(([id, result]: [string, any]) => ({
            name: models.find((m) => m.id === id)?.name || id,
            accuracy: result.metrics?.accuracy || 0,
            f1: result.metrics?.f1 || 0,
            executionTime: result.executionTime || 0,
          })),
          optimization: Object.keys(optimizationResults).map(benchmarkId => ({
            name: benchmarkFunctions.find(f => f.id === benchmarkId)?.name || benchmarkId,
            algorithms: Object.entries(optimizationResults[benchmarkId]).map(([algId, result]: [string, any]) => ({
              name: models.find((m) => m.id === algId)?.name || algId,
              bestFitness: result.metrics?.bestFitness || 0,
              executionTime: result.executionTime || 0,
              convergenceSpeed: result.metrics?.convergenceSpeed || 0,
            })),
          }))
        },
        convergence: Object.keys(optimizationResults).reduce((acc: Record<string, any[]>, benchmarkId: string) => {
          acc[benchmarkId] = Object.entries(optimizationResults[benchmarkId]).map(([algId, result]: [string, any]) => ({
            name: models.find((m) => m.id === algId)?.name || algId,
            data: result.convergence || []
          }));
          return acc;
        }, {}),
        raw: {
          regression: benchmarkResults.regression,
          classification: benchmarkResults.classification,
          optimization: optimizationResults
        }
      };

      setBenchmarkResults(processedResults);
      
      // Also set the results in the main results state to ensure proper display
      // Format the results for the BenchmarkComparison component
      const allBenchmarkResults: Record<string, Record<string, any[]>> = {};
      
      // Convert optimizationResults to the format expected by BenchmarkComparison
      Object.entries(optimizationResults).forEach(([benchmarkId, benchmarkData]: [string, any]) => {
        allBenchmarkResults[benchmarkId] = {};
        
        Object.entries(benchmarkData).forEach(([modelId, modelResults]: [string, any]) => {
          // Create array of results (we just have one result per model/benchmark)
          allBenchmarkResults[benchmarkId][modelId] = [{
            bestFitness: modelResults.metrics?.bestFitness || 0,
            executionTime: modelResults.executionTime || 0,
            convergence: modelResults.convergence || []
          }];
        });
      });
      
      // Set the formatted results
      setResults({
        type: "benchmark",
        data: allBenchmarkResults
      });
      
      setActiveTab("benchmarks");
      setExecutionProgress(100);
      setNotification({
        type: "success",
        message: "Benchmark comparison completed successfully!",
      });
    } catch (error) {
      console.error("Benchmark error:", error);
      setNotification({
        type: "error",
        message: "Failed to complete benchmark comparison.",
      });
    } finally {
      setTimeout(() => {
        setIsExecuting(false);
        setExecutionProgress(0);
      }, 500);
    }
  };

  // Generate confusion matrix for classification models
  const generateConfusionMatrix = () => {
    // 2x2 confusion matrix for binary classification
    const matrix = [
      [42, 8], // True Negatives, False Positives
      [5, 45], // False Negatives, True Positives
    ]

    const labels = ["Negative", "Positive"]

    const metrics = {
      accuracy: 0.87,
      precision: 0.85,
      recall: 0.9,
      f1: 0.87,
    }

    return { matrix, labels, metrics }
  }

  // Generate hyperparameter tuning results
  const generateHyperparameterTuningResults = (modelId: string) => {
    let parameterRanges: Record<string, any[]> = {}
    let bestParams: Record<string, any> = {}
    let metricName = "score"

    if (modelId.includes("linear-regression")) {
      parameterRanges = {
        alpha: [0.001, 0.01, 0.1, 1.0],
        fit_intercept: [true, false],
        normalize: [true, false],
        max_iter: [1000, 2000, 5000],
      }

      bestParams = {
        alpha: 0.01,
        fit_intercept: true,
        normalize: true,
        max_iter: 1000,
      }

      metricName = "r2_score"
    } else if (modelId.includes("random-forest")) {
      parameterRanges = {
        n_estimators: [50, 100, 150, 200],
        max_depth: [5, 10, 15, 20, 25],
        min_samples_split: [2, 5, 10],
        min_samples_leaf: [1, 2, 4],
      }

      bestParams = {
        n_estimators: 100,
        max_depth: 15,
        min_samples_split: 2,
        min_samples_leaf: 1,
      }

      metricName = "accuracy"
    } else if (modelId.includes("differential-evolution")) {
      parameterRanges = {
        population_size: [20, 50, 100],
        mutation: [0.5, 0.7, 0.9],
        crossover: [0.5, 0.7, 0.9],
        strategy: ["best1bin", "best2bin", "rand1bin", "rand2bin"],
      }

      bestParams = {
        population_size: 50,
        mutation: 0.7,
        crossover: 0.7,
        strategy: "best1bin",
      }

      metricName = "fitness"
    } else if (modelId.includes("evolution-strategy")) {
      parameterRanges = {
        population_size: [50, 100, 200],
        sigma: [0.5, 1.0, 1.5],
        learning_rate: [0.01, 0.1, 0.3],
        adaptive: [true, false],
      }

      bestParams = {
        population_size: 100,
        sigma: 1.0,
        learning_rate: 0.1,
        adaptive: true,
      }

      metricName = "fitness"
    }

    // Generate random tuning results
    const tuningResults: any[] = []

    // Generate grid of parameter combinations
    const generateParameterCombinations = (
      ranges: Record<string, any[]>,
      current: Record<string, any> = {},
      index = 0,
      keys = Object.keys(ranges),
    ) => {
      if (index === keys.length) {
        const score = Math.random() * 0.2 + 0.8 // Random score between 0.8 and 1.0
        const time = Math.random() * 3 + 0.5 // Random time between 0.5 and 3.5

        tuningResults.push({
          id: tuningResults.length + 1,
          params: { ...current },
          score,
          time,
          rank: 0, // Will be calculated later
        })
        return
      }

      const key = keys[index]
      for (const value of ranges[key]) {
        current[key] = value
        generateParameterCombinations(ranges, { ...current }, index + 1, keys)
      }
    }

    generateParameterCombinations(parameterRanges)

    // Sort by score and assign ranks
    tuningResults.sort((a, b) => b.score - a.score)
    tuningResults.forEach((result, i) => {
      result.rank = i + 1
    })

    return {
      tuningResults,
      parameterRanges,
      bestParams,
      metricName,
      modelType: modelId,
    }
  }

  // Generate SATZilla-style surrogate model results
  const generateSATZillaResults = (
    modelId: string, 
    data: any, 
    baseOptimizerIds: string[],
    benchmarkId: string
  ) => {
    // Analyze problem features
    const problemFeatures = {
      modality: Math.random() > 0.5 ? "Multimodal" : "Unimodal",
      variableScaling: Math.random() > 0.7 ? "Mixed" : "Uniform",
      separability: Math.random() > 0.6 ? "Non-separable" : "Separable",
      basinSize: Math.random() > 0.5 ? "Small" : "Large",
      globalStructure: Math.random() > 0.5 ? "Funnel" : "Random",
      localStructure: Math.random() > 0.6 ? "Rugged" : "Smooth",
      ruggedness: Math.random() * 10,
      dimensionality: benchmarkId.includes("sphere") ? "Low" : "High",
    };
    
    // Generate synthetic ML predictions (confidences) for each algorithm
    let algorithmConfidences = baseOptimizerIds.map(id => {
      // Base confidence value - initially random
      let baseConfidence;
      
      // Assign algorithm-specific confidence values instead of purely random ones
      if (id === 'meta-optimizer') {
        // Meta-optimizer should typically have a higher confidence (it selects the best algorithm)
        baseConfidence = 0.7 + Math.random() * 0.3; // 70-100% confidence
      } else if (id === 'evolution-strategy') {
        // Evolution Strategy often performs well on many problems
        baseConfidence = 0.6 + Math.random() * 0.3; // 60-90% confidence
      } else if (id === 'ant-colony') {
        // Ant Colony works well on certain problems
        baseConfidence = 0.4 + Math.random() * 0.3; // 40-70% confidence
      } else if (id === 'surrogate-optimizer') {
        // Surrogate model usually has good confidence
        baseConfidence = 0.5 + Math.random() * 0.3; // 50-80% confidence
      } else if (id === 'differential-evolution') {
        // Differential Evolution is generally reliable
        baseConfidence = 0.45 + Math.random() * 0.3; // 45-75% confidence
      } else {
        // Other algorithms get more variable confidence
        baseConfidence = 0.1 + Math.random() * 0.4; // 10-50% confidence
      }
      
      return {
        optimizerId: id,
        confidence: baseConfidence,
        predictedPerformance: baseConfidence * 2 // Scale performance with confidence
      };
    });
    
    // Normalize confidences so they sum to 1
    const totalConfidence = algorithmConfidences.reduce((sum, alg) => sum + alg.confidence, 0);
    algorithmConfidences = algorithmConfidences.map(alg => ({
      ...alg,
      confidence: alg.confidence / totalConfidence
    }));
    
    // Sort by confidence (higher is better)
    algorithmConfidences.sort((a, b) => b.confidence - a.confidence);
    
    // SATZilla selects the algorithm with highest confidence
    const selectedAlgorithm = algorithmConfidences[0];
    
    // Simulate running the selected algorithm
    const baseOptimizerResults = baseOptimizerIds.map(id => {
      const result = generateModelResults(id, data);
      
      // Get the confidence for this optimizer
      const confidence = algorithmConfidences.find(a => a.optimizerId === id)?.confidence || 0;
      
      // Now make actual performance somewhat related to confidence, but with some variation
      // This ensures that high confidence generally corresponds to better actual performance
      // But with some error margin to simulate realistic prediction errors
      // Base performance is inversely related to fitness (lower fitness is better)
      const baseFitness = result.bestFitness || Math.random();
      
      // Create an actual performance value that generally correlates with confidence
      // but has some natural error/variation
      const actualFitness = baseFitness * (1 + (Math.random() * 0.4 - 0.2)); // +/- 20% variation
      
      return {
        optimizerId: id,
        bestFitness: actualFitness,
        executionTime: result.executionTime || Math.random() * 5,
        convergence: result.convergence || []
      };
    });
    
    // Scale predicted and actual performance for visualization
    // (converting from fitness where lower is better to performance where higher is better)
    const maxFitness = Math.max(...baseOptimizerResults.map(r => r.bestFitness)) * 1.2; // Add headroom
    
    // The selected algorithm's actual performance
    const selectedResult = baseOptimizerResults.find(
      r => r.optimizerId === selectedAlgorithm.optimizerId
    ) || baseOptimizerResults[0];
    
    // Compute selection metrics
    const averageFitness = baseOptimizerResults.reduce((sum, opt) => sum + opt.bestFitness, 0) / baseOptimizerResults.length;
    const fitnessSavings = averageFitness - selectedResult.bestFitness;
    const timeOverhead = Math.random() * 0.5; // Small overhead for algorithm selection
    
    // Instead of modifying resultData directly, create a new object
    const satzillaResultData = {
      ...resultData,
      mlPredictions: algorithmConfidences,
      problemFeatures,
      predictionQuality,
      predictions: algorithmConfidences,
      selectedOptimizer: algorithmConfidences[0].optimizerId,
      metrics: {
        ...resultData.metrics,
        // Custom properties for SATZilla results
        selectionAccuracy: 0.7 + Math.random() * 0.3,
      } as SATZillaMetrics
    };
    
    return satzillaResultData;
  };

  // Function to export visualization
  const exportVisualization = async () => {
    if (visualizationRef.current && results) {
      const modelName = models.find(m => m.id === selectedModel)?.name || 'model'
      const fileName = generateExportFileName(`${modelName}_visualization`)
      await exportChartAsImage(visualizationRef.current, fileName)
    }
  }

  // Update the runModel function to use the correct functions and variables
  const runModel = async () => {
    if (!selectedDataset || !selectedModel) {
      setNotification({
        type: "error",
        message: "Please select both a dataset and a model"
      });
      return;
    }
    
    setIsOptimizing(true);
    setProgress(0);
    setProgressStatus("Initializing optimization...");
    setExecutionState("running");
    setResults(null);
    
    try {
      // Update progress in stages
      setProgressStatus("Preprocessing data...");
      setProgress(10);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setProgressStatus("Initializing optimization algorithm...");
      setProgress(20);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Simulate optimization process with progress updates
      const totalSteps = 50; // Arbitrary number of steps
      for (let step = 1; step <= totalSteps; step++) {
        setProgressStatus(`Optimizing parameters (iteration ${step}/${totalSteps})...`);
        setProgress(20 + (step / totalSteps) * 60);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setProgressStatus("Finalizing results...");
      setProgress(90);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Generate results using the existing function
      const mockResults = generateModelResults(selectedModel ?? "", {
        // Default parameters for all models
        epochs: 100,
        learningRate: 0.01,
        batchSize: 32,
        // Additional parameters for specific models
        ...((selectedModel ?? "").includes("differential-evolution") ? {
          populationSize: 50,
          F: 0.8,
          CR: 0.9
        } : {}),
        ...((selectedModel ?? "").includes("evolution-strategy") ? {
          populationSize: 100,
          sigma: 1.0,
          adaptSigma: true
        } : {}),
        ...((selectedModel ?? "").includes("ant-colony") ? {
          numAnts: 20,
          alpha: 1.0,
          beta: 2.0,
          evaporationRate: 0.1
        } : {}),
        ...((selectedModel ?? "").includes("grey-wolf") ? {
          numWolves: 30,
          a: 2.0
        } : {}),
        ...((selectedModel ?? "").includes("meta-optimizer") ? {
          candidateOptimizers: ["de", "es", "aco", "gwo"],
          selectionCriteria: "performance",
          adaptationStrategy: "ucb"
        } : {})
      });
      
      // Prepare chart data from mockResults
      const chartData = {
        ...mockResults,
        data: [],
        xAxis: "Iteration",
        yAxis: "Value"
      };
      
      // Generate chart data based on model type
      if (selectedModel?.includes("differential-evolution") || selectedModel?.includes("evolution-strategy")) {
        // Use convergence data for chart
        chartData.data = mockResults.convergence.map((item: any) => ({
          name: "Fitness",
          x: item.iteration,
          y: item.fitness
        }));
      } else if (mockResults.predictions && mockResults.predictions.length > 0) {
        // Use predictions for chart
        chartData.data = mockResults.predictions.map((value: number, index: number) => ({
          name: "Prediction",
          x: index,
          y: value
        }));
      } else {
        // Generate random data if nothing suitable exists
        chartData.data = Array.from({ length: 20 }, (_, i) => ({
          name: "Series 1",
          x: i,
          y: Math.random() * 10
        }));
      }
      
      setResults(chartData);
      
      setProgressStatus("Optimization completed successfully!");
      setProgress(100);
      setNotification({
        type: "success",
        message: "Model execution completed successfully"
      });
    } catch (error) {
      console.error("Error running model:", error);
      setNotification({
        type: "error",
        message: `Error running model: ${error instanceof Error ? error.message : "Unknown error"}`
      });
      setProgressStatus(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      // Keep progress visible for a moment before resetting
      setTimeout(() => {
        setExecutionState("completed");
        setIsOptimizing(false);
      }, 1000);
    }
  };

  // Update the notification type handler for SATZilla prediction
  const handleSaTZillaPrediction = async (datasetId: string, optimizerIds: string[], benchmarkId: string) => {
    try {
      setIsOptimizing(true);
      setExecutionState("running");
      setProgress(0);
      setProgressStatus("Initializing meta-learning analysis...");

      // Determine if this is a benchmark dataset or a real dataset
      const isBenchmarkDataset = datasetId.startsWith("benchmark-");
      
      // Reset results to clear previous data
      setResults(null);
      
      // Simulate loading and analysis steps
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgress(20);
      setProgressStatus("Extracting dataset features...");
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgress(40);
      setProgressStatus("Building meta-model for algorithm selection...");
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      setProgress(60);
      setProgressStatus("Analyzing algorithm performance...");
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgress(80);
      setProgressStatus("Determining best algorithm match...");
      
      // Wait for UI updates
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      let data;
      const datasetObj = datasets.find(d => d.id === datasetId);
      
      if (datasetObj) {
        if (datasetObj.data) {
          // If dataset already has data property, use it
          data = datasetObj.data;
        } else if (isBenchmarkDataset) {
          // Generate synthetic data for benchmark functions
          const syntheticData = generateSyntheticData(datasetId);
          data = syntheticData;
        } else {
          // For real datasets, fetch the dataset details or preview
          try {
            // You might need to adjust how you process the dataset details
            const datasetUrl = `/api/datasets/${datasetId}`;
            const response = await fetch(datasetUrl);
            if (!response.ok) {
              throw new Error(`Failed to fetch dataset: ${response.statusText}`);
            }
            const datasetDetails = await response.json();
            
            // Create a simplified version of the dataset for analysis
            data = {
              id: datasetDetails.id,
              name: datasetDetails.name,
              type: datasetDetails.type,
              features: datasetDetails.features,
              samples: datasetDetails.samples,
              dimensions: datasetDetails.features,
              complexity: datasetDetails.metadata?.complexity || 'medium',
              noise: 0.1,
            };
          } catch (error) {
            console.error("Error fetching dataset details:", error);
            throw new Error("Failed to load dataset details for meta-learning analysis");
          }
        }
        
        // Generate SATZilla results using the data
        const satzillaResults = generateSATZillaResults(
          "satzilla", 
          data, 
          optimizerIds,
          benchmarkId
        );
        
        setResults({
          type: "satzilla",
          data: satzillaResults
        });
        
        setProgressStatus("SATZilla meta-learning analysis complete!");
        setProgress(100);
        
        setNotification({
          type: "success",
          message: `SATZilla prediction complete. ${models.find(m => m.id === satzillaResults.selectedOptimizer)?.name || satzillaResults.selectedOptimizer} recommended for best performance.`
        });
      } else {
        throw new Error("Dataset not found");
      }
    } catch (error) {
      console.error("Error in SATZilla analysis:", error);
      setNotification({
        type: "error",
        message: `Error in meta-learning analysis: ${error instanceof Error ? error.message : "Unknown error"}`
      });
      setProgressStatus(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      // Keep progress visible for a moment before resetting
      setTimeout(() => {
        setExecutionState("completed");
        setIsOptimizing(false);
      }, 1500);
    }
  };

  // Update the runHyperparameterTuning function to work with multiple models
  const runHyperparameterTuning = async () => {
    const modelsToTune = selectedModelsForTuning.length > 0 ? selectedModelsForTuning : selectedModel ? [selectedModel] : [];
    
    if (modelsToTune.length === 0) {
      setNotification({
        type: "error",
        message: "Please select at least one model for hyperparameter tuning"
      });
      return;
    }
    
    setIsOptimizing(true);
    setProgress(0);
    setProgressStatus("Initializing hyperparameter tuning...");
    setExecutionState("running");
    
    try {
      // Update progress in stages
      setProgressStatus("Setting up parameter grid...");
      setProgress(10);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setProgressStatus("Preparing cross-validation splits...");
      setProgress(20);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Generate combined tuning results from all selected models
      const allTuningResults: any = {};
      
      // Simulate grid search process with progress updates for each model
      for (let modelIndex = 0; modelIndex < modelsToTune.length; modelIndex++) {
        const modelId = modelsToTune[modelIndex];
        const modelName = models.find(m => m.id === modelId)?.name || modelId;
        const modelProgressStart = 20 + (modelIndex * 60 / modelsToTune.length);
        const modelProgressEnd = 20 + ((modelIndex + 1) * 60 / modelsToTune.length);
        
        setProgressStatus(`Tuning ${modelName} (${modelIndex + 1}/${modelsToTune.length})...`);
        
        // Simulate steps for this model
        const totalSteps = 10; // Grid search iterations per model
        for (let step = 1; step <= totalSteps; step++) {
          setProgressStatus(`Evaluating ${modelName} parameter combination ${step}/${totalSteps}...`);
          const stepProgress = modelProgressStart + (step / totalSteps) * (modelProgressEnd - modelProgressStart);
          setProgress(stepProgress);
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Generate results for this model
        allTuningResults[modelId] = generateHyperparameterTuningResults(modelId);
      }
      
      setProgressStatus("Analyzing results...");
      setProgress(90);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Set results with hyperparameter-tuning type
      setResults({
        type: "hyperparameter-tuning",
        data: {
          // Use the first model's results as the primary data for visualization
          ...allTuningResults[modelsToTune[0]],
          // Include all results for comparison
          allResults: allTuningResults,
          // Keep track of all models that were tuned
          modelIds: modelsToTune,
          activeModelId: modelsToTune[0] // Track which model's results are currently being viewed
        }
      });
      
      setActiveTab("hyperparameter");
      setProgressStatus("Hyperparameter tuning completed!");
      setProgress(100);
      setNotification({
        type: "success",
        message: `Completed hyperparameter tuning for ${modelsToTune.length} model${modelsToTune.length > 1 ? 's' : ''}`
      });
    } catch (error) {
      console.error("Error running hyperparameter tuning:", error);
      setNotification({
        type: "error",
        message: `Error running hyperparameter tuning: ${error instanceof Error ? error.message : "Unknown error"}`
      });
      setProgressStatus(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      // Keep progress visible for a moment before resetting
      setTimeout(() => {
        setExecutionState("completed");
        setIsOptimizing(false);
      }, 1000);
    }
  };

  // Function to simulate SATZilla algorithm selection for benchmarking
  const simulateSATZillaSelection = (benchmarkId: string) => {
    // Standard optimization algorithms to compare
    const optimizers = models.filter(m => 
      m.type === 'optimization' && 
      m.id !== 'surrogate-optimizer'
    );
    
    if (optimizers.length === 0) {
      return null;
    }
    
    // Create results object
    let resultData = {
      benchmarkId,
      datasetName: "SATZilla Analysis",
      optimizerResults: {},
      metrics: {
        avgBestFitness: 0,
        avgExecutionTime: 0,
        bestOptimizer: '',
        bestFitness: 0,
      }
    };
    
    try {
      // Generate synthetic ML predictions (confidences) for each algorithm
      let algorithmConfidences = optimizers.map(opt => {
        // Base confidence value based on algorithm
        let baseConfidence;
        
        // Assign algorithm-specific confidence values
        if (opt.id === 'meta-optimizer') {
          // Meta-optimizer should typically have a higher confidence
          baseConfidence = 0.7 + Math.random() * 0.3; // 70-100% confidence
        } else if (opt.id === 'evolution-strategy') {
          // Evolution Strategy often performs well on many problems
          baseConfidence = 0.6 + Math.random() * 0.3; // 60-90% confidence
        } else if (opt.id === 'ant-colony') {
          // Ant Colony works well on certain problems
          baseConfidence = 0.4 + Math.random() * 0.3; // 40-70% confidence
        } else if (opt.id === 'differential-evolution') {
          // Differential Evolution is generally reliable
          baseConfidence = 0.45 + Math.random() * 0.3; // 45-75% confidence
        } else {
          // Other algorithms get more variable confidence
          baseConfidence = 0.1 + Math.random() * 0.4; // 10-50% confidence
        }
        
        return {
          optimizerId: opt.id,
          confidence: baseConfidence
        };
      });
      
      // Normalize confidences so they sum to 1
      const totalConfidence = algorithmConfidences.reduce((sum, alg) => sum + alg.confidence, 0);
      algorithmConfidences = algorithmConfidences.map(alg => ({
        ...alg,
        confidence: alg.confidence / totalConfidence
      }));
      
      // Sort by confidence (higher is better)
      algorithmConfidences.sort((a, b) => b.confidence - a.confidence);
      
      // Problem features that SATZilla would analyze
      const problemFeatures = {
        dimensions: Math.floor(Math.random() * 15) + 5,
        modality: Math.random() > 0.5 ? "Multimodal" : "Unimodal",
        variableScaling: Math.random() > 0.7 ? "Mixed" : "Uniform",
        separability: Math.random() > 0.6 ? "Non-separable" : "Separable",
        ruggedness: (Math.random() * 10).toFixed(2),
      };
      
      // Simulate individual optimizer results for this benchmark
      for (const optimizer of optimizers) {
        const bestFitness = Math.random(); // Lower is better
        const executionTime = Math.random() * 5 + 1; // 1-6 seconds
        
        // Store results for this optimizer on this benchmark
        resultData.optimizerResults = resultData.optimizerResults || {};
        (resultData.optimizerResults as any)[optimizer.id] = {
          bestFitness,
          executionTime,
          convergence: generateMockConvergence(50, bestFitness),
        };
      }
      
      // Calculate aggregate metrics
      const fitnessValues = Object.values(resultData.optimizerResults).map((r: any) => r.bestFitness);
      const timeValues = Object.values(resultData.optimizerResults).map((r: any) => r.executionTime);
      
      resultData.metrics.avgBestFitness = fitnessValues.reduce((a, b) => a + b, 0) / fitnessValues.length;
      resultData.metrics.avgExecutionTime = timeValues.reduce((a, b) => a + b, 0) / timeValues.length;
      
      // Find the best optimizer overall
      const bestOptimizerEntries = Object.entries(resultData.optimizerResults);
      bestOptimizerEntries.sort((a: [string, any], b: [string, any]) => a[1].bestFitness - b[1].bestFitness);
      
      if (bestOptimizerEntries.length > 0) {
        resultData.metrics.bestOptimizer = bestOptimizerEntries[0][0];
        resultData.metrics.bestFitness = (bestOptimizerEntries[0][1] as any).bestFitness;
      }
      
      // Generate comparison data between predicted and actual performance
      const predictionQuality = optimizers.map(opt => {
        const predicted = algorithmConfidences.find(c => c.optimizerId === opt.id)?.confidence || 0;
        // Add some noise to actual vs predicted
        const actual = predicted * (0.7 + Math.random() * 0.6);
        
        return {
          algorithm: opt.id,
          predictedPerformance: predicted,
          actualPerformance: actual
        };
      });
      
      // Add SATZilla-specific properties
      resultData = {
        ...resultData,
        mlPredictions: algorithmConfidences,
        problemFeatures,
        predictionQuality,
        predictions: algorithmConfidences,
        selectedOptimizer: algorithmConfidences[0].optimizerId,
        metrics: {
          ...resultData.metrics,
          // Custom properties for SATZilla results
          selectionAccuracy: 0.7 + Math.random() * 0.3,
        } as SATZillaMetrics
      };
      
      return resultData;
    } catch (error) {
      console.error("Error in SATZilla simulation:", error);
      return null;
    }
  };

  // Add the missing generateMockConvergence function
  const generateMockConvergence = (iterations: number, finalValue: number) => {
    const values = [];
    let currentValue = finalValue * 5; // Start higher than the final value
    
    for (let i = 0; i < iterations; i++) {
      // Exponential decay towards final value
      currentValue = currentValue * 0.95 + finalValue * 0.05;
      values.push({
        iteration: i,
        value: currentValue + Math.random() * 0.2 * currentValue // Add some noise
      });
    }
    
    return values;
  };

  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Data Visualization Hub</h1>
        {notification && (
          <Alert variant={notification.type === "error" ? "destructive" : "default"}>
            {notification.type === "error" && <AlertCircle className="h-4 w-4" />}
            {notification.type === "success" && <CheckCircle2 className="h-4 w-4" />}
            {notification.type === "info" && <Info className="h-4 w-4" />}
            <AlertTitle>
              {notification.type === "error"
                ? "Error"
                : notification.type === "success"
                ? "Success"
                : "Information"}
            </AlertTitle>
            <AlertDescription>{notification.message}</AlertDescription>
          </Alert>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-8">
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="hyperparameter">Hyperparameter Tuning</TabsTrigger>
          <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
          <TabsTrigger value="metalearning">Meta-Learning</TabsTrigger>
          <TabsTrigger value="framework">Framework Runner</TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="space-y-8">
          {/* Visualization tab content */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* First column - Dataset Selection */}
            <Card>
              <CardHeader>
                <CardTitle>Dataset Selection</CardTitle>
                <CardDescription>Choose a dataset to visualize</CardDescription>
              </CardHeader>
              <CardContent>
                {datasets.length > 0 ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 gap-2">
                      {datasets.map((dataset) => (
                        <div
                          key={dataset.id}
                          className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                            selectedDataset === dataset.id
                              ? "border-primary bg-primary/5"
                              : "border-border hover:border-primary/50"
                          }`}
                          onClick={() => handleDatasetSelect(dataset.id)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{dataset.name}</span>
                            <Badge variant="outline">{dataset.type}</Badge>
                          </div>
                        </div>
                      ))}
                    </div>

                    {currentData && (
                      <div className="pt-2 border-t">
                        <h4 className="text-sm font-medium mb-1">Dataset Info</h4>
                        <ul className="text-xs text-muted-foreground space-y-1">
                          <li>Type: {currentData.type}</li>
                          {currentData.features && (
                            <li>Features: {currentData.features.length}</li>
                          )}
                          {currentData.data && (
                            <li>Samples: {currentData.data.length}</li>
                          )}
                        </ul>
                      </div>
                    )}

                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => {
                        // Reset data selection
                        setSelectedDataset(undefined);
                        setCurrentData(null);
                      }}
                      disabled={!selectedDataset}
                    >
                      Clear Selection
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-40">
                    <Spinner className="mr-2" />
                    <span>Loading datasets...</span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Second column - Model Selection and Execution */}
            <Card>
              <CardHeader>
                <CardTitle>Model Selection</CardTitle>
                <CardDescription>Choose a model to run</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {models.length > 0 ? (
                    <>
                      <div className="grid grid-cols-1 gap-2">
                        {models
                          .filter((model) => !currentData || !currentData.type || model.type === currentData.type || (
                            currentData.type === "benchmark" && model.type === "optimization"
                          ))
                          .map((model) => (
                            <div
                              key={model.id}
                              className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                                selectedModel === model.id
                                  ? "border-primary bg-primary/5"
                                  : "border-border hover:border-primary/50"
                              }`}
                              onClick={() => setSelectedModel(model.id)}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-medium">{model.name}</span>
                                <Badge variant="outline">{model.type}</Badge>
                              </div>
                            </div>
                          ))}
                      </div>

                      <Button
                        className="w-full"
                        disabled={!selectedModel || !selectedDataset || isOptimizing}
                        onClick={runModel}
                      >
                        {isOptimizing ? (
                          <div className="flex items-center">
                            <Spinner className="mr-2" />
                            <span>Running...</span>
                          </div>
                        ) : (
                          "Run Model"
                        )}
                      </Button>

                      {progress > 0 && (
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span>Execution Progress</span>
                            <span>{progress}%</span>
                          </div>
                          <Progress value={progress} className="h-2" />
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-40">
                      <Spinner className="mr-2" />
                      <span>Loading models...</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Third column - Visualization Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Visualization Settings</CardTitle>
                <CardDescription>Configure visualization options</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="chart-type">Chart Type</Label>
                    <Select 
                      defaultValue="line" 
                      onValueChange={(value) => {
                        if (value === "line" || value === "bar" || value === "scatter") {
                          setVisualizationConfig({
                            ...visualizationConfig,
                            type: value
                          });
                        }
                      }}
                    >
                      <SelectTrigger id="chart-type">
                        <SelectValue placeholder="Select chart type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="line">Line Chart</SelectItem>
                        <SelectItem value="bar">Bar Chart</SelectItem>
                        <SelectItem value="scatter">Scatter Plot</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {currentData && currentData.features && (
                    <div className="space-y-2">
                      <Label>Feature Selection</Label>
                      <div className="grid grid-cols-2 gap-2">
                        {currentData.features.slice(0, 6).map((feature: string) => (
                          <div key={feature} className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              id={`feature-${feature}`}
                              checked={visualizationConfig.features.includes(feature)}
                              onChange={(e) => {
                                const newFeatures = e.target.checked
                                  ? [...visualizationConfig.features, feature]
                                  : visualizationConfig.features.filter((f) => f !== feature);
                                
                                setVisualizationConfig({
                                  ...visualizationConfig,
                                  features: newFeatures
                                });
                              }}
                              className="h-4 w-4 rounded border-gray-300"
                            />
                            <Label htmlFor={`feature-${feature}`} className="text-sm">
                              {feature}
                            </Label>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center space-x-2 pt-2">
                    <Switch
                      id="confidence-intervals"
                      checked={visualizationConfig.showConfidenceIntervals}
                      onCheckedChange={(checked) => {
                        setVisualizationConfig({
                          ...visualizationConfig,
                          showConfidenceIntervals: checked
                        });
                      }}
                    />
                    <Label htmlFor="confidence-intervals">Show Confidence Intervals</Label>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Visualization Results */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Visualization Results</CardTitle>
                <CardDescription>
                  {results ? "Displaying results from model execution" : "Run a model to see results"}
                </CardDescription>
              </div>
              {results && (
                <Button variant="outline" size="sm" onClick={exportVisualization}>
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              )}
            </CardHeader>
            <CardContent>
              <div ref={visualizationRef} className="h-[400px]">
                {results ? (
                  <>
                    {visualizationConfig.type === "line" && (
                      <LineChart 
                        data={results.data || []} 
                        xLabel={results.xAxis || "X Axis"}
                        yLabel={results.yAxis || "Y Axis"}
                        showLegend={true}
                      />
                    )}
                    {visualizationConfig.type === "bar" && (
                      <BarChart 
                        data={results.data || []} 
                        xLabel={results.xAxis || "X Axis"}
                        yLabel={results.yAxis || "Y Axis"}
                        showLegend={true}
                      />
                    )}
                    {visualizationConfig.type === "scatter" && (
                      <ScatterChart 
                        data={results.data || []} 
                        xLabel={results.xAxis || "X Axis"}
                        yLabel={results.yAxis || "Y Axis"}
                        showLegend={true}
                      />
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    <div className="text-center">
                      <p>Select a dataset and run a model to see results</p>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Add metrics display */}
              {results && results.metrics && (
                <div className="mt-6 border-t pt-4">
                  <h3 className="text-lg font-medium mb-3">Model Metrics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(results.metrics).map(([key, value]: [string, any]) => (
                      <div key={key} className="bg-muted/50 p-3 rounded-md">
                        <div className="text-xs text-muted-foreground mb-1 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                        <div className="font-medium">{typeof value === 'number' ? value.toFixed(4) : String(value)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hyperparameter" className="space-y-8">
          {/* Hyperparameter tuning tab content */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Hyperparameter Tuning</CardTitle>
              <CardDescription>Optimize model parameters for best performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {models.length > 0 ? (
                  <>
                    <div className="grid grid-cols-1 gap-2">
                      <Label htmlFor="model-selection">Select Models to Tune</Label>
                      <div className="space-y-4">
                        {/* Single model select for backward compatibility */}
                        <Select 
                          value={selectedModel} 
                          onValueChange={setSelectedModel}
                        >
                          <SelectTrigger id="model-selection">
                            <SelectValue placeholder="Select primary model" />
                          </SelectTrigger>
                          <SelectContent>
                            {models.map((model) => (
                              <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        
                        {/* Multi-select models for comparison */}
                        <div className="border rounded-md p-3">
                          <Label className="block mb-2">Additional Models for Comparison (optional)</Label>
                          <div className="grid grid-cols-1 gap-2 max-h-60 overflow-y-auto">
                            {models.map((model) => (
                              <div key={`multi-${model.id}`} className="flex items-center space-x-2">
                                <input
                                  type="checkbox"
                                  id={`model-${model.id}`}
                                  checked={selectedModelsForTuning.includes(model.id)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setSelectedModelsForTuning([...selectedModelsForTuning, model.id]);
                                    } else {
                                      setSelectedModelsForTuning(
                                        selectedModelsForTuning.filter(id => id !== model.id)
                                      );
                                    }
                                  }}
                                  className="h-4 w-4 rounded border-gray-300"
                                />
                                <Label htmlFor={`model-${model.id}`} className="text-sm cursor-pointer">
                                  {model.name}
                                </Label>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    <Button
                      className="w-full"
                      disabled={(!selectedModel && selectedModelsForTuning.length === 0) || isOptimizing}
                      onClick={runHyperparameterTuning}
                    >
                      {isOptimizing ? (
                        <div className="flex items-center">
                          <Spinner className="mr-2" />
                          <span>Tuning Parameters...</span>
                        </div>
                      ) : (
                        "Run Hyperparameter Tuning"
                      )}
                    </Button>

                    {progress > 0 && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span>{progressStatus}</span>
                          <span>{progress}%</span>
                        </div>
                        <Progress value={progress} className="h-2" />
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center h-40">
                    <Spinner className="mr-2" />
                    <span>Loading models...</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
          
          {/* Model selector when multiple models have been tuned */}
          {results?.type === "hyperparameter-tuning" && results.data.modelIds && results.data.modelIds.length > 1 && (
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="space-y-2">
                  <Label htmlFor="active-model">View Results For:</Label>
                  <Select 
                    value={results.data.activeModelId || results.data.modelIds[0]} 
                    onValueChange={(modelId) => {
                      // Update the active model ID to show different results
                      setResults({
                        type: "hyperparameter-tuning",
                        data: {
                          ...results.data.allResults[modelId],
                          allResults: results.data.allResults,
                          modelIds: results.data.modelIds,
                          activeModelId: modelId
                        }
                      });
                    }}
                  >
                    <SelectTrigger id="active-model">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {results.data.modelIds.map((modelId: string) => (
                        <SelectItem key={modelId} value={modelId}>
                          {models.find(m => m.id === modelId)?.name || modelId}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* HyperparameterTuning component for visualization */}
          <HyperparameterTuning 
            tuningResults={results?.type === "hyperparameter-tuning" ? results.data.tuningResults : []}
            parameterRanges={results?.type === "hyperparameter-tuning" ? results.data.parameterRanges : {}}
            bestParams={results?.type === "hyperparameter-tuning" ? results.data.bestParams : {}}
            metricName={results?.type === "hyperparameter-tuning" ? results.data.metricName || "Score" : "Score"}
            modelType={results?.type === "hyperparameter-tuning" && results.data.activeModelId 
              ? (models.find(m => m.id === results.data.activeModelId)?.name || results.data.activeModelId) 
              : (selectedModel || "Unknown")}
          />
        </TabsContent>

        <TabsContent value="benchmarks" className="space-y-8">
          {/* Benchmarks tab content */}
          <BenchmarkComparison 
            results={results?.type === "benchmark" ? results.data : null}
            benchmarkFunction={selectedBenchmark || "rastrigin"}
            metricName="fitness"
            onRunBenchmark={(benchmarkResults) => {
              setResults({
                type: "benchmark",
                data: benchmarkResults
              });
            }}
          />
        </TabsContent>

        <TabsContent value="metalearning" className="space-y-8">
          {/* Meta-learning (SATZilla) tab content */}
          <SATZillaPrediction 
            datasets={datasets}
            models={models}
            onExecute={handleSaTZillaPrediction}
            results={results?.type === "satzilla" ? results.data : null}
            predictions={results?.type === "satzilla" ? results.data?.predictions || [] : []}
            predictionQuality={results?.type === "satzilla" ? results.data?.predictionQuality || [] : []}
            selectedOptimizer={results?.type === "satzilla" ? results.data?.selectedOptimizer || "" : ""}
            algorithmNames={{
              "differential-evolution": "Differential Evolution",
              "evolution-strategy": "Evolution Strategy (ES)",
              "ant-colony": "Ant Colony Optimization",
              "grey-wolf": "Grey Wolf Optimizer",
              "meta-optimizer": "Meta-Optimizer",
              "surrogate-optimizer": "Surrogate ML Optimizer"
            }}
          />
        </TabsContent>

        <TabsContent value="framework" className="space-y-8">
          {/* Framework Runner tab content */}
          <FrameworkRunner />
        </TabsContent>
      </Tabs>

      <div className="mt-12 mb-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Paper Visualizations</h2>
            <p className="text-muted-foreground mt-1">
              Evolutionary Computation for Migraine Prediction and Optimization
            </p>
          </div>
          <Button asChild variant="default" className="mt-4 md:mt-0">
            <Link href="/visualization">
              Explore All Visualizations <ExternalLink className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
          {isLoadingVisualizations ? (
            <>
              <Card className="h-[250px] flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p className="mt-4 text-sm text-muted-foreground">Loading visualizations...</p>
                </div>
              </Card>
              <Card className="h-[250px] flex items-center justify-center">
                <div className="flex flex-col items-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                  <p className="mt-4 text-sm text-muted-foreground">Loading visualizations...</p>
                </div>
              </Card>
            </>
          ) : paperVisualizations.length > 0 ? (
            paperVisualizations.map((viz) => (
              <Card key={viz.id} className="overflow-hidden h-full flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle>{viz.title}</CardTitle>
                  <CardDescription>{viz.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <div className="relative aspect-[16/9] w-full bg-muted rounded-md overflow-hidden">
                    <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                      <p>Visualization placeholder - would render actual image from {viz.image}</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="pt-2">
                  <Button variant="outline" size="sm" asChild className="ml-auto">
                    <Link href={`/visualization?tab=${viz.id === 'performance_comparison' ? 'performance' : 'selection'}`}>
                      View Details
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))
          ) : (
            <div className="col-span-2">
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>No visualizations found</AlertTitle>
                <AlertDescription>
                  Generate benchmark results to create visualizations for your paper.
                </AlertDescription>
              </Alert>
            </div>
          )}
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Paper Integration</CardTitle>
            <CardDescription>Visualizations for your IEEE paper on Evolutionary Computation</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              These visualizations provide evidence for key sections in your paper including algorithm performance, selection mechanisms, and convergence behavior.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded p-3">
                <h3 className="font-medium text-sm mb-1">GA vs DE Comparison</h3>
                <p className="text-xs text-muted-foreground">Supports Section 3-4 findings on algorithm strengths and weaknesses</p>
              </div>
              <div className="border rounded p-3">
                <h3 className="font-medium text-sm mb-1">SATzilla Selection</h3>
                <p className="text-xs text-muted-foreground">Illustrates Section 6.2 on hybrid meta-optimizer approaches</p>
              </div>
              <div className="border rounded p-3">
                <h3 className="font-medium text-sm mb-1">Convergence Analysis</h3>
                <p className="text-xs text-muted-foreground">Validates real-time adaptation capabilities described in Section 5.2</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

