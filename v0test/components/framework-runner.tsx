"use client";

import { useState, useEffect } from "react";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription 
} from "./ui/card";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Spinner } from "./ui/spinner";
import { DatasetSelector } from "./dataset-selector";

interface Dataset {
  id: string;
  name: string;
  type: string;
  description?: string;
  size?: number;
  source?: string;
}

interface FunctionInfo {
  name: string;
  doc: string | null;
  module: string;
  parameters: Record<string, any>;
}

interface SavedRun {
  id: string;
  function: string;
  module: string;
  timestamp: string;
  parameters: Record<string, string>;
}

export function FrameworkRunner() {
  const [modules, setModules] = useState<string[]>([]);
  const [selectedModule, setSelectedModule] = useState<string>("");
  const [functions, setFunctions] = useState<string[]>([]);
  const [selectedFunction, setSelectedFunction] = useState<string>("");
  const [functionInfo, setFunctionInfo] = useState<FunctionInfo | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [resultType, setResultType] = useState<string | null>(null);
  const [visualization, setVisualization] = useState<string | null>(null);
  const [output, setOutput] = useState<string | null>(null);
  const [saveResult, setSaveResult] = useState(true);
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);
  const [activeTab, setActiveTab] = useState("modules");
  
  // Add state for datasets
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [datasetLoaded, setDatasetLoaded] = useState<boolean>(false);

  // Load modules and datasets on initial render
  useEffect(() => {
    const fetchModules = async () => {
      try {
        const response = await fetch("/api/framework/modules");
        const data = await response.json();
        setModules(data);
      } catch (error) {
        console.error("Error fetching modules:", error);
        setError("Failed to load modules. Please check if the API server is running.");
      }
    };

    const loadDatasets = async () => {
      try {
        // In a real application, these would be fetched from an API
        setDatasets([
          { id: "synthetic-1", name: "Synthetic Dataset 1", type: "regression", description: "A general purpose regression dataset" },
          { id: "synthetic-2", name: "Synthetic Dataset 2", type: "classification", description: "A general purpose classification dataset" },
          { id: "synthetic-3", name: "Optimization Benchmark", type: "optimization", description: "Dataset for testing optimization algorithms" },
          { id: "time-series-1", name: "Time Series Dataset", type: "time-series", description: "Sequential data for time series analysis" },
          { id: "benchmark-sphere", name: "Sphere Function", type: "benchmark", description: "Standard optimization test function" },
          { id: "benchmark-rastrigin", name: "Rastrigin Function", type: "benchmark", description: "Multi-modal test function with many local minima" }
        ]);
      } catch (error) {
        console.error("Error loading datasets:", error);
        setError("Failed to load available datasets.");
      }
    };

    fetchModules();
    fetchSavedRuns();
    loadDatasets();
  }, []);

  // Load functions when module changes
  useEffect(() => {
    if (selectedModule) {
      const fetchFunctions = async () => {
        try {
          const response = await fetch(`/api/framework/functions/${selectedModule}`);
          const data = await response.json();
          setFunctions(data);
          setSelectedFunction("");
          setFunctionInfo(null);
        } catch (error) {
          console.error("Error fetching functions:", error);
          setError(`Failed to load functions for module "${selectedModule}".`);
        }
      };

      fetchFunctions();
    } else {
      setFunctions([]);
      setSelectedFunction("");
      setFunctionInfo(null);
    }
  }, [selectedModule]);

  // Load function info when function changes
  useEffect(() => {
    if (selectedModule && selectedFunction) {
      const fetchFunctionInfo = async () => {
        try {
          const response = await fetch(`/api/framework/function-info/${selectedModule}/${selectedFunction}`);
          const data = await response.json();
          setFunctionInfo(data);
          
          // Initialize parameter values
          const initialValues: Record<string, any> = {};
          Object.entries(data.parameters).forEach(([name, defaultValue]) => {
            initialValues[name] = defaultValue;
          });
          setParamValues(initialValues);
        } catch (error) {
          console.error("Error fetching function info:", error);
          setError(`Failed to load details for function "${selectedFunction}".`);
        }
      };

      fetchFunctionInfo();
    }
  }, [selectedModule, selectedFunction]);

  // Fetch saved runs
  const fetchSavedRuns = async () => {
    try {
      const response = await fetch("/api/framework/saved-runs");
      const data = await response.json();
      setSavedRuns(data);
    } catch (error) {
      console.error("Error fetching saved runs:", error);
      setError("Failed to load saved runs.");
    }
  };

  // Handle dataset selection
  const handleDatasetSelect = (datasetId: string) => {
    setSelectedDataset(datasetId);
    setDatasetLoaded(true);
    
    // Clear previous results when changing datasets
    setResult(null);
    setResultType(null);
    setVisualization(null);
    setOutput(null);
  };

  const runFunction = async () => {
    // Prevent execution if no dataset is selected
    if (!selectedDataset) {
      setError("Please select a dataset before running a function.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setResultType(null);
    setVisualization(null);
    setOutput(null);

    try {
      const response = await fetch("/api/framework/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          module: selectedModule,
          function: selectedFunction,
          parameters: {
            ...paramValues,
            dataset_id: selectedDataset // Include the selected dataset
          },
          save_result: saveResult,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.message || "Error executing function");
      }

      setResult(data.result);
      setResultType(data.result_type);
      setOutput(data.output);
      setVisualization(data.visualization);

      // Switch to result tab
      setActiveTab("result");
    } catch (error) {
      console.error("Error executing function:", error);
      setError(`Failed to execute function: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleParamChange = (name: string, value: any) => {
    setParamValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const renderParameterControls = () => {
    if (!functionInfo) return null;
    
    return (
      <div className="space-y-4">
        {Object.entries(functionInfo.parameters).map(([name, value]) => (
          <div key={name} className="flex flex-col space-y-1">
            <Label htmlFor={name}>{name}</Label>
            <Input
              id={name}
              type={typeof value === "number" ? "number" : "text"}
              value={paramValues[name] || ""}
              onChange={(e) => handleParamChange(
                name, 
                typeof value === "number" ? parseFloat(e.target.value) : e.target.value
              )}
            />
          </div>
        ))}
        
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="saveResult"
            checked={saveResult}
            onChange={(e) => setSaveResult(e.target.checked)}
            className="h-4 w-4"
          />
          <Label htmlFor="saveResult">Save result for later reference</Label>
        </div>
      </div>
    );
  };

  const renderOutput = (outputText: string | null) => {
    if (!outputText) return null;
    return (
      <div className="mt-4">
        <h3 className="text-lg font-semibold mb-2">Output</h3>
        <pre className="bg-gray-100 p-4 rounded-md text-sm whitespace-pre-wrap">
          {outputText}
        </pre>
      </div>
    );
  };

  const renderVisualization = (base64Image: string | null) => {
    if (!base64Image) return null;
    
    // Check if the image source already starts with 'data:' scheme
    const imgSrc = base64Image.startsWith('data:') 
      ? base64Image 
      : `data:image/png;base64,${base64Image}`;
      
    return (
      <div className="mt-4">
        <h3 className="text-lg font-semibold mb-2">Visualization</h3>
        <img
          src={imgSrc}
          alt="Visualization"
          className="max-w-full"
        />
      </div>
    );
  };

  const renderResult = (result: any, type: string | null) => {
    if (result === null) return null;
    
    return (
      <div className="mt-4">
        <h3 className="text-lg font-semibold mb-2">Result {type && `(${type})`}</h3>
        <div>
          {Array.isArray(result) ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    {Object.keys(result[0] || {}).map((key) => (
                      <th 
                        key={key}
                        className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {result.map((item, i) => (
                    <tr key={i}>
                      {Object.values(item).map((value: any, j) => (
                        <td 
                          key={j}
                          className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                        >
                          {String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : typeof result === "object" ? (
            <pre className="bg-gray-100 p-4 rounded-md text-sm whitespace-pre-wrap">
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : (
            <p>{String(result)}</p>
          )}
        </div>
      </div>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Framework Runner</CardTitle>
        <CardDescription>Execute and visualize framework functions</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="datasets">Datasets</TabsTrigger>
            <TabsTrigger value="modules" disabled={!datasetLoaded}>Modules</TabsTrigger>
            <TabsTrigger value="functions" disabled={!selectedModule || !datasetLoaded}>Functions</TabsTrigger>
            <TabsTrigger value="parameters" disabled={!selectedFunction || !datasetLoaded}>Parameters</TabsTrigger>
            <TabsTrigger value="result" disabled={!result}>Result</TabsTrigger>
            <TabsTrigger value="saved" disabled={savedRuns.length === 0}>Saved Runs</TabsTrigger>
          </TabsList>

          {/* Dataset Selection Tab */}
          <TabsContent value="datasets">
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium">Select a Dataset</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Choose a dataset to work with before running any functions
                </p>
                {datasets.length > 0 ? (
                  <DatasetSelector
                    datasets={datasets}
                    selectedDataset={selectedDataset}
                    onSelectDataset={handleDatasetSelect}
                  />
                ) : (
                  <div className="flex items-center justify-center h-40 border rounded-md">
                    <p className="text-muted-foreground">Loading available datasets...</p>
                  </div>
                )}
              </div>
              
              <div className="flex justify-end mt-4">
                <Button
                  onClick={() => setActiveTab("modules")}
                  disabled={!selectedDataset}
                >
                  Next: Select Module
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* Modules Tab */}
          <TabsContent value="modules">
            <Card>
              <CardHeader>
                <CardTitle>Select Module and Function</CardTitle>
                <CardDescription>
                  Choose a module and function to run from the framework
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="module">Module</Label>
                    <select
                      id="module"
                      value={selectedModule}
                      onChange={(e) => setSelectedModule(e.target.value)}
                      className="w-full p-2 border rounded"
                    >
                      <option value="">Select a module</option>
                      {modules.map((module) => (
                        <option key={module} value={module}>
                          {module}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {selectedModule && (
                    <div>
                      <Label htmlFor="function">Function</Label>
                      <select
                        id="function"
                        value={selectedFunction}
                        onChange={(e) => setSelectedFunction(e.target.value)}
                        className="w-full p-2 border rounded"
                      >
                        <option value="">Select a function</option>
                        {functions.map((func) => (
                          <option key={func} value={func}>
                            {func}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  
                  {functionInfo && (
                    <div className="space-y-4">
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Description</h3>
                        <p className="text-gray-600">
                          {functionInfo.doc || "No description available."}
                        </p>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                        {renderParameterControls()}
                      </div>
                      
                      <Button
                        onClick={runFunction}
                        disabled={isLoading}
                        className="w-full"
                      >
                        {isLoading ? (
                          <>
                            <Spinner className="mr-2" />
                            Running...
                          </>
                        ) : (
                          "Run Function"
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Functions Tab */}
          <TabsContent value="functions">
            {/* ... existing functions tab content ... */}
          </TabsContent>
          
          {/* Parameters Tab */}
          <TabsContent value="parameters">
            {/* ... existing parameters tab content ... */}
          </TabsContent>
          
          {/* Result Tab */}
          <TabsContent value="result">
            <Card>
              <CardHeader>
                <CardTitle>Function Results</CardTitle>
                <CardDescription>
                  Results of the executed function
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <div className="bg-red-50 text-red-700 p-4 rounded mb-4">
                    <h3 className="text-lg font-semibold mb-1">Error</h3>
                    <p>{error}</p>
                  </div>
                )}
                
                {isLoading ? (
                  <div className="flex justify-center items-center py-12">
                    <Spinner className="h-8 w-8" />
                  </div>
                ) : (
                  <div className="space-y-6">
                    {visualization && renderVisualization(visualization)}
                    {output && renderOutput(output)}
                    {result && renderResult(result, resultType)}
                    
                    {!visualization && !output && !result && !error && (
                      <p className="text-center text-gray-500 py-12">
                        No results to display. Run a function to see results here.
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Saved Runs Tab */}
          <TabsContent value="saved">
            <Card>
              <CardHeader>
                <CardTitle>Saved Runs</CardTitle>
                <CardDescription>
                  View previously saved function runs
                </CardDescription>
              </CardHeader>
              <CardContent>
                {savedRuns.length === 0 ? (
                  <p className="text-center text-gray-500 py-12">
                    No saved runs found. Run functions with the "Save result" option enabled.
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead>
                        <tr>
                          <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Function
                          </th>
                          <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Module
                          </th>
                          <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Timestamp
                          </th>
                          <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {savedRuns.map((run) => (
                          <tr key={run.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {run.function}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {run.module}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {run.timestamp}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              <button
                                className="text-blue-500 hover:text-blue-700"
                                onClick={() => {
                                  // Load saved run
                                  window.open(`/api/framework/saved-run/${run.id}`, '_blank');
                                }}
                              >
                                View
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 