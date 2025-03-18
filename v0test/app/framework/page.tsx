"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { saveAs } from "file-saver";
import html2canvas from "html2canvas";

interface Parameter {
  name: string;
  type: string;
  defaultValue: any;
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

const FrameworkRunner = () => {
  const [activeTab, setActiveTab] = useState("run");
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
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [savedRunResult, setSavedRunResult] = useState<any>(null);
  const [savedRunVisualization, setSavedRunVisualization] = useState<string | null>(null);
  const [savedRunOutput, setSavedRunOutput] = useState<string | null>(null);
  const [savedRunType, setSavedRunType] = useState<string | null>(null);

  // Load modules on initial render
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

    fetchModules();
    fetchSavedRuns();
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

  // Run the function
  const runFunction = async () => {
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
          parameters: paramValues,
          save_result: saveResult,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.result);
        setResultType(data.result_type);
        setOutput(data.output);
        setVisualization(data.visualization);
        
        // Refresh saved runs if result was saved
        if (saveResult && data.saved_id) {
          fetchSavedRuns();
        }
      } else {
        setError(data.message);
        setOutput(data.output);
      }
    } catch (error) {
      console.error("Error running function:", error);
      setError("Failed to run function. Please check if the API server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  // Load saved run
  const loadSavedRun = async (runId: string) => {
    setIsLoading(true);
    setError(null);
    setSavedRunResult(null);
    setSavedRunType(null);
    setSavedRunVisualization(null);
    setSavedRunOutput(null);

    try {
      const response = await fetch(`/api/framework/saved-run/${runId}`);
      const data = await response.json();

      if (data.success) {
        setSavedRunResult(data.result);
        setSavedRunType(data.result_type);
        setSavedRunOutput(data.output);
        setSavedRunVisualization(data.visualization);
      } else {
        setError(data.message);
      }
    } catch (error) {
      console.error("Error loading saved run:", error);
      setError("Failed to load saved run.");
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

  const handleSelectRun = (runId: string) => {
    setSelectedRunId(runId);
    loadSavedRun(runId);
  };

  const renderDataAsTable = (data: any) => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return <p>No data available</p>;
    }

    // Convert to array if it's an object
    const dataArray = Array.isArray(data) ? data : [data];
    
    // Get all possible keys from all objects
    const allKeys = new Set<string>();
    dataArray.forEach((item) => {
      if (typeof item === "object" && item !== null) {
        Object.keys(item).forEach((key) => allKeys.add(key));
      }
    });
    
    const keys = Array.from(allKeys);

    return (
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              {keys.map((key) => (
                <TableHead key={key}>{key}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {dataArray.map((item, index) => (
              <TableRow key={index}>
                {keys.map((key) => (
                  <TableCell key={key}>
                    {item && typeof item === "object" && key in item
                      ? String(item[key])
                      : ""}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  };

  const renderVisualization = (base64Image: string | null) => {
    if (!base64Image) return null;
    return (
      <div className="mt-4">
        <img
          src={`data:image/png;base64,${base64Image}`}
          alt="Visualization"
          className="max-w-full"
        />
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

  const renderResult = (result: any, type: string | null) => {
    if (result === null) return null;
    
    return (
      <div className="mt-4">
        <h3 className="text-lg font-semibold mb-2">Result {type && `(${type})`}</h3>
        <div>
          {Array.isArray(result) ? (
            renderDataAsTable(result)
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

  const saveVisualizationImage = (base64Image: string | null, functionName: string) => {
    if (!base64Image) return;
    
    const byteString = atob(base64Image);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const intArray = new Uint8Array(arrayBuffer);
    
    for (let i = 0; i < byteString.length; i++) {
      intArray[i] = byteString.charCodeAt(i);
    }
    
    const blob = new Blob([arrayBuffer], { type: "image/png" });
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    saveAs(blob, `${functionName}_${timestamp}.png`);
  };

  const saveResultsAsJSON = (result: any, functionName: string) => {
    if (!result) return;
    
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    saveAs(blob, `${functionName}_${timestamp}.json`);
  };

  const renderParameterControl = (name: string, value: any) => {
    if (typeof value === "boolean") {
      return (
        <div className="flex items-center space-x-2" key={name}>
          <Checkbox
            id={name}
            checked={paramValues[name]}
            onCheckedChange={(checked) => handleParamChange(name, checked)}
          />
          <Label htmlFor={name}>{name}</Label>
        </div>
      );
    } else if (typeof value === "number") {
      return (
        <div className="flex flex-col space-y-1" key={name}>
          <Label htmlFor={name}>{name}</Label>
          <Input
            id={name}
            type="number"
            value={paramValues[name]}
            onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
          />
        </div>
      );
    } else {
      return (
        <div className="flex flex-col space-y-1" key={name}>
          <Label htmlFor={name}>{name}</Label>
          <Input
            id={name}
            value={paramValues[name] || ""}
            onChange={(e) => handleParamChange(name, e.target.value)}
          />
        </div>
      );
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Framework Runner</h1>
      
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="run">Run Functions</TabsTrigger>
          <TabsTrigger value="saved">Saved Runs</TabsTrigger>
        </TabsList>
        
        <TabsContent value="run">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Select Function</CardTitle>
                <CardDescription>
                  Choose a module and function to run
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="module">Module</Label>
                    <Select value={selectedModule} onValueChange={setSelectedModule}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select module" />
                      </SelectTrigger>
                      <SelectContent>
                        {modules.map((module) => (
                          <SelectItem key={module} value={module}>
                            {module}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  {selectedModule && (
                    <div>
                      <Label htmlFor="function">Function</Label>
                      <Select value={selectedFunction} onValueChange={setSelectedFunction}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select function" />
                        </SelectTrigger>
                        <SelectContent>
                          {functions.map((func) => (
                            <SelectItem key={func} value={func}>
                              {func}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  
                  {functionInfo && (
                    <div>
                      <h3 className="text-lg font-semibold mb-2">Description</h3>
                      <div className="text-sm text-gray-700 mb-4">
                        {functionInfo.doc || "No description available."}
                      </div>
                      
                      <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                      <div className="space-y-4">
                        {Object.entries(functionInfo.parameters).map(([name, value]) =>
                          renderParameterControl(name, value)
                        )}
                        
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="saveResult"
                            checked={saveResult}
                            onCheckedChange={(checked) => setSaveResult(checked as boolean)}
                          />
                          <Label htmlFor="saveResult">Save result for later reference</Label>
                        </div>
                      </div>
                      
                      <Button 
                        className="mt-4 w-full" 
                        onClick={runFunction}
                        disabled={isLoading}
                      >
                        {isLoading ? <Spinner className="mr-2" /> : null}
                        Run Function
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>
                  Function output and results will appear here
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <Alert variant="destructive" className="mb-4">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
                
                {isLoading && (
                  <div className="flex justify-center items-center py-8">
                    <Spinner className="h-8 w-8" />
                  </div>
                )}
                
                {!isLoading && (
                  <div>
                    {visualization && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="text-lg font-semibold">Visualization</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => saveVisualizationImage(visualization, selectedFunction)}
                          >
                            Download Image
                          </Button>
                        </div>
                        {renderVisualization(visualization)}
                      </div>
                    )}
                    
                    {renderOutput(output)}
                    
                    {result && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="text-lg font-semibold">Result {resultType && `(${resultType})`}</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => saveResultsAsJSON(result, selectedFunction)}
                          >
                            Download JSON
                          </Button>
                        </div>
                        {renderResult(result, resultType)}
                      </div>
                    )}
                    
                    {!visualization && !output && !result && !error && !isLoading && (
                      <div className="text-center py-8 text-gray-500">
                        Select a function and run it to see results here
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="saved">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Saved Runs</CardTitle>
                <CardDescription>
                  View previously saved function runs
                </CardDescription>
              </CardHeader>
              <CardContent>
                {savedRuns.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No saved runs found. Run functions with the "Save result" option enabled.
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Function</TableHead>
                            <TableHead>Module</TableHead>
                            <TableHead>Timestamp</TableHead>
                            <TableHead></TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {savedRuns.map((run) => (
                            <TableRow key={run.id} className={selectedRunId === run.id ? "bg-gray-100" : ""}>
                              <TableCell>{run.function}</TableCell>
                              <TableCell>{run.module}</TableCell>
                              <TableCell>{run.timestamp}</TableCell>
                              <TableCell>
                                <Button 
                                  variant="ghost" 
                                  size="sm"
                                  onClick={() => handleSelectRun(run.id)}
                                >
                                  View
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Run Details</CardTitle>
                <CardDescription>
                  Details of the selected run
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!selectedRunId ? (
                  <div className="text-center py-8 text-gray-500">
                    Select a run to view details
                  </div>
                ) : isLoading ? (
                  <div className="flex justify-center items-center py-8">
                    <Spinner className="h-8 w-8" />
                  </div>
                ) : (
                  <div>
                    {error && (
                      <Alert variant="destructive" className="mb-4">
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    )}
                    
                    {selectedRunId && (
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold mb-2">Run Information</h3>
                        {savedRuns.find(r => r.id === selectedRunId) && (
                          <div className="space-y-2">
                            <div>
                              <span className="font-medium">Function:</span>{" "}
                              {savedRuns.find(r => r.id === selectedRunId)?.function}
                            </div>
                            <div>
                              <span className="font-medium">Module:</span>{" "}
                              {savedRuns.find(r => r.id === selectedRunId)?.module}
                            </div>
                            <div>
                              <span className="font-medium">Timestamp:</span>{" "}
                              {savedRuns.find(r => r.id === selectedRunId)?.timestamp}
                            </div>
                            <div>
                              <span className="font-medium">Parameters:</span>
                              <div className="ml-4 mt-1">
                                {Object.entries(savedRuns.find(r => r.id === selectedRunId)?.parameters || {}).map(([key, value]) => (
                                  <div key={key}>
                                    <span className="font-medium">{key}:</span> {value}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {savedRunVisualization && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="text-lg font-semibold">Visualization</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => saveVisualizationImage(
                              savedRunVisualization, 
                              savedRuns.find(r => r.id === selectedRunId)?.function || "result"
                            )}
                          >
                            Download Image
                          </Button>
                        </div>
                        {renderVisualization(savedRunVisualization)}
                      </div>
                    )}
                    
                    {renderOutput(savedRunOutput)}
                    
                    {savedRunResult && (
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="text-lg font-semibold">Result {savedRunType && `(${savedRunType})`}</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => saveResultsAsJSON(
                              savedRunResult, 
                              savedRuns.find(r => r.id === selectedRunId)?.function || "result"
                            )}
                          >
                            Download JSON
                          </Button>
                        </div>
                        {renderResult(savedRunResult, savedRunType)}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default FrameworkRunner; 