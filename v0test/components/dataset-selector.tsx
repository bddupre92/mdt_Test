"use client"
import { useState, useEffect, ChangeEvent } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Upload, FileUp, FileSymlink, Loader2 } from "lucide-react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchDatasets, Dataset as ApiDataset, uploadDataset, generateSyntheticDataset, SyntheticDatasetParams } from "@/lib/api/datasets"
import { DatasetPreview } from "@/components/dataset-preview";
import { SyntheticDatasetGenerator } from "@/components/synthetic-dataset-generator";

// Use the ApiDataset interface directly
type Dataset = ApiDataset;

export interface DatasetSelectorProps {
  onSelectDataset: (datasetId: string) => void;
  selectedDataset?: string;
  showUploadButton?: boolean;
  showSyntheticButton?: boolean;
  initialView?: 'browse' | 'upload' | 'generate';
}

export function DatasetSelector({
  onSelectDataset,
  selectedDataset,
  showUploadButton = true,
  showSyntheticButton = true,
  initialView = 'browse'
}: DatasetSelectorProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // For the upload dialog
  const [uploadDialogOpen, setUploadDialogOpen] = useState(initialView === 'upload');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMeta, setUploadMeta] = useState({
    name: '',
    description: '',
    category: 'tabular'
  });
  const [isUploading, setIsUploading] = useState(false);
  
  // For the synthetic dataset dialog
  const [syntheticDialogOpen, setSyntheticDialogOpen] = useState(initialView === 'generate');
  const [syntheticParams, setSyntheticParams] = useState<SyntheticDatasetParams>({
    name: '',
    description: '',
    category: 'tabular',
    type: 'regression',
    features: 10,
    samples: 1000,
    noise: 0.1,
    complexity: 'medium',
    missingValues: 0,
    outlierPercentage: 0
  });
  const [isGenerating, setIsGenerating] = useState(false);

  // Function to refresh datasets
  const refreshDatasets = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const apiDatasets = await fetchDatasets();
      setDatasets(apiDatasets);
    } catch (error) {
      console.error("Failed to load datasets:", error);
      setError("Failed to load datasets. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch datasets on component mount
  useEffect(() => {
    refreshDatasets();
  }, []);

  // Refresh datasets when selectedDataset changes
  useEffect(() => {
    if (selectedDataset) {
      refreshDatasets();
    }
  }, [selectedDataset]);

  // Handle file upload
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setUploadFile(e.target.files[0]);
      
      // Auto-populate name from filename if not set
      if (!uploadMeta.name) {
        const fileName = e.target.files[0].name.split('.')[0];
        setUploadMeta(prev => ({ ...prev, name: fileName }));
      }
    }
  };

  // Handle upload submission
  const handleUploadSubmit = async () => {
    if (!uploadFile || !uploadMeta.name) return;
    
    setIsUploading(true);
    
    try {
      const newDataset = await uploadDataset(uploadFile, uploadMeta);
      setDatasets(prevDatasets => [...prevDatasets, newDataset]);
      setUploadDialogOpen(false);
      onSelectDataset(newDataset.id);
      
      // Reset the form
      setUploadFile(null);
      setUploadMeta({
        name: '',
        description: '',
        category: 'tabular'
      });
    } catch (error) {
      console.error("Failed to upload dataset:", error);
      setError("Failed to upload dataset. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  // Handle synthetic dataset generation
  const handleSyntheticSubmit = async () => {
    if (!syntheticParams.name) return;
    
    setIsGenerating(true);
    
    try {
      const newDataset = await generateSyntheticDataset(syntheticParams);
      setDatasets(prevDatasets => [...prevDatasets, newDataset]);
      setSyntheticDialogOpen(false);
      onSelectDataset(newDataset.id);
      
      // Reset the form
      setSyntheticParams({
        name: '',
        description: '',
        category: 'tabular',
        type: 'regression',
        features: 10,
        samples: 1000,
        noise: 0.1,
        complexity: 'medium',
        missingValues: 0,
        outlierPercentage: 0
      });
    } catch (error) {
      console.error("Failed to generate synthetic dataset:", error);
      setError("Failed to generate synthetic dataset. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Loading state 
  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-2">Loading datasets...</span>
      </div>
    );
  }

  // Render error state
  if (error && datasets.length === 0) {
    return (
      <div className="border border-red-200 bg-red-50 p-4 rounded-md">
        <p className="text-red-600">{error}</p>
        <Button 
          variant="outline" 
          className="mt-2"
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </div>
    );
  }

  // Render empty state with upload options
  if (datasets.length === 0) {
    return (
      <div className="space-y-8">
        <div className="border border-dashed rounded-lg p-8 text-center">
          <h3 className="font-medium text-lg mb-2">No Datasets Available</h3>
          <p className="text-muted-foreground mb-6">Upload a dataset or generate a synthetic one to get started.</p>
          
          <div className="flex justify-center gap-4">
            {showUploadButton && (
              <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline">
                    <Upload className="mr-2 h-4 w-4" />
                    Upload Dataset
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Upload Dataset</DialogTitle>
                    <DialogDescription>
                      Upload a CSV, Excel, or JSON file containing your dataset.
                    </DialogDescription>
                  </DialogHeader>
                  
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label htmlFor="dataset-file">File</Label>
                      <div className="border border-dashed rounded-md p-4">
                        <Input 
                          id="dataset-file" 
                          type="file" 
                          accept=".csv,.xlsx,.json" 
                          onChange={handleFileChange}
                        />
                        {uploadFile && (
                          <p className="mt-2 text-sm text-muted-foreground">Selected: {uploadFile.name}</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-name">Name</Label>
                      <Input 
                        id="dataset-name" 
                        value={uploadMeta.name}
                        onChange={(e) => setUploadMeta({...uploadMeta, name: e.target.value})}
                        placeholder="Dataset name"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-description">Description</Label>
                      <Textarea 
                        id="dataset-description"
                        value={uploadMeta.description}
                        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setUploadMeta({...uploadMeta, description: e.target.value})}
                        placeholder="Briefly describe this dataset"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-category">Category</Label>
                      <Select 
                        value={uploadMeta.category}
                        onValueChange={(value) => setUploadMeta({...uploadMeta, category: value})}
                      >
                        <SelectTrigger id="dataset-category">
                          <SelectValue placeholder="Select category" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tabular">Tabular</SelectItem>
                          <SelectItem value="time-series">Time Series</SelectItem>
                          <SelectItem value="spatial">Spatial</SelectItem>
                          <SelectItem value="text">Text</SelectItem>
                          <SelectItem value="image">Image</SelectItem>
                          <SelectItem value="mixed">Mixed</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <DialogFooter>
                    <Button variant="ghost" onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
                    <Button 
                      onClick={handleUploadSubmit} 
                      disabled={!uploadFile || !uploadMeta.name || isUploading}
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <FileUp className="mr-2 h-4 w-4" />
                          Upload
                        </>
                      )}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            )}
            
            {showSyntheticButton && (
              <Dialog open={syntheticDialogOpen} onOpenChange={setSyntheticDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline">
                    <FileSymlink className="mr-2 h-4 w-4" />
                    Generate Synthetic
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>Generate Synthetic Dataset</DialogTitle>
                    <DialogDescription>
                      Create a synthetic dataset with customizable parameters.
                    </DialogDescription>
                  </DialogHeader>
                  
                  <SyntheticDatasetGenerator 
                    onDatasetGenerated={(datasetId) => {
                      setSyntheticDialogOpen(false);
                      onSelectDataset(datasetId);
                    }} 
                  />
                </DialogContent>
              </Dialog>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Main view with datasets
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium">Available Datasets</h3>
          <div className="flex gap-2">
            {showUploadButton && (
              <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Upload className="mr-2 h-4 w-4" />
                    Upload
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Upload Dataset</DialogTitle>
                    <DialogDescription>
                      Upload a CSV, Excel, or JSON file containing your dataset.
                    </DialogDescription>
                  </DialogHeader>
                  
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label htmlFor="dataset-file">File</Label>
                      <div className="border border-dashed rounded-md p-4">
                        <Input 
                          id="dataset-file" 
                          type="file" 
                          accept=".csv,.xlsx,.json" 
                          onChange={handleFileChange}
                        />
                        {uploadFile && (
                          <p className="mt-2 text-sm text-muted-foreground">Selected: {uploadFile.name}</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-name">Name</Label>
                      <Input 
                        id="dataset-name" 
                        value={uploadMeta.name}
                        onChange={(e) => setUploadMeta({...uploadMeta, name: e.target.value})}
                        placeholder="Dataset name"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-description">Description</Label>
                      <Textarea 
                        id="dataset-description"
                        value={uploadMeta.description}
                        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setUploadMeta({...uploadMeta, description: e.target.value})}
                        placeholder="Briefly describe this dataset"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="dataset-category">Category</Label>
                      <Select 
                        value={uploadMeta.category}
                        onValueChange={(value) => setUploadMeta({...uploadMeta, category: value})}
                      >
                        <SelectTrigger id="dataset-category">
                          <SelectValue placeholder="Select category" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tabular">Tabular</SelectItem>
                          <SelectItem value="time-series">Time Series</SelectItem>
                          <SelectItem value="spatial">Spatial</SelectItem>
                          <SelectItem value="text">Text</SelectItem>
                          <SelectItem value="image">Image</SelectItem>
                          <SelectItem value="mixed">Mixed</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <DialogFooter>
                    <Button variant="ghost" onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
                    <Button 
                      onClick={handleUploadSubmit} 
                      disabled={!uploadFile || !uploadMeta.name || isUploading}
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <FileUp className="mr-2 h-4 w-4" />
                          Upload
                        </>
                      )}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            )}
            
            {showSyntheticButton && (
              <Dialog open={syntheticDialogOpen} onOpenChange={setSyntheticDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <FileSymlink className="mr-2 h-4 w-4" />
                    Generate
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>Generate Synthetic Dataset</DialogTitle>
                    <DialogDescription>
                      Create a synthetic dataset with customizable parameters for testing algorithms.
                    </DialogDescription>
                  </DialogHeader>
                  
                  <SyntheticDatasetGenerator 
                    onDatasetGenerated={(datasetId) => {
                      setSyntheticDialogOpen(false);
                      onSelectDataset(datasetId);
                    }} 
                  />
                </DialogContent>
              </Dialog>
            )}
          </div>
        </div>
        
        {/* Dataset selection section */}
        <RadioGroup value={selectedDataset} onValueChange={onSelectDataset}>
          <div className="grid gap-2 max-h-52 overflow-y-auto pr-2">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`border rounded-md p-3 flex items-center justify-between ${
                  selectedDataset === dataset.id ? 'border-blue-500 bg-blue-50' : ''
                }`}
              >
                <div className="flex items-center gap-3">
                  <RadioGroupItem value={dataset.id} id={dataset.id} />
                  <div>
                    <Label htmlFor={dataset.id} className="font-medium">
                      {dataset.name}
                    </Label>
                    <p className="text-sm text-muted-foreground line-clamp-1">
                      {dataset.description}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {dataset.type}
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    {dataset.samples.toLocaleString()} rows
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </RadioGroup>
      </div>
      
      {/* Dataset Preview Section */}
      {selectedDataset && (
        <div className="mt-8 pt-6 border-t">
          <DatasetPreview datasetId={selectedDataset} />
        </div>
      )}
    </div>
  );
}

