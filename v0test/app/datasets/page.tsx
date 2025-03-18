"use client"

import { DatasetSelector } from "@/components/dataset-selector"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SyntheticDatasetGenerator } from "@/components/synthetic-dataset-generator"
import { Separator } from "@/components/ui/separator"
import { useState } from "react"

export default function DatasetsPage() {
  const [selectedDataset, setSelectedDataset] = useState<string | undefined>(undefined);
  
  return (
    <div className="container mx-auto py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Datasets</h1>
        <p className="text-muted-foreground">
          Explore, upload, generate, and manage your datasets for machine learning pipelines.
        </p>
      </div>
      
      <Tabs defaultValue="explore" className="w-full">
        <TabsList className="grid w-full grid-cols-3 max-w-md mb-4">
          <TabsTrigger value="explore">Explore</TabsTrigger>
          <TabsTrigger value="upload">Upload</TabsTrigger>
          <TabsTrigger value="generate">Generate</TabsTrigger>
        </TabsList>
        
        <TabsContent value="explore" className="space-y-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle>Dataset Explorer</CardTitle>
              <CardDescription>
                Select from available datasets to view details and visualize data distributions.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DatasetSelector 
                onSelectDataset={setSelectedDataset}
                selectedDataset={selectedDataset}
                showSyntheticButton={false} 
                showUploadButton={false}
              />
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Dataset</CardTitle>
              <CardDescription>
                Upload your own datasets in CSV, Excel, or JSON format.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <DatasetSelector 
                onSelectDataset={setSelectedDataset}
                selectedDataset={selectedDataset}
                showSyntheticButton={false}
                initialView="upload"
              />
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="generate" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Synthetic Dataset Generator</CardTitle>
              <CardDescription>
                Create custom synthetic datasets with configurable parameters.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SyntheticDatasetGenerator onDatasetGenerated={setSelectedDataset} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      <Separator className="my-8" />
      
      {/* Dataset Usage Guidelines */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Best Practices</CardTitle>
          <CardDescription>Guidelines for effective dataset management</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="border rounded-md p-4">
              <h3 className="font-medium mb-2">Dataset Preparation</h3>
              <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                <li>Clean your data to remove outliers and errors</li>
                <li>Normalize or standardize numerical features</li>
                <li>Encode categorical variables appropriately</li>
                <li>Split into training, validation, and test sets</li>
              </ul>
            </div>
            
            <div className="border rounded-md p-4">
              <h3 className="font-medium mb-2">Dataset Validation</h3>
              <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                <li>Check for class imbalance in classification tasks</li>
                <li>Verify feature distributions match expectations</li>
                <li>Ensure no data leakage between splits</li>
                <li>Validate temporal consistency for time series</li>
              </ul>
            </div>
            
            <div className="border rounded-md p-4">
              <h3 className="font-medium mb-2">Dataset Storage</h3>
              <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                <li>Use appropriate file formats (CSV, Parquet, Arrow)</li>
                <li>Document dataset properties and sources</li>
                <li>Version control datasets with changes</li>
                <li>Consider data privacy and security requirements</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 