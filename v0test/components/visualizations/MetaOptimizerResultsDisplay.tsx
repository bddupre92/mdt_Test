"use client";

import { useState, useEffect } from "react";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Download, ExternalLink } from "lucide-react";

interface MetaOptimizerResultsDisplayProps {
  results: any;
}

export function MetaOptimizerResultsDisplay({ results }: MetaOptimizerResultsDisplayProps) {
  const [activeTab, setActiveTab] = useState("visualizations");
  const [imageUrls, setImageUrls] = useState<{[key: string]: string[]}>({
    performance: [],
    convergence: [],
    radar: [],
    features: [],
    drift: [],
    dynamic: [],
    other: []
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Enhanced helper function to check directories for visualization files
  const checkDirectoryForFiles = async (directory: string) => {
    try {
      setIsLoading(true);
      console.log(`Checking directory: ${directory}`);
      
      // First try to list the directory to see if it has subdirectories
      const response = await fetch(`/api/file?path=${encodeURIComponent(directory)}&list=true`);
      
      if (!response.ok) {
        console.warn(`Failed to list directory ${directory}: ${response.statusText}`);
        return directory;
      }
      
      const data = await response.json();
      
      if (data.files) {
        // Look for timestamp subdirectories (format: YYYYMMDD_HHMMSS)
        const timestampDirs = data.files.filter((file: any) => 
          file.isDirectory && /\d{8}_\d{6}/.test(file.name)
        );
        
        if (timestampDirs.length > 0) {
          // Sort by name to get the latest timestamp
          const latestDir = timestampDirs.sort((a: any, b: any) => b.name.localeCompare(a.name))[0];
          console.log(`Found timestamp directory: ${latestDir.name}`);
          return `${directory}/${latestDir.name}`;
        }
        
        // Look for visualizations subdirectory
        const visualizationsDir = data.files.find((file: any) => 
          file.isDirectory && file.name.toLowerCase() === 'visualizations'
        );
        
        if (visualizationsDir) {
          console.log(`Found visualizations directory: ${visualizationsDir.name}`);
          return `${directory}/${visualizationsDir.name}`;
        }
      }
      
      // If no timestamp directories were found, return the original directory
      return directory;
    } catch (error) {
      console.error("Error checking directory:", error);
      return directory;
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Convert file paths to URLs
    if (results && results.outputPath) {
      const fetchFiles = async () => {
        setIsLoading(true);
        
        try {
          // Try to find the correct visualization directory
          const baseDir = await checkDirectoryForFiles(results.outputPath);
          console.log("Looking for visualizations in:", baseDir);
          
          // Initialize URLs object
          const urls: {[key: string]: string[]} = {
            performance: [],
            convergence: [],
            radar: [],
            features: [],
            drift: [],
            dynamic: [],
            other: []
          };
          
          // Categorize visualization files based on their names
          if (results.visualizationPaths && results.visualizationPaths.length > 0) {
            for (const filePath of results.visualizationPaths) {
              const fileName = filePath.split('/').pop()?.toLowerCase() || '';
              const url = `/api/file?path=${encodeURIComponent(filePath)}`;
              
              if (fileName.includes('performance') || fileName.includes('fitness') || fileName.includes('best_')) {
                urls.performance.push(url);
              } else if (fileName.includes('convergence')) {
                urls.convergence.push(url);
              } else if (fileName.includes('radar') || fileName.includes('chart')) {
                urls.radar.push(url);
              } else if (fileName.includes('feature') || fileName.includes('clustering') || fileName.includes('pca')) {
                urls.features.push(url);
              } else if (fileName.includes('drift')) {
                urls.drift.push(url);
              } else if (fileName.includes('dynamic') || fileName.includes('sudden')) {
                urls.dynamic.push(url);
              } else {
                urls.other.push(url);
              }
            }
          }
          // Alternative approach: search for visualization files in the directory
          else {
            // Try to list the directory and find visualization files
            const response = await fetch(`/api/file?path=${encodeURIComponent(baseDir)}&list=true`);
            
            if (response.ok) {
              const data = await response.json();
              
              if (data.files) {
                // Filter for image files
                const imageFiles = data.files.filter((file: any) => 
                  !file.isDirectory && /\.(png|jpg|jpeg|gif|svg)$/i.test(file.name)
                );
                
                // Categorize them based on filenames
                for (const file of imageFiles) {
                  const fileName = file.name.toLowerCase();
                  const url = `/api/file?path=${encodeURIComponent(`${baseDir}/${file.name}`)}`;
                  
                  if (fileName.includes('performance') || fileName.includes('fitness') || fileName.includes('best_')) {
                    urls.performance.push(url);
                  } else if (fileName.includes('convergence')) {
                    urls.convergence.push(url);
                  } else if (fileName.includes('radar') || fileName.includes('chart')) {
                    urls.radar.push(url);
                  } else if (fileName.includes('feature') || fileName.includes('clustering') || fileName.includes('pca')) {
                    urls.features.push(url);
                  } else if (fileName.includes('drift')) {
                    urls.drift.push(url);
                  } else if (fileName.includes('dynamic') || fileName.includes('sudden')) {
                    urls.dynamic.push(url);
                  } else {
                    urls.other.push(url);
                  }
                }
              }
            }
          }
          
          setImageUrls(urls);
        } catch (error) {
          console.error("Error fetching visualization files:", error);
        } finally {
          setIsLoading(false);
        }
      };
      
      fetchFiles();
    }
  }, [results]);

  const renderImageGroup = (images: string[], emptyMessage: string) => {
    if (images.length === 0) {
      return (
        <div className="bg-muted p-8 rounded-lg text-center">
          <p className="text-muted-foreground">{emptyMessage}</p>
        </div>
      );
    }
    
    return (
      <div className="space-y-6">
        {images.map((url, index) => (
          <div key={index} className="border rounded-lg overflow-hidden">
            <div className="bg-muted p-2 flex justify-between items-center">
              <span className="text-sm font-medium">
                Figure {index + 1}
              </span>
              <Button variant="ghost" size="sm" asChild>
                <a href={url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-1" />
                  Open
                </a>
              </Button>
            </div>
            <div className="p-4 flex justify-center">
              <img 
                src={url} 
                alt={`Meta-Optimizer Visualization ${index + 1}`}
                className="max-w-full h-auto"
                style={{ maxHeight: '400px' }}
                onError={(e) => {
                  // Handle image loading errors
                  console.warn(`Failed to load image: ${url}`);
                  (e.target as HTMLImageElement).src = '/placeholder-image.png';
                  (e.target as HTMLImageElement).alt = 'Image not available';
                }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  if (!results) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Results Available</CardTitle>
          <CardDescription>Run a Meta-Optimizer analysis to see results</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const hasImages = Object.values(imageUrls).some(group => group.length > 0);

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>{results.title || "Meta-Optimizer Results"}</CardTitle>
            <CardDescription>{results.description || "Results from Meta-Optimizer analysis"}</CardDescription>
          </div>
          {results.outputPath && (
            <Button variant="outline" size="sm" asChild>
              <a href={`/api/file?path=${encodeURIComponent(results.outputPath)}&download=true`} download>
                <Download className="h-4 w-4 mr-2" />
                Export
              </a>
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="performance" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="convergence">Convergence</TabsTrigger>
            <TabsTrigger value="radar">Radar Charts</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="drift">Drift Analysis</TabsTrigger>
            <TabsTrigger value="dynamic">Dynamic Optimization</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
            <TabsTrigger value="data">Raw Data</TabsTrigger>
          </TabsList>
          
          <TabsContent value="performance">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.performance, 
              "No performance visualizations available. Try running the baseline comparison with additional algorithms."
            )}
          </TabsContent>
          
          <TabsContent value="convergence">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.convergence, 
              "No convergence plots available. Try running the baseline comparison with the --visualize-convergence flag."
            )}
          </TabsContent>
          
          <TabsContent value="radar">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.radar, 
              "No radar charts available. Try running the baseline comparison with the --create-radar-charts flag."
            )}
          </TabsContent>
          
          <TabsContent value="features">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.features, 
              "No feature importance visualizations available. Try running the SATzilla training with the --visualize-features flag."
            )}
          </TabsContent>
          
          <TabsContent value="drift">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.drift, 
              "No drift analysis visualizations available. Try running the drift detection command."
            )}
          </TabsContent>
          
          <TabsContent value="dynamic">
            {isLoading ? (
              <div className="flex justify-center items-center p-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : renderImageGroup(
              imageUrls.dynamic, 
              "No dynamic optimization visualizations available. Try running the dynamic optimization command."
            )}
          </TabsContent>
          
          <TabsContent value="data">
            <div className="bg-muted rounded-lg p-4 overflow-auto max-h-[500px]">
              <pre className="text-xs">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          </TabsContent>
          
          <TabsContent value="logs">
            <div className="bg-muted rounded-lg p-4 overflow-auto max-h-[500px]">
              <pre className="text-xs whitespace-pre-wrap">
                {results.stdout || "No logs available"}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 