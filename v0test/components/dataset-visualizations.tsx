import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchDatasetStatistics, DatasetStatistics } from "@/lib/api/datasets";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

// Mock chart components - in a real application, you'd use a chart library like Recharts or Chart.js
function BarChart({ data }: { data: { name: string; value: number }[] }) {
  const max = Math.max(...data.map(d => d.value));
  
  return (
    <div className="h-64 flex items-end gap-1">
      {data.map((d, i) => (
        <div key={i} className="flex flex-col items-center gap-1 flex-1">
          <div
            className="w-full bg-blue-500 rounded-t"
            style={{ height: `${(d.value / max) * 100}%` }}
          ></div>
          <span className="text-xs text-muted-foreground truncate w-full text-center">
            {d.name}
          </span>
        </div>
      ))}
    </div>
  );
}

function ScatterPlot({ data }: { data: { x: number; y: number; label?: string }[] }) {
  return (
    <div className="h-64 w-full border rounded-md relative">
      {data.map((point, i) => (
        <div
          key={i}
          className="absolute h-2 w-2 rounded-full bg-blue-500"
          style={{
            left: `${point.x}%`,
            bottom: `${point.y}%`,
          }}
          title={point.label}
        ></div>
      ))}
    </div>
  );
}

function Histogram({ data }: { data: { bins: number[]; counts: number[] } }) {
  const max = Math.max(...data.counts);
  
  return (
    <div className="h-64 flex items-end gap-1">
      {data.counts.map((count, i) => (
        <div key={i} className="flex flex-col items-center gap-1 flex-1">
          <div
            className="w-full bg-blue-500 rounded-t"
            style={{ height: `${(count / max) * 100}%` }}
          ></div>
          <span className="text-xs text-muted-foreground truncate w-full text-center">
            {data.bins[i].toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  );
}

function CorrelationMatrix({ data }: { data: Record<string, Record<string, number>> }) {
  const features = Object.keys(data);
  
  return (
    <div className="overflow-x-auto">
      <div className="min-w-max">
        <div className="grid grid-cols-[auto_repeat(auto,minmax(40px,1fr))]">
          <div className="h-10"></div>
          {features.map(feature => (
            <div 
              key={feature} 
              className="h-10 flex items-center justify-center font-medium text-xs truncate px-1"
              style={{ transform: 'rotate(-45deg)', transformOrigin: 'center' }}
            >
              {feature}
            </div>
          ))}
          
          {features.map(row => (
            <>
              <div key={`${row}-label`} className="flex items-center h-10 font-medium text-xs px-2 truncate">
                {row}
              </div>
              {features.map(col => {
                const correlation = row === col ? 1 : (data[row]?.[col] || 0);
                const color = correlation > 0
                  ? `rgba(0, 0, 255, ${Math.abs(correlation)})`
                  : `rgba(255, 0, 0, ${Math.abs(correlation)})`;
                
                return (
                  <div
                    key={`${row}-${col}`}
                    className="h-10 flex items-center justify-center text-xs"
                    style={{ 
                      backgroundColor: color,
                      color: Math.abs(correlation) > 0.5 ? 'white' : 'black'
                    }}
                  >
                    {correlation.toFixed(2)}
                  </div>
                );
              })}
            </>
          ))}
        </div>
      </div>
    </div>
  );
}

// Define better types for visualization data
type HistogramData = {
  bins: number[];
  counts: number[];
  type: 'histogram';
};

type BarChartData = {
  data: { name: string; value: number }[];
  type: 'barChart';
};

type DistributionData = HistogramData | BarChartData;

export interface DatasetVisualizationsProps {
  datasetId: string;
}

export function DatasetVisualizations({ datasetId }: DatasetVisualizationsProps) {
  const [statistics, setStatistics] = useState<DatasetStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [visualizationType, setVisualizationType] = useState("distribution");
  
  useEffect(() => {
    async function fetchStatistics() {
      setLoading(true);
      setError(null);
      
      try {
        const stats = await fetchDatasetStatistics(datasetId);
        
        if (!stats) {
          setError("No statistics available for this dataset");
          setStatistics(null);
          return;
        }
        
        setStatistics(stats);
        
        // Set default selected feature to the first numeric column
        const numericColumns = Object.entries(stats.columns)
          .filter(([_, col]) => col.type === 'numeric')
          .map(([name]) => name);
          
        if (numericColumns.length > 0 && !selectedFeature) {
          setSelectedFeature(numericColumns[0]);
        }
      } catch (error) {
        console.error("Failed to fetch dataset statistics:", error);
        setError("Failed to load visualizations. Please try again.");
      } finally {
        setLoading(false);
      }
    }
    
    if (datasetId) {
      fetchStatistics();
    }
  }, [datasetId]);
  
  // Generate visualization data based on selected options
  const getDistributionData = (): DistributionData | null => {
    if (!statistics || !selectedFeature) return null;
    
    const column = statistics.columns[selectedFeature];
    
    if (column.type === 'numeric' && column.histogram) {
      return {
        ...column.histogram,
        type: 'histogram'
      };
    }
    
    if (column.type === 'categorical' && column.topValues) {
      return {
        data: column.topValues.map(tv => ({
          name: tv.value,
          value: tv.count
        })),
        type: 'barChart'
      };
    }
    
    return null;
  };
  
  const getCorrelationData = () => {
    if (!statistics) return {};
    
    const correlationData: Record<string, Record<string, number>> = {};
    
    // Extract all numeric columns with correlations
    Object.entries(statistics.columns)
      .filter(([_, col]) => col.type === 'numeric' && col.correlations)
      .forEach(([colName, col]) => {
        correlationData[colName] = col.correlations || {};
      });
    
    return correlationData;
  };
  
  const getScatterData = () => {
    if (!statistics || !selectedFeature) return [];
    
    // Find another numeric column to pair with the selected feature
    const secondFeature = Object.entries(statistics.columns)
      .filter(([name, col]) => 
        col.type === 'numeric' && 
        name !== selectedFeature
      )
      .map(([name]) => name)[0];
    
    if (!secondFeature) return [];
    
    // Generate mock scatter data
    return Array.from({ length: 30 }, (_, i) => ({
      x: Math.random() * 90 + 5,
      y: Math.random() * 90 + 5,
      label: `Point ${i+1}`
    }));
  };
  
  // Handle loading state
  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }
  
  // Handle error state
  if (error) {
    return (
      <div className="p-4 border border-red-200 bg-red-50 rounded-md">
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
  
  // Handle no data state
  if (!statistics) {
    return (
      <div className="p-4 border border-yellow-200 bg-yellow-50 rounded-md">
        <p className="text-yellow-600">No visualization data available for this dataset.</p>
      </div>
    );
  }
  
  const numericColumns = Object.entries(statistics.columns)
    .filter(([_, col]) => col.type === 'numeric')
    .map(([name]) => name);
    
  const categoricalColumns = Object.entries(statistics.columns)
    .filter(([_, col]) => col.type === 'categorical')
    .map(([name]) => name);
  
  const distributionData = getDistributionData();
  const correlationData = getCorrelationData();
  const scatterData = getScatterData();
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center gap-4 justify-between">
        <h3 className="text-lg font-medium">Dataset Visualizations</h3>
        
        <div className="flex flex-col md:flex-row gap-2">
          <div className="w-full md:w-48">
            <Select 
              value={visualizationType} 
              onValueChange={setVisualizationType}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select visualization" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="distribution">Distribution</SelectItem>
                <SelectItem value="correlation">Correlation Matrix</SelectItem>
                <SelectItem value="scatter">Scatter Plot</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {visualizationType !== 'correlation' && (
            <div className="w-full md:w-48">
              <Select 
                value={selectedFeature || ''} 
                onValueChange={setSelectedFeature}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select feature" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="" disabled>Select a feature</SelectItem>
                  {numericColumns.map(col => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                  {categoricalColumns.map(col => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>
      
      <div className="border rounded-md">
        {visualizationType === 'distribution' && distributionData && (
          <div className="p-4">
            <div className="mb-4">
              <h4 className="text-sm font-medium">Distribution of {selectedFeature}</h4>
              <p className="text-xs text-muted-foreground">
                {statistics.columns[selectedFeature || '']?.type === 'numeric' 
                  ? `Range: ${statistics.columns[selectedFeature || '']?.min} to ${statistics.columns[selectedFeature || '']?.max}`
                  : `${statistics.columns[selectedFeature || '']?.uniqueCount} unique values`
                }
              </p>
            </div>
            
            <div className="h-64">
              {distributionData.type === 'histogram' 
                ? <Histogram data={distributionData} />
                : <BarChart data={distributionData.data} />
              }
            </div>
          </div>
        )}
        
        {visualizationType === 'correlation' && (
          <div className="p-4">
            <div className="mb-4 flex justify-between items-start">
              <div>
                <h4 className="text-sm font-medium">Feature Correlation Matrix</h4>
                <p className="text-xs text-muted-foreground">
                  Pearson correlation between numeric features
                </p>
              </div>
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <Info className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
                      0 indicates no correlation. Blue indicates positive correlation, red indicates negative.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            
            <div className="overflow-x-auto">
              <CorrelationMatrix data={correlationData} />
            </div>
          </div>
        )}
        
        {visualizationType === 'scatter' && scatterData && (
          <div className="p-4">
            <div className="mb-4">
              <h4 className="text-sm font-medium">
                Scatter Plot - {selectedFeature} vs Other Features
              </h4>
              <p className="text-xs text-muted-foreground">
                Relationship between {selectedFeature} and other numeric features
              </p>
            </div>
            
            <div className="h-64">
              <ScatterPlot data={scatterData} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 