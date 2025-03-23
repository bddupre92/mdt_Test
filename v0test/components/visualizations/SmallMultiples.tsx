import { useMemo } from "react";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { BarChart2, LineChart as LineChartIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface MetricData {
  [key: string]: number | undefined;
}

interface AlgorithmPerformance {
  algorithm: string;
  algorithmName?: string;
  metrics: MetricData;
  convergence?: Array<{
    iteration: number;
    fitness: number;
  }>;
}

interface SmallMultiplesProps {
  data: AlgorithmPerformance[];
  metrics?: Array<{
    key: string;
    name: string;
    isLowerBetter?: boolean;
    formatter?: (value: number) => string;
  }>;
  title?: string;
  description?: string;
  columns?: number;
  highlightAlgorithm?: string;
}

// Define color palette for algorithms
const ALGORITHM_COLORS = [
  "#3b82f6", // blue
  "#10b981", // green
  "#f97316", // orange
  "#8b5cf6", // purple
  "#ec4899", // pink
  "#f43f5e", // rose
  "#06b6d4", // cyan
  "#84cc16", // lime
  "#eab308", // yellow
  "#6366f1", // indigo
];

export function SmallMultiples({
  data,
  metrics = [
    { key: "fitness", name: "Fitness", isLowerBetter: true },
    { key: "executionTime", name: "Time (s)", isLowerBetter: true },
    { key: "iterations", name: "Iterations", isLowerBetter: false }
  ],
  title = "Algorithm Comparison",
  description = "Compare algorithm performance across multiple metrics",
  columns = 3,
  highlightAlgorithm
}: SmallMultiplesProps) {
  const [viewType, setViewType] = React.useState<string>("bars");
  
  // Process data for visualization
  const processedData = useMemo(() => {
    // For each metric, normalize the values
    return metrics.map(metric => {
      const isLowerBetter = metric.isLowerBetter ?? false;
      
      // Extract values for this metric
      const values = data
        .map(item => item.metrics[metric.key] ?? 0)
        .filter(value => value !== undefined);
      
      // Find min and max
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;
      
      // For each algorithm, calculate normalized value
      const metricData = data.map((item, index) => {
        const value = item.metrics[metric.key] ?? 0;
        
        // Normalize to 0-1 range (1 is always better)
        let normalizedValue;
        if (range === 0) {
          normalizedValue = 0.5; // If all values are the same
        } else if (isLowerBetter) {
          normalizedValue = (max - value) / range;
        } else {
          normalizedValue = (value - min) / range;
        }
        
        return {
          algorithm: item.algorithm,
          algorithmName: item.algorithmName || item.algorithm,
          value,
          normalizedValue,
          color: item.color || ALGORITHM_COLORS[index % ALGORITHM_COLORS.length],
          isHighlighted: highlightAlgorithm === item.algorithm
        };
      });
      
      // Sort by performance (normalized value)
      metricData.sort((a, b) => b.normalizedValue - a.normalizedValue);
      
      return {
        metric: metric.key,
        metricName: metric.name,
        isLowerBetter,
        formatter: metric.formatter || (value => value.toFixed(4)),
        data: metricData,
        bestValue: isLowerBetter ? min : max
      };
    });
  }, [data, metrics, highlightAlgorithm]);
  
  // Prepare convergence data if available
  const convergenceData = useMemo(() => {
    const algorithmsWithConvergence = data.filter(item => item.convergence && item.convergence.length > 0);
    
    if (algorithmsWithConvergence.length === 0) {
      return null;
    }
    
    // Find max iterations
    const maxIterations = Math.max(
      ...algorithmsWithConvergence.map(item => 
        item.convergence ? item.convergence[item.convergence.length - 1].iteration : 0
      )
    );
    
    // Create data points for each algorithm
    return {
      maxIterations,
      algorithms: algorithmsWithConvergence.map((item, index) => ({
        algorithm: item.algorithm,
        algorithmName: item.algorithmName || item.algorithm,
        convergence: item.convergence || [],
        color: item.color || ALGORITHM_COLORS[index % ALGORITHM_COLORS.length],
        isHighlighted: highlightAlgorithm === item.algorithm
      }))
    };
  }, [data, highlightAlgorithm]);
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <ToggleGroup 
            type="single" 
            value={viewType} 
            onValueChange={(value) => value && setViewType(value)}
          >
            <ToggleGroupItem value="bars" aria-label="Bar charts">
              <BarChart2 className="h-4 w-4" />
            </ToggleGroupItem>
            {convergenceData && (
              <ToggleGroupItem value="lines" aria-label="Line charts">
                <LineChartIcon className="h-4 w-4" />
              </ToggleGroupItem>
            )}
          </ToggleGroup>
        </div>
      </CardHeader>
      <CardContent>
        {viewType === "bars" ? (
          <div className={`grid grid-cols-1 md:grid-cols-${columns} gap-4`}>
            {processedData.map(metricInfo => (
              <Card key={metricInfo.metric} className="overflow-hidden">
                <CardHeader className="p-4 pb-2">
                  <div className="flex justify-between items-center">
                    <CardTitle className="text-base">{metricInfo.metricName}</CardTitle>
                    <Badge variant="outline" className="text-xs">
                      {metricInfo.isLowerBetter ? "Lower is better" : "Higher is better"}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    Best: {metricInfo.formatter(metricInfo.bestValue)}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  <div className="h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={metricInfo.data}
                        layout="vertical"
                        margin={{ top: 5, right: 5, bottom: 5, left: 40 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                        <XAxis type="number" />
                        <YAxis 
                          type="category" 
                          dataKey="algorithmName" 
                          width={80}
                          tick={{ fontSize: 12 }}
                        />
                        <Tooltip
                          formatter={(value) => [
                            metricInfo.formatter(value as number),
                            metricInfo.metricName
                          ]}
                        />
                        <Bar dataKey="value">
                          {metricInfo.data.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={entry.color}
                              fillOpacity={entry.isHighlighted ? 1 : 0.7}
                              stroke={entry.isHighlighted ? "#000" : "none"}
                              strokeWidth={entry.isHighlighted ? 1 : 0}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className={`grid grid-cols-1 md:grid-cols-${columns} gap-4`}>
            {convergenceData && convergenceData.algorithms.map(algorithm => (
              <Card key={algorithm.algorithm} className="overflow-hidden">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-base">{algorithm.algorithmName}</CardTitle>
                  <CardDescription className="text-xs">
                    Convergence over {convergenceData.maxIterations} iterations
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-4 pt-2">
                  <div className="h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={algorithm.convergence}
                        margin={{ top: 5, right: 5, bottom: 5, left: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis 
                          dataKey="iteration" 
                          type="number"
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis 
                          tick={{ fontSize: 10 }}
                        />
                        <Tooltip
                          formatter={(value) => [
                            (value as number).toFixed(6),
                            "Fitness"
                          ]}
                          labelFormatter={(value) => `Iteration ${value}`}
                        />
                        <Line
                          type="monotone"
                          dataKey="fitness"
                          stroke={algorithm.color}
                          dot={false}
                          strokeWidth={algorithm.isHighlighted ? 3 : 2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 