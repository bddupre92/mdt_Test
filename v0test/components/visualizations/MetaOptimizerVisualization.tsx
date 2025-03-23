import { useState } from "react";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle, 
  CardFooter
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Download, Info } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

// Types for Meta-Optimizer results
interface AlgorithmPrediction {
  algorithmId: string;
  algorithmName: string;
  confidence: number;
  actualPerformance?: number;
  rank: number;
}

interface ProblemFeature {
  name: string;
  value: number;
  description?: string;
  importance?: number;
}

interface MetaOptimizerResult {
  id: string;
  predictions: AlgorithmPrediction[];
  selectedAlgorithm: string;
  problemFeatures: ProblemFeature[];
  accuracy?: number;
  aggregateImprovement?: number;
}

interface MetaOptimizerVisualizationProps {
  result: MetaOptimizerResult;
  title?: string;
  description?: string;
  showExport?: boolean;
}

// Helper functions
const getAlgorithmColor = (algorithm: string, index: number): string => {
  const colors = [
    "#3b82f6", // blue 
    "#10b981", // green
    "#f97316", // orange
    "#8b5cf6", // purple
    "#ec4899", // pink
    "#f43f5e", // rose
    "#06b6d4", // cyan
    "#84cc16", // lime
  ];
  
  // Base color on algorithm name to keep consistent
  const nameHash = algorithm.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[nameHash % colors.length] || colors[index % colors.length];
};

// Format percentage display
const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

export function MetaOptimizerVisualization({
  result,
  title = "Meta-Optimizer Analysis",
  description = "Algorithm selection and performance predictions",
  showExport = true
}: MetaOptimizerVisualizationProps) {
  const [tab, setTab] = useState("predictions");
  
  // Get the selected algorithm details
  const selectedAlgorithm = result.predictions.find(
    p => p.algorithmId === result.selectedAlgorithm
  );
  
  // Sort features by importance (if available)
  const sortedFeatures = [...result.problemFeatures].sort((a, b) => 
    (b.importance || 0) - (a.importance || 0)
  );
  
  // Calculate prediction accuracy if actual performance is available
  const hasActualPerformance = result.predictions.some(p => p.actualPerformance !== undefined);
  
  // Prepare data for radar chart
  const radarData = result.problemFeatures.map(feature => ({
    subject: feature.name,
    value: feature.value,
    fullMark: 1
  }));
  
  // Export result data as JSON
  const exportData = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `meta-optimizer-result-${result.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          {showExport && (
            <Button variant="outline" size="sm" onClick={exportData}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="predictions" value={tab} onValueChange={setTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="predictions">Algorithm Predictions</TabsTrigger>
            <TabsTrigger value="features">Problem Characterization</TabsTrigger>
            {hasActualPerformance && (
              <TabsTrigger value="accuracy">Prediction Accuracy</TabsTrigger>
            )}
          </TabsList>
          
          <TabsContent value="predictions">
            <div className="mb-4">
              <div className="bg-muted/30 p-4 rounded-lg">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-3">
                  <h3 className="text-base font-medium">Selected Algorithm</h3>
                  <Badge className="mt-1 sm:mt-0">
                    Confidence: {formatPercentage(selectedAlgorithm?.confidence || 0)}
                  </Badge>
                </div>
                <p className="text-2xl font-bold text-primary">
                  {selectedAlgorithm?.algorithmName || "Unknown"}
                </p>
                {result.aggregateImprovement && (
                  <p className="text-sm mt-2 text-muted-foreground">
                    Expected improvement: +{formatPercentage(result.aggregateImprovement)}
                  </p>
                )}
              </div>
            </div>
            
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={result.predictions}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="algorithmName" 
                    angle={-45} 
                    textAnchor="end" 
                    height={70}
                  />
                  <YAxis
                    label={{
                      value: "Confidence",
                      angle: -90,
                      position: 'insideLeft',
                      style: { textAnchor: 'middle' }
                    }}
                    tickFormatter={(value) => formatPercentage(value)}
                  />
                  <Tooltip 
                    formatter={(value) => [formatPercentage(value as number), "Confidence"]}
                  />
                  <Legend />
                  <Bar dataKey="confidence" name="Predicted Confidence">
                    {result.predictions.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={getAlgorithmColor(entry.algorithmId, index)}
                        opacity={entry.algorithmId === result.selectedAlgorithm ? 1 : 0.7}
                        stroke={entry.algorithmId === result.selectedAlgorithm ? "#000" : "none"}
                        strokeWidth={entry.algorithmId === result.selectedAlgorithm ? 1 : 0}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="features">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium mb-3">Problem Characteristics</h3>
                <div className="space-y-3">
                  {sortedFeatures.map((feature) => (
                    <div key={feature.name} className="bg-muted/30 p-3 rounded-lg">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center">
                          <span className="font-medium text-sm">{feature.name}</span>
                          {feature.description && (
                            <Popover>
                              <PopoverTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-6 w-6 ml-1">
                                  <Info className="h-3 w-3" />
                                </Button>
                              </PopoverTrigger>
                              <PopoverContent className="w-80">
                                <p className="text-sm">{feature.description}</p>
                              </PopoverContent>
                            </Popover>
                          )}
                        </div>
                        <Badge variant="outline">
                          {feature.value.toFixed(3)}
                          {feature.importance && (
                            <span className="ml-2 text-muted-foreground">
                              (Importance: {(feature.importance * 100).toFixed(0)}%)
                            </span>
                          )}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={30} domain={[0, 1]} />
                    <Radar
                      name="Problem Features"
                      dataKey="value"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.5}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </TabsContent>
          
          {hasActualPerformance && (
            <TabsContent value="accuracy">
              <div className="h-72 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={result.predictions.filter(p => p.actualPerformance !== undefined)}
                    margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                      dataKey="algorithmName" 
                      angle={-45} 
                      textAnchor="end" 
                      height={70}
                    />
                    <YAxis
                      tickFormatter={(value) => formatPercentage(value)}
                    />
                    <Tooltip 
                      formatter={(value, name) => {
                        return [formatPercentage(value as number), name as string];
                      }}
                    />
                    <Legend />
                    <Bar dataKey="confidence" name="Predicted" fill="#8884d8" />
                    <Bar dataKey="actualPerformance" name="Actual" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="bg-muted/30 p-4 rounded-lg">
                <h3 className="font-medium mb-2">Prediction Accuracy</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Average accuracy</p>
                    <p className="font-bold text-lg">
                      {formatPercentage(result.accuracy || 0)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Best algorithm predicted</p>
                    <p className="font-bold text-lg">
                      {result.predictions.some(p => 
                        p.actualPerformance !== undefined && 
                        p.algorithmId === result.selectedAlgorithm && 
                        p.rank === 1
                      ) ? "Yes" : "No"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Top-3 algorithm predicted</p>
                    <p className="font-bold text-lg">
                      {result.predictions.some(p => 
                        p.actualPerformance !== undefined && 
                        p.algorithmId === result.selectedAlgorithm && 
                        p.rank <= 3
                      ) ? "Yes" : "No"}
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
      <CardFooter className="border-t pt-4">
        <div className="text-xs text-muted-foreground">
          Meta-optimizer prediction based on {result.problemFeatures.length} problem features
          and {result.predictions.length} candidate algorithms.
        </div>
      </CardFooter>
    </Card>
  );
} 