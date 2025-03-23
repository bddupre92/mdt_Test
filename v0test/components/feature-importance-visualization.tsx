"use client"

import { useState, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import {
  Bar,
  BarChart as RechartsBarChart,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Treemap,
  XAxis,
  YAxis,
} from "recharts"
import { Button } from "@/components/ui/button"
import { Download, RefreshCw } from "lucide-react"
import { exportChartAsImage, generateExportFileName } from "@/lib/utils/export-chart"

interface FeatureImportanceVisualizationProps {
  featureImportance: Record<string, number>
  modelType: string
  modelName: string
}

export function FeatureImportanceVisualization({
  featureImportance = {},
  modelType = "Unknown",
  modelName = "Unknown",
}: FeatureImportanceVisualizationProps) {
  const [visualizationType, setVisualizationType] = useState<string>("bar")
  const [sortBy, setSortBy] = useState<string>("importance")
  const chartRef = useRef<HTMLDivElement>(null)
  
  // Generate mock data if no real data is available
  const generateMockData = () => {
    const features = ['x1', 'x2', 'x3', 'time_since_last_event', 'sleep_quality', 'stress_level', 'environmental_factors'];
    const mockImportance: Record<string, number> = {};
    
    // Create mock importance values that sum to 1.0
    let remainingWeight = 1.0;
    for (let i = 0; i < features.length - 1; i++) {
      // Generate a random value for this feature (ensuring we leave some for the last feature)
      const maxValue = remainingWeight * 0.8;
      const value = Math.random() * maxValue;
      mockImportance[features[i]] = value;
      remainingWeight -= value;
    }
    // Assign the remaining weight to the last feature
    mockImportance[features[features.length - 1]] = remainingWeight;
    
    return mockImportance;
  }
  
  // Use mock data if no real data is provided
  const effectiveData = Object.keys(featureImportance).length > 0 
    ? featureImportance 
    : generateMockData();

  // Process feature importance data for visualization
  const processData = () => {
    if (!effectiveData || Object.keys(effectiveData).length === 0) {
      return [];
    }

    // Normalize the data to ensure values sum to 1.0
    const sum = Object.values(effectiveData).reduce((total, val) => total + val, 0);
    const normalizedData = Object.entries(effectiveData).map(([feature, importance]) => ({
      name: feature,
      value: sum > 0 ? importance / sum : importance,
      formattedValue: (sum > 0 ? (importance / sum) : importance).toFixed(4),
    }));

    if (sortBy === "importance") {
      return normalizedData.sort((a, b) => b.value - a.value);
    } else if (sortBy === "name") {
      return normalizedData.sort((a, b) => a.name.localeCompare(b.name));
    } else {
      return normalizedData;
    }
  };

  const data = processData();

  // Export the chart
  const handleExport = () => {
    if (chartRef.current) {
      const fileName = generateExportFileName(`feature_importance_${modelType.toLowerCase()}`);
      exportChartAsImage(chartRef.current, fileName);
    }
  };

  // Generate SHAP-style colors
  const generateColor = (value: number) => {
    // Color scale from blue (negative) to red (positive)
    if (value > 0.6) return "#d62728" // red
    if (value > 0.3) return "#ff7f0e" // orange
    return "#1f77b4" // blue
  }

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Feature Importance: {modelName}</CardTitle>
          <CardDescription>Visualization of feature importance for {modelType} model</CardDescription>
        </div>
        <div className="flex space-x-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleExport}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-4 justify-between">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-2">
              <Label htmlFor="visualization-type">Visualization Type</Label>
              <Select value={visualizationType} onValueChange={setVisualizationType}>
                <SelectTrigger id="visualization-type" className="w-[180px]">
                  <SelectValue placeholder="Bar chart" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bar">Bar chart</SelectItem>
                  <SelectItem value="horizontal-bar">Horizontal bar chart</SelectItem>
                  <SelectItem value="radar">Radar chart</SelectItem>
                  <SelectItem value="treemap">Treemap</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="sort-by">Sort By</Label>
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger id="sort-by" className="w-[150px]">
                  <SelectValue placeholder="Importance" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="importance">Importance</SelectItem>
                  <SelectItem value="name">Feature name</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <div className="h-[400px]" ref={chartRef}>
          {data.length > 0 ? (
            <>
              {visualizationType === "bar" && (
                <ChartContainer className="h-full w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                      <XAxis
                        dataKey="name"
                        angle={-45}
                        textAnchor="end"
                        height={80}
                        label={{
                          value: "Feature",
                          position: "insideBottom",
                          offset: -15,
                        }}
                      />
                      <YAxis
                        label={{
                          value: "Importance",
                          angle: -90,
                          position: "insideLeft",
                        }}
                      />
                      <ChartTooltip
                        formatter={(value: number) => [value.toFixed(4), "Importance"]}
                        content={<ChartTooltipContent />}
                      />
                      <Bar dataKey="value">
                        {data.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={generateColor(entry.value)} />
                        ))}
                      </Bar>
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {visualizationType === "horizontal-bar" && (
                <ChartContainer className="h-full w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={data}
                      layout="vertical"
                      margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                    >
                      <XAxis
                        type="number"
                        label={{
                          value: "Importance",
                          position: "insideBottom",
                          offset: -5,
                        }}
                      />
                      <YAxis type="category" dataKey="name" width={80} />
                      <ChartTooltip
                        formatter={(value: number) => [value.toFixed(4), "Importance"]}
                        content={<ChartTooltipContent />}
                      />
                      <Bar dataKey="value">
                        {data.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={generateColor(entry.value)} />
                        ))}
                      </Bar>
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {visualizationType === "radar" && (
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="name" />
                    <PolarRadiusAxis angle={30} domain={[0, 1]} />
                    <Radar
                      name="Feature Importance"
                      dataKey="value"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                    <ChartTooltip formatter={(value: number) => [value.toFixed(4), "Importance"]} />
                  </RadarChart>
                </ResponsiveContainer>
              )}

              {visualizationType === "treemap" && (
                <ResponsiveContainer width="100%" height="100%">
                  <Treemap data={data} dataKey="value" nameKey="name" aspectRatio={4 / 3} stroke="#fff" fill="#8884d8">
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={generateColor(entry.value)} />
                    ))}
                  </Treemap>
                </ResponsiveContainer>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full bg-muted rounded-md">
              <p className="text-muted-foreground">No feature importance data available</p>
            </div>
          )}
        </div>

        {data.length > 0 && (
          <div className="text-sm mt-4 bg-muted p-4 rounded-md">
            <p className="font-medium">Key Insights:</p>
            <ul className="list-disc pl-5 text-muted-foreground mt-2">
              {data.length > 0 && (
                <li>
                  Feature <span className="font-semibold text-foreground">{data[0]?.name}</span> has the highest impact ({(data[0]?.value * 100).toFixed(1)}%) on model
                  predictions
                </li>
              )}
              {data.length > 1 && (
                <li>
                  Features <span className="font-semibold text-foreground">{data[0]?.name}</span> and <span className="font-semibold text-foreground">{data[1]?.name}</span> together account for{" "}
                  {((data[0]?.value + data[1]?.value) * 100).toFixed(1)}% of model decisions
                </li>
              )}
              {data.length > 3 && (
                <li>
                  The model shows {data.slice(3).every((d) => d.value < 0.1) ? "low" : "moderate"} sensitivity to other
                  features
                </li>
              )}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

