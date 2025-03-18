"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart } from "@/components/charts"

interface ExplainabilityPanelProps {
  modelType: string
  featureImportance: Record<string, number>
  sampleExplanations?: Array<{
    id: number
    features: Record<string, { value: number; contribution: number }>
    prediction: number
    explanation: string
  }>
}

export function ExplainabilityPanel({
  modelType,
  featureImportance,
  sampleExplanations = [],
}: ExplainabilityPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Explainability</CardTitle>
        <CardDescription>Understand model decisions and behavior</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Tabs defaultValue="feature">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="feature">Feature Impact</TabsTrigger>
              <TabsTrigger value="sample">Sample Explanation</TabsTrigger>
            </TabsList>

            <TabsContent value="feature" className="space-y-4">
              <div className="h-[150px] bg-muted rounded-md">
                <BarChart
                  data={Object.entries(featureImportance).map(([feature, value]) => ({
                    x: feature,
                    y: value,
                  }))}
                  xLabel="Feature"
                  yLabel="Impact"
                />
              </div>

              <div className="text-sm">
                <p className="font-medium">Key Insights:</p>
                <ul className="list-disc pl-5 text-muted-foreground">
                  {Object.entries(featureImportance)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 3)
                    .map(([feature, value], index) => (
                      <li key={index}>
                        Feature {feature} has {index === 0 ? "the highest" : "significant"} impact (
                        {(value * 100).toFixed(1)}%) on model predictions
                      </li>
                    ))}
                  <li>
                    {modelType.includes("forest")
                      ? "Tree-based models capture non-linear relationships between features"
                      : modelType.includes("regression")
                        ? "Linear relationships dominate the model behavior"
                        : "The model shows complex feature interactions"}
                  </li>
                </ul>
              </div>
            </TabsContent>

            <TabsContent value="sample" className="space-y-4">
              {sampleExplanations.length > 0 ? (
                <div className="bg-muted p-3 rounded-md">
                  <h3 className="text-sm font-medium mb-2">Sample #{sampleExplanations[0].id} Explanation</h3>
                  <div className="space-y-2">
                    {Object.entries(sampleExplanations[0].features).map(([feature, data]) => (
                      <div key={feature} className="flex items-center">
                        <div className={`w-1 h-6 ${data.contribution > 0 ? "bg-green-500" : "bg-red-500"} mr-2`}></div>
                        <div className="flex-1">
                          <div className="flex justify-between text-sm">
                            <span>
                              {feature} = {data.value.toFixed(2)}
                            </span>
                            <span>{data.contribution.toFixed(2)}</span>
                          </div>
                          <div className="w-full bg-secondary rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full ${data.contribution > 0 ? "bg-green-500" : "bg-red-500"}`}
                              style={{ width: `${Math.abs(data.contribution) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-muted p-3 rounded-md">
                  <h3 className="text-sm font-medium mb-2">Sample #42 Explanation</h3>
                  <div className="space-y-2">
                    {Object.entries(featureImportance).map(([feature, value]) => (
                      <div key={feature} className="flex items-center">
                        <div className={`w-1 h-6 ${value > 0.4 ? "bg-green-500" : "bg-red-500"} mr-2`}></div>
                        <div className="flex-1">
                          <div className="flex justify-between text-sm">
                            <span>{feature}</span>
                            <span>{value.toFixed(2)}</span>
                          </div>
                          <div className="w-full bg-secondary rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full ${value > 0.4 ? "bg-green-500" : "bg-red-500"}`}
                              style={{ width: `${value * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="text-sm">
                <p className="font-medium">Interpretation:</p>
                <p className="text-muted-foreground">
                  {sampleExplanations.length > 0
                    ? sampleExplanations[0].explanation
                    : `This sample was classified as ${modelType.includes("regression") ? "having a high value" : "positive"} primarily due to the high value of feature ${
                        Object.entries(featureImportance).sort((a, b) => b[1] - a[1])[0][0]
                      } which strongly indicates the ${modelType.includes("regression") ? "target value" : "positive class"} in this model.`}
                </p>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>
    </Card>
  )
}

