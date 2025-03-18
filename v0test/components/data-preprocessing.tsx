"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"

interface DataPreprocessingProps {
  onPreprocess: (options: any) => void
  dataType: string
}

export function DataPreprocessing({ onPreprocess, dataType }: DataPreprocessingProps) {
  const [options, setOptions] = useState({
    normalize: true,
    removeOutliers: false,
    outlierThreshold: 3.0,
    fillMissingValues: "mean",
    trainTestSplit: 0.2,
    randomSeed: 42,
    featureSelection: false,
    maxFeatures: 10,
  })

  const handleOptionChange = (key: string, value: any) => {
    setOptions((prev) => ({ ...prev, [key]: value }))
  }

  const handleApply = () => {
    onPreprocess(options)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Preprocessing</CardTitle>
        <CardDescription>Configure preprocessing options for your dataset</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Switch
              id="normalize"
              checked={options.normalize}
              onCheckedChange={(checked) => handleOptionChange("normalize", checked)}
            />
            <Label htmlFor="normalize">Normalize features</Label>
          </div>
          <Badge variant="outline">{options.normalize ? "Enabled" : "Disabled"}</Badge>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Switch
              id="removeOutliers"
              checked={options.removeOutliers}
              onCheckedChange={(checked) => handleOptionChange("removeOutliers", checked)}
            />
            <Label htmlFor="removeOutliers">Remove outliers</Label>
          </div>
          <Badge variant="outline">{options.removeOutliers ? "Enabled" : "Disabled"}</Badge>
        </div>

        {options.removeOutliers && (
          <div className="space-y-2 pl-6">
            <div className="flex items-center justify-between">
              <Label htmlFor="outlierThreshold">Outlier threshold (z-score)</Label>
              <span className="text-sm">{options.outlierThreshold}</span>
            </div>
            <Slider
              id="outlierThreshold"
              value={[options.outlierThreshold]}
              min={1.5}
              max={5}
              step={0.1}
              onValueChange={([value]) => handleOptionChange("outlierThreshold", value)}
            />
          </div>
        )}

        <div className="space-y-2">
          <Label htmlFor="fillMissingValues">Fill missing values</Label>
          <Select
            value={options.fillMissingValues}
            onValueChange={(value) => handleOptionChange("fillMissingValues", value)}
          >
            <SelectTrigger id="fillMissingValues">
              <SelectValue placeholder="Select method" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mean">Mean</SelectItem>
              <SelectItem value="median">Median</SelectItem>
              <SelectItem value="mode">Mode</SelectItem>
              <SelectItem value="zero">Zero</SelectItem>
              <SelectItem value="none">Don't fill</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="trainTestSplit">Train/Test split ratio</Label>
            <span className="text-sm">{options.trainTestSplit}</span>
          </div>
          <Slider
            id="trainTestSplit"
            value={[options.trainTestSplit]}
            min={0.1}
            max={0.5}
            step={0.05}
            onValueChange={([value]) => handleOptionChange("trainTestSplit", value)}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Switch
              id="featureSelection"
              checked={options.featureSelection}
              onCheckedChange={(checked) => handleOptionChange("featureSelection", checked)}
            />
            <Label htmlFor="featureSelection">Automatic feature selection</Label>
          </div>
          <Badge variant="outline">{options.featureSelection ? "Enabled" : "Disabled"}</Badge>
        </div>

        {options.featureSelection && (
          <div className="space-y-2 pl-6">
            <div className="flex items-center justify-between">
              <Label htmlFor="maxFeatures">Maximum features</Label>
              <span className="text-sm">{options.maxFeatures}</span>
            </div>
            <Slider
              id="maxFeatures"
              value={[options.maxFeatures]}
              min={1}
              max={20}
              step={1}
              onValueChange={([value]) => handleOptionChange("maxFeatures", value)}
            />
          </div>
        )}

        <Separator className="my-4" />

        <Button className="w-full" onClick={handleApply}>
          Apply Preprocessing
        </Button>
      </CardContent>
    </Card>
  )
}

