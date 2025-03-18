"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"

interface AdvancedModelConfigProps {
  modelType: string
  onConfigChange: (config: any) => void
  onReset: () => void
}

export function AdvancedModelConfig({ modelType, onConfigChange, onReset }: AdvancedModelConfigProps) {
  // Default configurations for different model types
  const defaultConfigs = {
    "linear-regression": {
      alpha: 0.01,
      normalize: true,
      fit_intercept: true,
      max_iter: 1000,
    },
    "random-forest": {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 2,
      min_samples_leaf: 1,
      bootstrap: true,
    },
    "differential-evolution": {
      population_size: 50,
      mutation: 0.8,
      crossover: 0.7,
      strategy: "best1bin",
      adaptive: true,
    },
    "evolution-strategy": {
      population_size: 100,
      sigma: 1.0,
      learning_rate: 0.1,
      adaptive: true,
    },
  }

  // State for current configuration
  const [config, setConfig] = useState<any>(defaultConfigs[modelType as keyof typeof defaultConfigs] || {})

  // Handle configuration changes
  const handleConfigChange = (key: string, value: any) => {
    const newConfig = { ...config, [key]: value }
    setConfig(newConfig)
    onConfigChange(newConfig)
  }

  // Reset configuration to defaults
  const handleReset = () => {
    const defaultConfig = defaultConfigs[modelType as keyof typeof defaultConfigs] || {}
    setConfig(defaultConfig)
    onConfigChange(defaultConfig)
    onReset()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Advanced Configuration</CardTitle>
        <CardDescription>Fine-tune model parameters</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {modelType === "linear-regression" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="alpha">Regularization (alpha)</Label>
                <span className="text-sm">{config.alpha}</span>
              </div>
              <Slider
                id="alpha"
                value={[config.alpha]}
                min={0.0001}
                max={1}
                step={0.0001}
                onValueChange={([value]) => handleConfigChange("alpha", value)}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="normalize"
                checked={config.normalize}
                onCheckedChange={(checked) => handleConfigChange("normalize", checked)}
              />
              <Label htmlFor="normalize">Normalize features</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="fit_intercept"
                checked={config.fit_intercept}
                onCheckedChange={(checked) => handleConfigChange("fit_intercept", checked)}
              />
              <Label htmlFor="fit_intercept">Fit intercept</Label>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="max_iter">Maximum iterations</Label>
                <span className="text-sm">{config.max_iter}</span>
              </div>
              <Slider
                id="max_iter"
                value={[config.max_iter]}
                min={100}
                max={10000}
                step={100}
                onValueChange={([value]) => handleConfigChange("max_iter", value)}
              />
            </div>
          </div>
        )}

        {modelType === "random-forest" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="n_estimators">Number of trees</Label>
                <span className="text-sm">{config.n_estimators}</span>
              </div>
              <Slider
                id="n_estimators"
                value={[config.n_estimators]}
                min={10}
                max={500}
                step={10}
                onValueChange={([value]) => handleConfigChange("n_estimators", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="max_depth">Maximum depth</Label>
                <span className="text-sm">{config.max_depth === null ? "None" : config.max_depth}</span>
              </div>
              <Slider
                id="max_depth"
                value={[config.max_depth || 10]}
                min={1}
                max={50}
                step={1}
                onValueChange={([value]) => handleConfigChange("max_depth", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="min_samples_split">Min samples split</Label>
                <span className="text-sm">{config.min_samples_split}</span>
              </div>
              <Slider
                id="min_samples_split"
                value={[config.min_samples_split]}
                min={2}
                max={20}
                step={1}
                onValueChange={([value]) => handleConfigChange("min_samples_split", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="min_samples_leaf">Min samples leaf</Label>
                <span className="text-sm">{config.min_samples_leaf}</span>
              </div>
              <Slider
                id="min_samples_leaf"
                value={[config.min_samples_leaf]}
                min={1}
                max={20}
                step={1}
                onValueChange={([value]) => handleConfigChange("min_samples_leaf", value)}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="bootstrap"
                checked={config.bootstrap}
                onCheckedChange={(checked) => handleConfigChange("bootstrap", checked)}
              />
              <Label htmlFor="bootstrap">Use bootstrap samples</Label>
            </div>
          </div>
        )}

        {modelType === "differential-evolution" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="population_size">Population size</Label>
                <span className="text-sm">{config.population_size}</span>
              </div>
              <Slider
                id="population_size"
                value={[config.population_size]}
                min={10}
                max={200}
                step={10}
                onValueChange={([value]) => handleConfigChange("population_size", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="mutation">Mutation factor</Label>
                <span className="text-sm">{config.mutation}</span>
              </div>
              <Slider
                id="mutation"
                value={[config.mutation]}
                min={0.1}
                max={1}
                step={0.1}
                onValueChange={([value]) => handleConfigChange("mutation", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="crossover">Crossover rate</Label>
                <span className="text-sm">{config.crossover}</span>
              </div>
              <Slider
                id="crossover"
                value={[config.crossover]}
                min={0.1}
                max={1}
                step={0.1}
                onValueChange={([value]) => handleConfigChange("crossover", value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="strategy">Strategy</Label>
              <Select value={config.strategy} onValueChange={(value) => handleConfigChange("strategy", value)}>
                <SelectTrigger id="strategy">
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="best1bin">best1bin</SelectItem>
                  <SelectItem value="rand1bin">rand1bin</SelectItem>
                  <SelectItem value="best2bin">best2bin</SelectItem>
                  <SelectItem value="rand2bin">rand2bin</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="adaptive"
                checked={config.adaptive}
                onCheckedChange={(checked) => handleConfigChange("adaptive", checked)}
              />
              <Label htmlFor="adaptive">Adaptive parameters</Label>
            </div>
          </div>
        )}

        {modelType === "evolution-strategy" && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="population_size">Population size</Label>
                <span className="text-sm">{config.population_size}</span>
              </div>
              <Slider
                id="population_size"
                value={[config.population_size]}
                min={10}
                max={200}
                step={10}
                onValueChange={([value]) => handleConfigChange("population_size", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="sigma">Initial sigma</Label>
                <span className="text-sm">{config.sigma}</span>
              </div>
              <Slider
                id="sigma"
                value={[config.sigma]}
                min={0.1}
                max={3}
                step={0.1}
                onValueChange={([value]) => handleConfigChange("sigma", value)}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="learning_rate">Learning rate</Label>
                <span className="text-sm">{config.learning_rate}</span>
              </div>
              <Slider
                id="learning_rate"
                value={[config.learning_rate]}
                min={0.01}
                max={1}
                step={0.01}
                onValueChange={([value]) => handleConfigChange("learning_rate", value)}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="adaptive"
                checked={config.adaptive}
                onCheckedChange={(checked) => handleConfigChange("adaptive", checked)}
              />
              <Label htmlFor="adaptive">Adaptive parameters</Label>
            </div>
          </div>
        )}

        <Separator className="my-4" />

        <Button variant="outline" className="w-full" onClick={handleReset}>
          Reset to Defaults
        </Button>
      </CardContent>
    </Card>
  )
}

