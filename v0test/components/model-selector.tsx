"use client"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

interface Model {
  id: string
  name: string
  type: string
  description: string
}

interface ModelSelectorProps {
  models: Model[]
  selectedModel: string
  onSelectModel: (modelId: string) => void
}

export function ModelSelector({ models, selectedModel, onSelectModel }: ModelSelectorProps) {
  return (
    <RadioGroup value={selectedModel} onValueChange={onSelectModel}>
      <div className="space-y-2">
        {models.map((model) => (
          <div
            key={model.id}
            className={`border rounded-lg p-4 transition-colors ${
              selectedModel === model.id ? "border-primary bg-primary/5" : "border-border"
            }`}
          >
            <RadioGroupItem value={model.id} id={model.id} className="sr-only" />
            <Label htmlFor={model.id} className="flex flex-col space-y-1 cursor-pointer">
              <div className="flex items-center justify-between">
                <span className="font-medium">{model.name}</span>
                <Badge variant="outline">{model.type}</Badge>
              </div>
              <span className="text-sm text-muted-foreground">{model.description}</span>
            </Label>
          </div>
        ))}
      </div>
    </RadioGroup>
  )
}

