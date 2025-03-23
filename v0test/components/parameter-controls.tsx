"use client"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Parameter {
  id: string
  name: string
  type: "slider" | "switch" | "input" | "select"
  value: any
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
}

interface ParameterControlsProps {
  parameters: Parameter[]
  onChange: (id: string, value: any) => void
}

export function ParameterControls({ parameters, onChange }: ParameterControlsProps) {
  return (
    <div className="space-y-4">
      {parameters.map((param) => (
        <div key={param.id} className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor={param.id}>{param.name}</Label>
            {param.type === "slider" && <span className="text-sm">{param.value}</span>}
          </div>

          {param.type === "slider" && (
            <Slider
              id={param.id}
              value={[param.value]}
              min={param.min}
              max={param.max}
              step={param.step}
              onValueChange={([value]) => onChange(param.id, value)}
            />
          )}

          {param.type === "switch" && (
            <div className="flex items-center space-x-2">
              <Switch id={param.id} checked={param.value} onCheckedChange={(checked) => onChange(param.id, checked)} />
              <Label htmlFor={param.id}>Enabled</Label>
            </div>
          )}

          {param.type === "input" && (
            <Input id={param.id} value={param.value} onChange={(e) => onChange(param.id, e.target.value)} />
          )}

          {param.type === "select" && param.options && (
            <Select value={param.value} onValueChange={(value) => onChange(param.id, value)}>
              <SelectTrigger id={param.id}>
                <SelectValue placeholder="Select option" />
              </SelectTrigger>
              <SelectContent>
                {param.options.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      ))}
    </div>
  )
}

