"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { AlertCircle, Download, RefreshCw } from 'lucide-react'

interface VisualizationFile {
  name: string
  path: string
  url: string
}

interface MetaOptimizerVisualizerProps {
  visualizationFiles?: VisualizationFile[]
  resultData?: any
  outputDir?: string
  isLoading?: boolean
  onRefresh?: () => void
}

export function MetaOptimizerVisualizer({
  visualizationFiles = [],
  resultData,
  outputDir,
  isLoading = false,
  onRefresh
}: MetaOptimizerVisualizerProps) {
  // Handle image download
  const handleDownloadImage = (url: string, name: string) => {
    const link = document.createElement('a')
    link.href = url
    link.download = name
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
  
  // Format file name to be more human-readable
  const formatTitle = (name: string) => {
    // Remove file extension
    const baseName = name.replace(/\.[^/.]+$/, '')
    // Replace underscores with spaces and capitalize
    return baseName
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
  }
  
  return (
    <div className="w-full">
      {isLoading ? (
        <div className="flex flex-col items-center justify-center p-12">
          <Spinner className="w-12 h-12 mb-4" />
          <p className="text-muted-foreground">Loading visualizations...</p>
        </div>
      ) : visualizationFiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center border rounded-lg p-12">
          <AlertCircle className="w-12 h-12 mb-4 text-muted-foreground" />
          <p className="text-lg mb-2">No visualizations available</p>
          <p className="text-sm text-muted-foreground mb-4">Run the Meta-Optimizer to generate visualizations</p>
          {onRefresh && (
            <Button onClick={onRefresh} variant="outline" size="sm">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Meta-Optimizer Visualizations</h3>
            {onRefresh && (
              <Button onClick={onRefresh} variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            )}
          </div>
          
          {outputDir && (
            <p className="text-sm text-muted-foreground">
              Results saved in: {outputDir}
            </p>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {visualizationFiles.map((file, idx) => (
              <Card key={idx} className="overflow-hidden">
                <CardHeader className="p-4">
                  <CardTitle className="text-sm truncate" title={file.name}>
                    {formatTitle(file.name)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="relative aspect-[4/3] bg-muted flex items-center justify-center">
                    <img 
                      src={file.url} 
                      alt={file.name}
                      className="object-contain w-full h-full"
                    />
                  </div>
                  <div className="p-2 flex justify-end">
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => handleDownloadImage(file.url, file.name)}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
} 