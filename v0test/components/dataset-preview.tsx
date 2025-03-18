import { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { fetchDatasetDetails, fetchDatasetPreview, fetchDatasetStatistics, DatasetDetails, DatasetPreviewData, DatasetStatistics } from '@/lib/api/datasets';
import { Loader2 } from 'lucide-react';
import { DatasetVisualizations } from "@/components/dataset-visualizations";

interface DatasetPreviewProps {
  datasetId: string;
}

export function DatasetPreview({ datasetId }: DatasetPreviewProps) {
  const [dataset, setDataset] = useState<DatasetDetails | null>(null);
  const [previewData, setPreviewData] = useState<DatasetPreviewData | null>(null);
  const [statistics, setStatistics] = useState<DatasetStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      
      try {
        // Fetch dataset details and preview data in parallel
        const [detailsData, previewData, statsData] = await Promise.all([
          fetchDatasetDetails(datasetId),
          fetchDatasetPreview(datasetId),
          fetchDatasetStatistics(datasetId)
        ]);
        
        // Check if we got valid data
        if (!detailsData) {
          setError(`Dataset with ID ${datasetId} not found.`);
          return;
        }
        
        setDataset(detailsData);
        setPreviewData(previewData);
        setStatistics(statsData);
      } catch (error) {
        console.error('Error fetching dataset information:', error);
        setError('Failed to load dataset information. Please try again.');
      } finally {
        setLoading(false);
      }
    }

    if (datasetId) {
      fetchData();
    }
  }, [datasetId]);

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  // Format number for display
  const formatNumber = (num: number) => {
    return num >= 1000 ? `${(num / 1000).toFixed(1)}k` : num.toString();
  };

  // Show loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center py-8">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-2">Loading dataset details...</span>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="border border-red-200 bg-red-50 p-4 rounded-md">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  // Show empty state
  if (!dataset) {
    return (
      <div className="border border-yellow-200 bg-yellow-50 p-4 rounded-md">
        <p className="text-yellow-600">No dataset information available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="summary">
        <TabsList>
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="preview">Data Preview</TabsTrigger>
          <TabsTrigger value="statistics">Statistics</TabsTrigger>
          <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
        </TabsList>
        
        <TabsContent value="summary" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Dataset Overview</CardTitle>
              <CardDescription>Key information about this dataset</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Name</h4>
                    <p>{dataset.name}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Description</h4>
                    <p>{dataset.description || 'No description available'}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Type</h4>
                    <Badge variant="outline">{dataset.type}</Badge>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Category</h4>
                    <p>{dataset.category}</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Dimensions</h4>
                    <p>{formatNumber(dataset.samples)} rows × {dataset.features} features</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">File format</h4>
                    <p>{dataset.fileFormat}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Source</h4>
                    <p>{dataset.sourcePath || 'Unknown'}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Last updated</h4>
                    <p>{formatDate(dataset.updatedAt)}</p>
                  </div>
                  
                  {dataset.metadata && Object.keys(dataset.metadata).length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Additional metadata</h4>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {Object.entries(dataset.metadata).map(([key, value]) => (
                          <Badge key={key} variant="secondary" className="text-xs">
                            {key}: {typeof value === 'object' ? 'Object' : String(value)}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {dataset.tags && dataset.tags.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground">Tags</h4>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {dataset.tags.map(tag => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Column Information</CardTitle>
              <CardDescription>Details about the dataset columns</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="border rounded-md overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Description</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataset.columns.map((column) => (
                      <TableRow key={column.name}>
                        <TableCell className="font-medium">{column.name}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{column.type}</Badge>
                        </TableCell>
                        <TableCell>{column.description || 'No description'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="preview" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Data Preview</CardTitle>
              <CardDescription>
                Showing first {previewData?.rows.length || 0} of {previewData?.totalRows.toLocaleString() || 'unknown'} rows
              </CardDescription>
            </CardHeader>
            <CardContent>
              {previewData ? (
                <div className="border rounded-md overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {previewData.columns.map((col) => (
                          <TableHead key={col}>{col}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {previewData.rows.map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                          {previewData.columns.map((col) => (
                            <TableCell key={`${rowIndex}-${col}`}>
                              {row[col]?.toString() || '-'}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
                  No preview data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="statistics" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Dataset Statistics</CardTitle>
              <CardDescription>
                Statistical information about this dataset
              </CardDescription>
            </CardHeader>
            <CardContent>
              {statistics ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                    <div className="border rounded-md p-3">
                      <h4 className="text-sm font-medium text-muted-foreground">Rows</h4>
                      <p className="text-2xl">{formatNumber(statistics.summary.rowCount)}</p>
                    </div>
                    <div className="border rounded-md p-3">
                      <h4 className="text-sm font-medium text-muted-foreground">Columns</h4>
                      <p className="text-2xl">{statistics.summary.columnCount}</p>
                    </div>
                    <div className="border rounded-md p-3">
                      <h4 className="text-sm font-medium text-muted-foreground">Missing cells</h4>
                      <p className="text-2xl">{formatNumber(statistics.summary.missingCells)}</p>
                      <p className="text-xs text-muted-foreground">{statistics.summary.missingPercentage}% of total</p>
                    </div>
                    <div className="border rounded-md p-3">
                      <h4 className="text-sm font-medium text-muted-foreground">Duplicate rows</h4>
                      <p className="text-2xl">{formatNumber(statistics.summary.duplicateRows)}</p>
                    </div>
                    <div className="border rounded-md p-3">
                      <h4 className="text-sm font-medium text-muted-foreground">File size</h4>
                      <p className="text-2xl">{statistics.summary.fileSize}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-md font-medium mb-3">Column Statistics</h3>
                    <div className="border rounded-md overflow-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Missing</TableHead>
                            <TableHead>Unique</TableHead>
                            <TableHead>Statistics</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(statistics.columns).map(([colName, colStats]) => (
                            <TableRow key={colName}>
                              <TableCell className="font-medium">{colName}</TableCell>
                              <TableCell>
                                <Badge variant="outline">{colStats.type}</Badge>
                              </TableCell>
                              <TableCell>{colStats.missingCount}</TableCell>
                              <TableCell>{colStats.uniqueCount}</TableCell>
                              <TableCell>
                                {colStats.type === 'numeric' ? (
                                  <span>
                                    Min: {colStats.min?.toFixed(2)}, 
                                    Max: {colStats.max?.toFixed(2)}, 
                                    Mean: {colStats.mean?.toFixed(2)}
                                  </span>
                                ) : colStats.type === 'categorical' && colStats.topValues ? (
                                  <span>
                                    Top: {colStats.topValues[0]?.value} 
                                    ({colStats.topValues[0]?.count} times)
                                  </span>
                                ) : (
                                  '-'
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
                  No statistics available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="visualizations" className="space-y-4">
          <DatasetVisualizations datasetId={datasetId} />
        </TabsContent>
      </Tabs>
    </div>
  );
} 