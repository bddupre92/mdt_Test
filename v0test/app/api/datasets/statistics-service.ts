import { Dataset, getDatasetById, getAllDatasets } from "./data-service";

// Types for statistics
export interface DatasetStatistics {
  summary: {
    rowCount: number;
    columnCount: number;
    missingCells: number;
    missingPercentage: number;
    duplicateRows: number;
    fileSize: string;
  };
  columns: Record<string, {
    type: string;
    missingCount: number;
    uniqueCount: number;
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    stdDev?: number;
    histogram?: {
      bins: number[];
      counts: number[];
    };
    topValues?: Array<{value: string; count: number}>;
    correlations?: Record<string, number>;
  }>;
}

// Service function to get statistics
export async function getDatasetStatistics(datasetId: string): Promise<DatasetStatistics | null> {
  // Get the dataset first
  console.log(`Statistics service: Fetching dataset with ID: ${datasetId}`);
  const dataset = await getDatasetById(datasetId);
  
  if (!dataset) {
    console.log(`Statistics service: Dataset with ID ${datasetId} not found!`);
    // Check if we can get all datasets to debug
    const allDatasets = await getAllDatasets();
    console.log(`Statistics service: Total datasets available: ${allDatasets.length}`);
    if (allDatasets.length > 0) {
      console.log(`Statistics service: Available dataset IDs: ${allDatasets.map(d => d.id).join(', ')}`);
    }
    return null;
  }
  
  console.log(`Statistics service: Found dataset "${dataset.name}" (${dataset.id}), generating statistics...`);
  
  // Generate column names based on dataset type
  const columnNames = generateMockColumnNames(dataset);
  const columns: Record<string, any> = {};
  
  // Generate statistics for each column
  columnNames.forEach(column => {
    const isNumeric = !column.includes('category') && 
                     !column.includes('class') && 
                     !column.includes('type') && 
                     !column.includes('date') && 
                     !column.includes('time');
    
    if (isNumeric) {
      // Numeric column statistics
      const min = Math.floor(Math.random() * 10);
      const max = min + Math.floor(Math.random() * 90) + 10;
      const mean = min + (max - min) / 2 + (Math.random() * 10 - 5);
      const median = mean + (Math.random() * 5 - 2.5);
      const stdDev = (max - min) / 6;
      
      columns[column] = {
        type: 'numeric',
        missingCount: Math.floor(Math.random() * 10),
        uniqueCount: Math.floor(dataset.samples * 0.8),
        min,
        max,
        mean,
        median,
        stdDev,
        histogram: {
          bins: Array.from({length: 10}, (_, i) => min + (i * (max - min) / 10)),
          counts: Array.from({length: 10}, () => Math.floor(Math.random() * 50) + 5)
        },
        correlations: generateMockCorrelations(columnNames)
      };
    } else if (column.includes('date') || column.includes('time')) {
      // Date column statistics
      columns[column] = {
        type: 'datetime',
        missingCount: Math.floor(Math.random() * 5),
        uniqueCount: Math.floor(dataset.samples * 0.9),
        topValues: [
          {value: '2023-01-01', count: 15},
          {value: '2023-01-02', count: 12},
          {value: '2023-01-03', count: 10},
          {value: '2023-01-04', count: 8},
          {value: '2023-01-05', count: 5}
        ]
      };
    } else {
      // Categorical column statistics
      columns[column] = {
        type: 'categorical',
        missingCount: Math.floor(Math.random() * 5),
        uniqueCount: Math.floor(Math.random() * 10) + 2,
        topValues: [
          {value: 'A', count: 42},
          {value: 'B', count: 38},
          {value: 'C', count: 25},
          {value: 'D', count: 15}
        ]
      };
    }
  });
  
  // Calculate summary statistics
  const rowCount = dataset.samples;
  const columnCount = columnNames.length;
  const missingCells = Object.values(columns).reduce((sum, col: any) => sum + col.missingCount, 0);
  
  return {
    summary: {
      rowCount,
      columnCount,
      missingCells,
      missingPercentage: parseFloat(((missingCells / (rowCount * columnCount)) * 100).toFixed(2)),
      duplicateRows: Math.floor(rowCount * 0.02), // 2% duplicate rows
      fileSize: `${Math.floor(rowCount * columnCount * 8 / 1024)} KB`
    },
    columns
  };
}

// Helper function to generate mock column names
function generateMockColumnNames(dataset: Dataset): string[] {
  const columnNames = [];
  
  // Add id column
  columnNames.push('id');
  
  if (dataset.type === 'classification') {
    // Feature columns
    for (let i = 1; i < dataset.features; i++) {
      columnNames.push(`feature_${i}`);
    }
    
    // Class column
    columnNames.push('class');
  } else if (dataset.type === 'regression') {
    // Feature columns
    for (let i = 1; i < dataset.features; i++) {
      columnNames.push(`feature_${i}`);
    }
    
    // Target column
    columnNames.push('target');
  } else if (dataset.type === 'time-series') {
    // Date column
    columnNames.push('date');
    
    // Value columns
    for (let i = 1; i < dataset.features; i++) {
      columnNames.push(`value_${i}`);
    }
  } else if (dataset.type === 'clustering') {
    // Feature columns for clustering
    for (let i = 1; i <= dataset.features; i++) {
      if (i % 3 === 0) {
        columnNames.push(`category_${i}`);
      } else {
        columnNames.push(`feature_${i}`);
      }
    }
  }
  
  return columnNames;
}

// Helper function to generate mock correlations
function generateMockCorrelations(columns: string[]): Record<string, number> {
  const correlations: Record<string, number> = {};
  
  columns.forEach(col => {
    // Don't create correlation with non-numeric columns
    if (col.includes('category') || 
        col.includes('class') || 
        col.includes('type') || 
        col.includes('date') || 
        col.includes('time')) {
      return;
    }
    
    correlations[col] = parseFloat((Math.random() * 2 - 1).toFixed(2)); // -1 to 1
  });
  
  return correlations;
} 