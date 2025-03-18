import { v4 as uuidv4 } from "uuid"

// Dataset types
export interface Dataset {
  id: string;
  name: string;
  description: string;
  type: string;
  category: string;
  features: number;
  samples: number;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, any>;
}

// Mock datasets
export let mockDatasets: Dataset[] = [
  {
    id: "1",
    name: "Migraine EEG Recordings",
    description: "Electroencephalogram recordings from migraine patients",
    type: "time-series",
    category: "physiological",
    features: 32,
    samples: 5000,
    createdAt: "2023-04-15T09:22:43Z",
    updatedAt: "2023-04-15T09:22:43Z",
    metadata: {
      source: "Clinical study",
      format: "CSV"
    }
  },
  {
    id: "2",
    name: "Environmental Triggers",
    description: "Environmental factors correlated with migraine onset",
    type: "regression",
    category: "environmental",
    features: 15,
    samples: 3200,
    createdAt: "2023-05-20T14:30:12Z",
    updatedAt: "2023-05-20T14:30:12Z",
    metadata: {
      source: "Patient survey",
      format: "CSV"
    }
  },
  {
    id: "3",
    name: "Symptom Classification",
    description: "Patient-reported symptoms for migraine classification",
    type: "classification",
    category: "clinical",
    features: 24,
    samples: 2800,
    createdAt: "2023-06-05T11:15:32Z",
    updatedAt: "2023-06-05T11:15:32Z",
    metadata: {
      source: "Clinical records",
      format: "CSV"
    }
  }
];

// Service functions
export async function getAllDatasets() {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  console.log(`Data service: Returning all datasets (${mockDatasets.length} total)`);
  return mockDatasets;
}

export async function getDatasetById(id: string) {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  const dataset = mockDatasets.find(d => d.id === id);
  if (dataset) {
    console.log(`Data service: Found dataset "${dataset.name}" with ID ${id}`);
  } else {
    console.log(`Data service: Dataset with ID ${id} not found. Available IDs: ${mockDatasets.map(d => d.id).join(', ')}`);
  }
  
  return dataset;
}

export async function createDataset(data: Partial<Dataset>) {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  const newDataset: Dataset = {
    id: uuidv4(),
    name: data.name || "Untitled Dataset",
    description: data.description || "",
    type: data.type || "classification",
    category: data.category || "tabular",
    features: data.features || 10,
    samples: data.samples || 1000,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    metadata: data.metadata || {}
  };
  
  // Add the dataset to our mock storage
  mockDatasets.push(newDataset);
  
  console.log(`Data service: Created dataset "${newDataset.name}" with ID ${newDataset.id}`);
  console.log(`Data service: Total datasets now: ${mockDatasets.length}`);
  
  return newDataset;
} 