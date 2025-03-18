// In-memory storage for generated datasets
// In a real application, this would be stored in a database
export const mockDatasets = [
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