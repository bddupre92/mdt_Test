// Simple test script for dataset API functions
import fetch from 'node-fetch';

// Base URL for API requests
const API_BASE_URL = 'http://localhost:3000';

// Sleep function to wait for specified milliseconds
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Test dataset generation and retrieval
async function testDatasetFlow() {
  console.log('Starting dataset API test...');
  
  try {
    // Step 1: Generate a synthetic dataset
    console.log('\n1. Generating synthetic dataset...');
    const dataset = await generateSyntheticDataset({
      name: 'Test Dataset ' + Date.now(),
      description: 'A test dataset created by the API test script',
      type: 'classification',
      category: 'tabular',
      features: 15,
      samples: 2000,
      noise: 0.1,
      complexity: 'medium',
      missingValues: 0.05,
      outlierPercentage: 0.02,
      classBalance: 'balanced',
      numClasses: 3
    });
    
    console.log(`Dataset created with ID: ${dataset.id}`);
    console.log('Dataset name:', dataset.name);
    
    // Step 2: Wait a bit to ensure dataset is processed
    console.log('\n2. Waiting for dataset to be processed...');
    await sleep(2000);
    
    // Step 3: Fetch the dataset details
    console.log('\n3. Fetching dataset details...');
    const details = await fetchDatasetById(dataset.id);
    console.log('Dataset details retrieved:', details ? 'Success' : 'Failed');
    if (details) {
      console.log(`Name: ${details.name}, Type: ${details.type}, Features: ${details.features}`);
    }
    
    // Step 4: Fetch all datasets to verify
    console.log('\n4. Fetching all datasets to verify creation...');
    const allDatasets = await fetchAllDatasets();
    const datasetExists = allDatasets.some(d => d.id === dataset.id);
    console.log(`Dataset exists in list: ${datasetExists ? 'Yes' : 'No'}`);
    
    // Step 5: Fetch dataset statistics
    console.log('\n5. Fetching dataset statistics...');
    const statistics = await fetchDatasetStatistics(dataset.id);
    console.log('Statistics retrieved:', statistics ? 'Success' : 'Failed');
    if (statistics) {
      console.log(`Row count: ${statistics.summary.rowCount}, Column count: ${statistics.summary.columnCount}`);
    }
    
    console.log('\nTest completed successfully!');
  } catch (error) {
    console.error('Test failed with error:', error);
  }
}

// API Functions

async function generateSyntheticDataset(params) {
  const response = await fetch(`${API_BASE_URL}/api/datasets/synthetic`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to generate dataset: ${response.status} - ${errorText}`);
  }
  
  return response.json();
}

async function fetchAllDatasets() {
  const response = await fetch(`${API_BASE_URL}/api/datasets`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch datasets: ${response.status}`);
  }
  
  const data = await response.json();
  return data.datasets;
}

async function fetchDatasetById(id) {
  const response = await fetch(`${API_BASE_URL}/api/datasets/${id}`);
  
  if (!response.ok) {
    if (response.status === 404) {
      console.warn(`Dataset with ID ${id} not found`);
      return null;
    }
    throw new Error(`Failed to fetch dataset: ${response.status}`);
  }
  
  return response.json();
}

async function fetchDatasetStatistics(id) {
  const response = await fetch(`${API_BASE_URL}/api/datasets/${id}/statistics`);
  
  if (!response.ok) {
    if (response.status === 404) {
      console.warn(`Statistics for dataset ${id} not found`);
      return null;
    }
    throw new Error(`Failed to fetch statistics: ${response.status}`);
  }
  
  return response.json();
}

// Run the test
testDatasetFlow(); 