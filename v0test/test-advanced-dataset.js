// Test script to diagnose the advanced dataset generation issue
import fetch from 'node-fetch';

// Base URL for API requests - update to match your current port
const API_BASE_URL = 'http://localhost:3005'; // Update this to match your running server port

// Sleep function to wait for specified milliseconds
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Test advanced dataset generation and statistics
async function testAdvancedDatasetGeneration() {
  console.log('=== TESTING ADVANCED DATASET GENERATION ===');
  
  try {
    // Step 1: Generate an advanced synthetic dataset
    console.log('\n1. Generating advanced synthetic dataset...');
    const advancedDataset = await generateSyntheticDataset({
      name: 'Advanced Test Dataset ' + Date.now(),
      description: 'A complex dataset with advanced settings',
      type: 'classification',
      category: 'tabular',
      features: 25,
      samples: 5000,
      noise: 0.2,
      complexity: 'high',
      missingValues: 0.1,
      outlierPercentage: 0.05,
      classBalance: 'imbalanced',
      numClasses: 5
    });
    
    console.log(`Dataset created with ID: ${advancedDataset.id}`);
    console.log(`Dataset details: ${JSON.stringify(advancedDataset, null, 2)}`);
    
    // Step 2: Verify dataset exists in the list
    console.log('\n2. Verifying dataset exists in the list...');
    await sleep(1000); // Wait a second to ensure dataset is saved
    
    const allDatasets = await fetchAllDatasets();
    console.log(`Total datasets: ${allDatasets.length}`);
    console.log(`All dataset IDs: ${allDatasets.map(d => d.id).join(', ')}`);
    
    const datasetExists = allDatasets.some(d => d.id === advancedDataset.id);
    console.log(`Dataset exists in list: ${datasetExists ? 'Yes ✅' : 'No ❌'}`);
    
    if (!datasetExists) {
      console.error('ERROR: The newly created dataset is not in the datasets list!');
    }
    
    // Step 3: Fetch the dataset by ID directly
    console.log('\n3. Fetching dataset by ID directly...');
    const datasetById = await fetchDatasetById(advancedDataset.id);
    
    if (datasetById) {
      console.log(`Successfully retrieved dataset by ID: ${datasetById.id} ✅`);
      console.log(`Dataset name: ${datasetById.name}`);
    } else {
      console.error(`ERROR: Could not fetch dataset by ID: ${advancedDataset.id} ❌`);
    }
    
    // Step 4: Attempt to get statistics (this is where the error occurs)
    console.log('\n4. Fetching dataset statistics...');
    
    // Wait a bit more to ensure all processing is complete
    console.log('Waiting 3 seconds before fetching statistics...');
    await sleep(3000);
    
    const statistics = await fetchDatasetStatistics(advancedDataset.id);
    
    if (statistics) {
      console.log('Successfully retrieved statistics ✅');
      console.log(`Row count: ${statistics.summary.rowCount}, Column count: ${statistics.summary.columnCount}`);
    } else {
      console.error(`ERROR: Could not fetch statistics for dataset ID: ${advancedDataset.id} ❌`);
    }
    
    console.log('\nTest completed.');
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
testAdvancedDatasetGeneration(); 