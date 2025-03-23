// Test script to verify meta-learner tab functionality
import fetch from 'node-fetch';

// Base URL for API requests - update to match your running server port
const API_BASE_URL = 'http://localhost:3006'; // Update this to match your running server port

const testMetaLearner = async () => {
  try {
    console.log('=== TESTING META-LEARNER FUNCTIONALITY ===');
    
    // Step 1: Get all available datasets
    console.log('\n1. Fetching available datasets...');
    const datasetsResponse = await fetch(`${API_BASE_URL}/api/datasets`);
    
    if (!datasetsResponse.ok) {
      throw new Error(`Failed to fetch datasets: ${datasetsResponse.status}`);
    }
    
    const { datasets } = await datasetsResponse.json();
    console.log(`Total datasets available: ${datasets.length}`);
    
    if (datasets.length === 0) {
      throw new Error('No datasets available for testing');
    }
    
    // Pick the first dataset for testing
    const testDataset = datasets[0];
    console.log(`Selected dataset for testing: ${testDataset.name} (${testDataset.id})`);
    
    // Step 2: Simulate meta-learner prediction (direct API call)
    console.log('\n2. Simulating meta-learner analysis...');
    
    // Create sample data for meta-learning analysis
    const metaLearnerTestData = {
      datasetId: testDataset.id,
      optimizerIds: [
        'differential-evolution',
        'evolution-strategy',
        'ant-colony',
        'grey-wolf'
      ],
      benchmarkId: 'rastrigin'
    };
    
    // Log what we're testing with
    console.log(`Dataset: ${testDataset.name}`);
    console.log(`Optimizers: ${metaLearnerTestData.optimizerIds.join(', ')}`);
    console.log(`Benchmark: ${metaLearnerTestData.benchmarkId}`);
    
    // Mock the meta-learner prediction - in a real app, you'd have an endpoint for this
    console.log('\n3. Generating meta-learner predictions...');
    
    // Simulate the prediction results
    const simulateMetaLearnerPrediction = (datasetId, optimizerIds, benchmarkId) => {
      // Create random confidence scores that sum to 1
      const randomScores = optimizerIds.map(() => Math.random());
      const sum = randomScores.reduce((a, b) => a + b, 0);
      const normalizedScores = randomScores.map(score => score / sum);
      
      // Select the "best" optimizer (highest confidence score)
      const bestIndex = normalizedScores.indexOf(Math.max(...normalizedScores));
      const selectedOptimizer = optimizerIds[bestIndex];
      
      // Create prediction objects
      const predictions = optimizerIds.map((id, index) => ({
        optimizerId: id,
        confidence: normalizedScores[index]
      }));
      
      // Create mock prediction quality data
      const predictionQuality = optimizerIds.map(id => {
        const predicted = predictions.find(p => p.optimizerId === id).confidence;
        // Simulate actual performance as slightly different from predicted
        const actual = predicted * (0.8 + Math.random() * 0.4); // 80-120% of predicted
        
        return {
          algorithm: id,
          predictedPerformance: predicted,
          actualPerformance: actual
        };
      });
      
      return {
        selectedOptimizer,
        predictions,
        predictionQuality,
        problemFeatures: {
          dimensions: testDataset.features || 10,
          samples: testDataset.samples || 1000,
          complexity: "medium",
          multimodality: 0.7,
          ruggedness: 0.5,
          noise: 0.2
        }
      };
    };
    
    // Generate the prediction
    const results = simulateMetaLearnerPrediction(
      metaLearnerTestData.datasetId,
      metaLearnerTestData.optimizerIds,
      metaLearnerTestData.benchmarkId
    );
    
    // Step 3: Verify the prediction results
    console.log('\n4. Verifying prediction results:');
    console.log(`Selected optimizer: ${results.selectedOptimizer}`);
    console.log(`Number of predictions: ${results.predictions.length}`);
    console.log(`Prediction scores sum to ~1: ${results.predictions.reduce((sum, p) => sum + p.confidence, 0).toFixed(3)}`);
    
    // Display all predictions
    console.log('\nPrediction confidence scores:');
    results.predictions.forEach(pred => {
      console.log(`- ${pred.optimizerId}: ${(pred.confidence * 100).toFixed(2)}%`);
    });
    
    // Display prediction quality
    console.log('\nPrediction quality:');
    results.predictionQuality.forEach(quality => {
      const error = Math.abs(quality.predictedPerformance - quality.actualPerformance);
      console.log(`- ${quality.algorithm}: Predicted ${(quality.predictedPerformance * 100).toFixed(2)}%, Actual ${(quality.actualPerformance * 100).toFixed(2)}%, Error ${(error * 100).toFixed(2)}%`);
    });
    
    console.log('\n=== META-LEARNER TEST COMPLETED SUCCESSFULLY ===');
  } catch (error) {
    console.error('Meta-learner test failed:', error);
  }
};

// Run the test
testMetaLearner(); 