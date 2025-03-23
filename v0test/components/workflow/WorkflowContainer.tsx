"use client"

import React, { useState } from 'react';
import { WorkflowStepper, WorkflowStep, WorkflowStepInfo } from './WorkflowStepper';
import { WorkflowData } from '@/lib/utils/workflow-validation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DatasetSelector } from '@/components/dataset-selector';
import { AlgorithmSelector } from './AlgorithmSelector';
import { ExecutionControl } from './ExecutionControl';

// Type definitions from the workflow validation file
import { 
  DatasetSelectionData, 
  AlgorithmConfigurationData, 
  ExecutionData,
  ResultsData
} from '@/lib/utils/workflow-validation';

// Placeholder component for results step
const ResultsStep: React.FC<{ updateData: (data: ResultsData) => void }> = ({ updateData }) => {
  // In a real implementation, this would be the proper results component
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Results Analysis</h2>
      <p className="text-gray-600">View and analyze the results of your algorithm execution.</p>
      
      <div className="p-4 border rounded-md bg-gray-50 text-center">
        <p>Results visualization will be implemented in Phase 2.</p>
      </div>
      
      {/* Dummy update for demo purposes */}
      <button
        className="px-4 py-2 bg-blue-600 text-white rounded"
        onClick={() => updateData({
          metrics: { accuracy: 0.92, precision: 0.89, recall: 0.95, f1: 0.92 },
          visualizations: ['confusion-matrix', 'roc-curve'],
          recommendations: ['Increase model capacity for better performance'],
          exportFormats: ['csv', 'json']
        })}
      >
        Load Sample Results
      </button>
    </div>
  );
};

export const WorkflowContainer: React.FC = () => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [workflowData, setWorkflowData] = useState<Partial<WorkflowData>>({});
  
  const workflowSteps: WorkflowStepInfo[] = [
    {
      id: 'dataset-selection',
      title: 'Dataset Selection',
      description: 'Select and configure the dataset',
      isCompleted: false,
      isActive: currentStepIndex === 0,
    },
    {
      id: 'algorithm-configuration',
      title: 'Algorithm Configuration',
      description: 'Configure algorithm parameters',
      isCompleted: false,
      isActive: currentStepIndex === 1,
    },
    {
      id: 'execution',
      title: 'Execution',
      description: 'Execute the algorithm',
      isCompleted: false,
      isActive: currentStepIndex === 2,
    },
    {
      id: 'results',
      title: 'Results',
      description: 'View and analyze results',
      isCompleted: false,
      isActive: currentStepIndex === 3,
    },
  ];
  
  const handleStepChange = (index: number) => {
    setCurrentStepIndex(index);
  };
  
  const handleNextStep = () => {
    if (currentStepIndex < workflowSteps.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
    }
  };
  
  const handlePrevStep = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };
  
  const updateStepData = <T extends keyof WorkflowData>(
    step: T, 
    data: WorkflowData[T]
  ) => {
    setWorkflowData(prev => ({
      ...prev,
      [step]: data
    }));
  };
  
  // Render the appropriate step content based on the current step
  const renderStepContent = () => {
    switch (workflowSteps[currentStepIndex].id) {
      case 'dataset-selection':
        return (
          <DatasetSelector 
            onSelectDataset={(datasetId) => {
              const selectedDataset: DatasetSelectionData = {
                datasetId,
                datasetName: `Dataset ${datasetId}`, // This would come from actual dataset metadata
                datasetType: 'custom' // Changed from 'tabular' to one of the allowed types
              };
              updateStepData('dataset-selection', selectedDataset);
            }}
            selectedDataset={workflowData['dataset-selection']?.datasetId}
          />
        );
      case 'algorithm-configuration':
        return (
          <AlgorithmSelector 
            value={workflowData['algorithm-configuration']} 
            onChange={(data) => updateStepData('algorithm-configuration', data)}
            datasetId={workflowData['dataset-selection']?.datasetId}
          />
        );
      case 'execution':
        return workflowData['dataset-selection'] && workflowData['algorithm-configuration'] ? (
          <ExecutionControl
            algorithmId={workflowData['algorithm-configuration'].algorithmId}
            datasetId={workflowData['dataset-selection'].datasetId}
            parameters={workflowData['algorithm-configuration'].parameters}
            onExecutionComplete={(executionId) => {
              updateStepData('execution', {
                executionId,
                status: 'completed',
                progress: 100,
                logs: ['Execution completed successfully']
              });
            }}
            onExecutionFailed={(error) => {
              updateStepData('execution', {
                executionId: `failed-${Date.now()}`,
                status: 'failed',
                progress: 0,
                logs: [`Execution failed: ${error}`]
              });
            }}
          />
        ) : (
          <div className="p-4 border rounded-md bg-red-50 text-red-600">
            Please complete the dataset selection and algorithm configuration steps first.
          </div>
        );
      case 'results':
        return (
          <ResultsStep 
            updateData={(data) => updateStepData('results', data)} 
          />
        );
      default:
        return null;
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <WorkflowStepper 
            steps={workflowSteps}
            currentStepIndex={currentStepIndex}
            workflowData={workflowData}
            onStepChange={handleStepChange}
            onNextStep={handleNextStep}
            onPrevStep={handlePrevStep}
          />
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-6">
          {renderStepContent()}
        </CardContent>
      </Card>
      
      {/* Debug Panel - Remove in production */}
      <Card className="bg-gray-100">
        <CardHeader>
          <CardTitle className="text-sm">Current Workflow Data (Debug)</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="text-xs overflow-auto max-h-40">
            {JSON.stringify(workflowData, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
};

export default WorkflowContainer; 