import React, { createContext, useContext } from 'react';
import { useWorkflow } from '@/hooks/useWorkflow';
import type { WorkflowStep } from '@/components/workflow/WorkflowStepper';

// Define the workflow steps for the optimization process
export const OPTIMIZATION_WORKFLOW_STEPS: Omit<WorkflowStep, 'isActive' | 'isCompleted'>[] = [
  {
    id: 'dataset-selection',
    title: 'Dataset Selection',
    description: 'Select and configure your dataset',
  },
  {
    id: 'algorithm-configuration',
    title: 'Algorithm Setup',
    description: 'Choose and configure optimization algorithms',
  },
  {
    id: 'execution',
    title: 'Execution',
    description: 'Run and monitor optimization process',
  },
  {
    id: 'results',
    title: 'Results',
    description: 'View and analyze results',
  },
];

interface WorkflowContextType {
  steps: WorkflowStep[];
  currentStepIndex: number;
  goToStep: (index: number) => Promise<void>;
  nextStep: () => Promise<void>;
  previousStep: () => Promise<void>;
  completeStep: (stepId: string) => void;
}

const WorkflowContext = createContext<WorkflowContextType | undefined>(undefined);

export const WorkflowProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const workflow = useWorkflow({
    initialSteps: OPTIMIZATION_WORKFLOW_STEPS,
  });

  return (
    <WorkflowContext.Provider value={workflow}>
      {children}
    </WorkflowContext.Provider>
  );
};

export const useWorkflowContext = () => {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error('useWorkflowContext must be used within a WorkflowProvider');
  }
  return context;
};

export default WorkflowContext; 