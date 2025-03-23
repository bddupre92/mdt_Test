"use client"

import React, { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { WorkflowData, getStepStatus } from '@/lib/utils/workflow-validation';

export type WorkflowStep = 
  | 'dataset-selection' 
  | 'algorithm-configuration' 
  | 'execution' 
  | 'results';

export interface WorkflowStepInfo {
  id: WorkflowStep;
  title: string;
  description: string;
  isCompleted: boolean;
  isActive: boolean;
}

interface WorkflowStepperProps {
  steps: WorkflowStepInfo[];
  currentStepIndex: number;
  workflowData: Partial<WorkflowData>;
  onStepChange: (index: number) => void;
  onNextStep: () => void;
  onPrevStep: () => void;
  className?: string;
}

export const WorkflowStepper: React.FC<WorkflowStepperProps> = ({
  steps,
  currentStepIndex,
  workflowData,
  onStepChange,
  onNextStep,
  onPrevStep,
  className,
}) => {
  const [stepStatuses, setStepStatuses] = useState<Record<string, 'incomplete' | 'in-progress' | 'complete' | 'error'>>({});

  useEffect(() => {
    const updateStepStatuses = async () => {
      const statuses: Record<string, 'incomplete' | 'in-progress' | 'complete' | 'error'> = {};
      
      for (const step of steps) {
        statuses[step.id] = await getStepStatus(step.id, workflowData);
      }
      
      setStepStatuses(statuses);
    };
    
    updateStepStatuses();
  }, [steps, workflowData]);
  
  const handleStepClick = (index: number) => {
    const targetStep = steps[index];
    const currentStepId = steps[currentStepIndex].id;
    
    // Only allow navigation to completed steps or the next available step
    if (stepStatuses[targetStep.id] === 'complete' || index === currentStepIndex + 1) {
      onStepChange(index);
    }
  };
  
  const isNextStepAvailable = () => {
    if (currentStepIndex >= steps.length - 1) return false;
    
    const currentStepId = steps[currentStepIndex].id;
    return stepStatuses[currentStepId] === 'complete';
  };

  return (
    <div className={cn('w-full space-y-8', className)}>
      <div className="relative flex justify-between">
        {/* Progress Bar */}
        <div className="absolute top-1/2 h-0.5 w-full -translate-y-1/2 bg-gray-200">
          <div
            className="h-full bg-blue-600 transition-all duration-300"
            style={{
              width: `${(currentStepIndex / (steps.length - 1)) * 100}%`,
            }}
          />
        </div>

        {/* Steps */}
        {steps.map((step, index) => (
          <div
            key={step.id}
            className="relative flex flex-col items-center"
            onClick={() => handleStepClick(index)}
          >
            {/* Step Circle */}
            <div
              className={cn(
                'flex h-10 w-10 cursor-pointer items-center justify-center rounded-full border-2 transition-colors duration-300',
                {
                  'border-green-600 bg-green-600 text-white': stepStatuses[step.id] === 'complete',
                  'border-blue-600 bg-blue-600 text-white': stepStatuses[step.id] === 'in-progress' && index === currentStepIndex,
                  'border-yellow-500 bg-yellow-500 text-white': stepStatuses[step.id] === 'in-progress' && index !== currentStepIndex,
                  'border-red-500 bg-red-500 text-white': stepStatuses[step.id] === 'error',
                  'border-gray-300 bg-white': stepStatuses[step.id] === 'incomplete',
                  'cursor-not-allowed opacity-60': 
                    stepStatuses[step.id] === 'incomplete' && 
                    index > currentStepIndex && 
                    index !== currentStepIndex + 1,
                }
              )}
            >
              {stepStatuses[step.id] === 'complete' ? (
                <CheckIcon className="h-5 w-5" />
              ) : stepStatuses[step.id] === 'error' ? (
                <ErrorIcon className="h-5 w-5" />
              ) : (
                <span className="text-sm font-medium">{index + 1}</span>
              )}
            </div>

            {/* Step Title */}
            <div className="absolute mt-12 text-center">
              <h3
                className={cn('text-sm font-medium', {
                  'text-blue-600': index === currentStepIndex,
                  'text-green-600': stepStatuses[step.id] === 'complete' && index !== currentStepIndex,
                  'text-red-500': stepStatuses[step.id] === 'error',
                  'text-gray-500': stepStatuses[step.id] === 'incomplete' && index !== currentStepIndex,
                })}
              >
                {step.title}
              </h3>
              <p className="mt-0.5 text-xs text-gray-500 max-w-[120px]">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
      
      {/* Navigation Controls */}
      <div className="flex justify-between pt-8">
        <Button 
          variant="outline" 
          onClick={onPrevStep} 
          disabled={currentStepIndex === 0}
        >
          Previous
        </Button>
        
        <div className="flex items-center">
          <StepStatusIndicator 
            status={stepStatuses[steps[currentStepIndex]?.id] || 'incomplete'} 
          />
        </div>
        
        <Button 
          onClick={onNextStep} 
          disabled={!isNextStepAvailable()}
        >
          {currentStepIndex >= steps.length - 1 ? 'Finish' : 'Next'}
        </Button>
      </div>
    </div>
  );
};

const StepStatusIndicator: React.FC<{ status: 'incomplete' | 'in-progress' | 'complete' | 'error' }> = ({ status }) => {
  const statusMessages = {
    'incomplete': 'This step needs attention',
    'in-progress': 'In progress - complete required fields',
    'complete': 'Step completed successfully',
    'error': 'Error - please resolve issues',
  };
  
  const statusColors = {
    'incomplete': 'text-gray-500',
    'in-progress': 'text-blue-600',
    'complete': 'text-green-600',
    'error': 'text-red-500',
  };
  
  return (
    <span className={`text-sm ${statusColors[status]}`}>
      {statusMessages[status]}
    </span>
  );
};

const CheckIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M5 13l4 4L19 7"
    />
  </svg>
);

const ErrorIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

export default WorkflowStepper; 