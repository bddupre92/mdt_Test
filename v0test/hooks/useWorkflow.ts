import { useState, useCallback } from 'react';
import type { WorkflowStep } from '@/components/workflow/WorkflowStepper';

interface UseWorkflowProps {
  initialSteps: Omit<WorkflowStep, 'isActive' | 'isCompleted'>[];
}

export const useWorkflow = ({ initialSteps }: UseWorkflowProps) => {
  const [steps, setSteps] = useState<WorkflowStep[]>(() =>
    initialSteps.map((step, index) => ({
      ...step,
      isActive: index === 0,
      isCompleted: false,
    }))
  );
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const goToStep = useCallback(async (index: number) => {
    if (index < 0 || index >= steps.length) return;

    // Check if we can move to this step
    const currentStep = steps[currentStepIndex];
    if (currentStep.validationFn) {
      const isValid = await currentStep.validationFn();
      if (!isValid) return;
    }

    setSteps((prevSteps) =>
      prevSteps.map((step, i) => ({
        ...step,
        isActive: i === index,
        isCompleted: i < currentStepIndex || (i === currentStepIndex && currentStep.validationFn ? true : false),
      }))
    );
    setCurrentStepIndex(index);
  }, [steps, currentStepIndex]);

  const nextStep = useCallback(async () => {
    await goToStep(currentStepIndex + 1);
  }, [currentStepIndex, goToStep]);

  const previousStep = useCallback(async () => {
    await goToStep(currentStepIndex - 1);
  }, [currentStepIndex, goToStep]);

  const completeStep = useCallback((stepId: string) => {
    setSteps((prevSteps) =>
      prevSteps.map((step) =>
        step.id === stepId ? { ...step, isCompleted: true } : step
      )
    );
  }, []);

  return {
    steps,
    currentStepIndex,
    goToStep,
    nextStep,
    previousStep,
    completeStep,
  };
};

export default useWorkflow; 