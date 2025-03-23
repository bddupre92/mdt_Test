import React from 'react';
import WorkflowContainer from '@/components/workflow/WorkflowContainer';

export default function WorkflowPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Pipeline Workflow</h1>
      <WorkflowContainer />
    </div>
  );
} 