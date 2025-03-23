import { NextResponse } from 'next/server';

// Mock saved runs data
const mockSavedRuns = [
  {
    id: '1686753601234',
    function: 'differential_evolution',
    module: 'optimization',
    timestamp: '2023-06-14T12:00:01.234Z',
    parameters: {
      'population_size': '30',
      'F': '0.8',
      'CR': '0.5',
      'max_iterations': '100',
      'bounds': '[-5, 5]',
      'objective_function': 'sphere'
    }
  },
  {
    id: '1686840001234',
    function: 'simulated_annealing',
    module: 'optimization',
    timestamp: '2023-06-15T12:00:01.234Z',
    parameters: {
      'initial_temp': '100.0',
      'cooling_rate': '0.95',
      'max_iterations': '100',
      'bounds': '[-5, 5]',
      'objective_function': 'rastrigin'
    }
  },
  {
    id: '1686926401234',
    function: 'plot_convergence',
    module: 'visualization',
    timestamp: '2023-06-16T12:00:01.234Z',
    parameters: {
      'algorithm': 'differential_evolution',
      'show_percentiles': 'true',
      'save_to_file': 'false',
      'filename': 'convergence.png'
    }
  }
];

export async function GET() {
  try {
    // Simulate a delay to mimic a real API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return NextResponse.json(mockSavedRuns);
  } catch (error) {
    console.error('Error fetching saved runs:', error);
    return NextResponse.json({ error: 'Failed to fetch saved runs' }, { status: 500 });
  }
} 