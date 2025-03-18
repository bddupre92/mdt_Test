import { NextResponse } from 'next/server';

// Mock data for functions by module
const mockFunctions: Record<string, string[]> = {
  'optimization': [
    'differential_evolution',
    'simulated_annealing',
    'particle_swarm',
    'gray_wolf_optimizer',
    'genetic_algorithm'
  ],
  'visualization': [
    'plot_convergence',
    'plot_population',
    'plot_contour',
    'plot_surface',
    'plot_comparison'
  ],
  'preprocessing': [
    'normalize_data',
    'standardize_data',
    'handle_missing_values',
    'reduce_dimensions',
    'encode_categorical'
  ],
  'statistics': [
    'descriptive_stats',
    'correlation_analysis',
    'hypothesis_test',
    'anova',
    'regression_analysis'
  ],
  'models': [
    'linear_regression',
    'random_forest',
    'svm',
    'neural_network',
    'ensemble_methods'
  ]
};

export async function GET(
  request: Request,
  context: { params: { module: string } }
) {
  try {
    // Get the module directly from context.params
    const moduleParam = context.params.module;

    // Simulate a delay to mimic a real API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // Return the functions for the requested module
    if (moduleParam in mockFunctions) {
      return NextResponse.json(mockFunctions[moduleParam]);
    } else {
      // Return an empty array if the module doesn't exist
      return NextResponse.json([]);
    }
  } catch (error) {
    console.error('Error fetching functions:', error);
    return NextResponse.json({ error: 'Failed to fetch functions' }, { status: 500 });
  }
} 