import { NextResponse } from 'next/server';

// Mock data for function info
const mockFunctionInfo: Record<string, Record<string, any>> = {
  'optimization': {
    'differential_evolution': {
      name: 'differential_evolution',
      doc: 'Differential Evolution algorithm for global optimization.',
      module: 'optimization',
      parameters: {
        'population_size': 30,
        'F': 0.8,
        'CR': 0.5,
        'max_iterations': 100,
        'bounds': '[-5, 5]',
        'objective_function': 'sphere'
      }
    },
    'simulated_annealing': {
      name: 'simulated_annealing',
      doc: 'Simulated Annealing for global optimization problems.',
      module: 'optimization',
      parameters: {
        'initial_temp': 100.0,
        'cooling_rate': 0.95,
        'max_iterations': 100,
        'bounds': '[-5, 5]',
        'objective_function': 'sphere'
      }
    },
    'particle_swarm': {
      name: 'particle_swarm',
      doc: 'Particle Swarm Optimization algorithm.',
      module: 'optimization',
      parameters: {
        'num_particles': 30,
        'inertia_weight': 0.7,
        'cognitive_coeff': 1.5,
        'social_coeff': 1.5,
        'max_iterations': 100,
        'bounds': '[-5, 5]',
        'objective_function': 'sphere'
      }
    },
    'gray_wolf_optimizer': {
      name: 'gray_wolf_optimizer',
      doc: 'Gray Wolf Optimizer algorithm for global optimization.',
      module: 'optimization',
      parameters: {
        'num_wolves': 20,
        'max_iterations': 100,
        'bounds': '[-5, 5]',
        'objective_function': 'sphere'
      }
    },
    'genetic_algorithm': {
      name: 'genetic_algorithm',
      doc: 'Genetic Algorithm for optimization problems.',
      module: 'optimization',
      parameters: {
        'population_size': 50,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'max_generations': 100,
        'bounds': '[-5, 5]',
        'objective_function': 'sphere'
      }
    }
  },
  'visualization': {
    'plot_convergence': {
      name: 'plot_convergence',
      doc: 'Plot convergence history of an optimization algorithm.',
      module: 'visualization',
      parameters: {
        'algorithm': 'differential_evolution',
        'show_percentiles': true,
        'save_to_file': false,
        'filename': 'convergence.png'
      }
    },
    'plot_population': {
      name: 'plot_population',
      doc: 'Plot population distribution over generations/iterations.',
      module: 'visualization',
      parameters: {
        'algorithm': 'differential_evolution',
        'generation': -1,
        'dimensions': '0,1',
        'save_to_file': false,
        'filename': 'population.png'
      }
    }
  }
};

// Add some default info for other modules
const defaultInfo = {
  parameters: {
    'param1': 'value1',
    'param2': 10,
    'param3': true
  }
};

export async function GET(
  request: Request,
  context: { params: { module: string, function: string } }
) {
  try {
    // Get the module and function directly from context.params
    const moduleParam = context.params.module;
    const functionParam = context.params.function;

    // Simulate a delay to mimic a real API call
    await new Promise(resolve => setTimeout(resolve, 500));

    // Check if we have detailed info for this function
    if (
      moduleParam in mockFunctionInfo && 
      functionParam in mockFunctionInfo[moduleParam]
    ) {
      return NextResponse.json(mockFunctionInfo[moduleParam][functionParam]);
    } else {
      // Return generic info for functions we don't have details for
      return NextResponse.json({
        name: functionParam,
        doc: `Function ${functionParam} from module ${moduleParam}`,
        module: moduleParam,
        ...defaultInfo
      });
    }
  } catch (error) {
    console.error('Error fetching function info:', error);
    return NextResponse.json({ error: 'Failed to fetch function info' }, { status: 500 });
  }
} 