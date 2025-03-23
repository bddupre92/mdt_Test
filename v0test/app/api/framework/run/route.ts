import { NextResponse } from 'next/server';
import { executeCommand } from '@/lib/utils/command-execution';
import { buildPythonCommand } from '@/lib/utils/script-locator';
import path from 'path';
import fs from 'fs';

// Create visualization output directory
const VISUALIZATION_DIR = '/tmp/metaopt_visualizations';

// Ensure the directory exists
try {
  if (!fs.existsSync(VISUALIZATION_DIR)) {
    fs.mkdirSync(VISUALIZATION_DIR, { recursive: true });
  }
} catch (error) {
  console.error('Error creating visualization directory:', error);
}

export async function POST(request: Request) {
  try {
    const { module, function: functionName, parameters } = await request.json();
    
    console.log(`Running function ${functionName} from module ${module} with parameters:`, parameters);
    
    // Generate a timestamp-based subdirectory for this run
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const runDir = path.join(VISUALIZATION_DIR, timestamp);
    
    try {
      fs.mkdirSync(runDir, { recursive: true });
    } catch (error: any) {
      console.error(`Error creating run directory ${runDir}:`, error);
      return NextResponse.json(
        { success: false, error: `Failed to create output directory: ${error.message}` },
        { status: 500 }
      );
    }
    
    // Convert function and parameters to command format
    let command: string;
    let pythonArgs: string;

    switch (module) {
      case 'optimization':
        // For optimization commands
        const paramArgs = Object.entries(parameters)
          .map(([key, value]) => `--${key}=${value}`)
          .join(' ');
        
        // Special handling for baseline_comparison to generate visualizations
        if (functionName === 'baseline_comparison') {
          pythonArgs = `baseline_comparison ${paramArgs} --output-dir=${runDir} --visualize-comparison --visualize-convergence --visualize-algorithm-selection`;
        } else {
          pythonArgs = `run_optimizer --algorithm=${functionName} ${paramArgs} --output-dir=${runDir} --visualize`;
        }
        break;
      
      case 'analysis':
        // For analysis commands like drift_detection, etc.
        const analysisArgs = Object.entries(parameters)
          .map(([key, value]) => `--${key}=${value}`)
          .join(' ');
        pythonArgs = `${functionName} ${analysisArgs} --output-dir=${runDir} --generate-visualizations`;
        break;
      
      case 'meta_optimizer':
        // For SATzilla training and prediction
        const metaArgs = Object.entries(parameters)
          .map(([key, value]) => `--${key}=${value}`)
          .join(' ');
        pythonArgs = `train_satzilla ${metaArgs} --output-dir=${runDir} --visualize-features --visualize-importance`;
        break;
      
      default:
        // Generic case - direct pass-through
        const defaultArgs = Object.entries(parameters)
          .map(([key, value]) => `--${key}=${value}`)
          .join(' ');
        pythonArgs = `${functionName} ${defaultArgs} --output-dir=${runDir}`;
    }
    
    // Build the full command with the correct Python executable
    command = await buildPythonCommand("main_v2.py", pythonArgs);
    console.log("Full command:", command);
    
    // Execute the command and handle the response
    const result = await executeCommand(command);
    
    // Check for generated visualization files in the output directory
    let visualizationFiles: Array<{name: string, path: string, url: string}> = [];
    try {
      visualizationFiles = fs.readdirSync(runDir)
        .filter(file => file.endsWith('.png') || file.endsWith('.svg') || file.endsWith('.jpg'))
        .map(file => ({
          name: file,
          path: path.join(runDir, file),
          url: `/api/file?path=${encodeURIComponent(path.join(runDir, file))}`
        }));
      
      console.log(`Found ${visualizationFiles.length} visualization files in ${runDir}`);
    } catch (err) {
      console.warn('Error reading visualization directory:', err);
    }
    
    // Return success with visualizations
    return NextResponse.json({
      success: true,
      message: `Executed ${module}.${functionName}`,
      parameters,
      result,
      visualizationFiles,
      outputDir: runDir
    });
  } catch (error: any) {
    console.error('Error executing framework function:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: error.message 
      },
      { status: 500 }
    );
  }
} 