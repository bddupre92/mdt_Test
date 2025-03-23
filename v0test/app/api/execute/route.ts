import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { executeCommandServer, buildPythonCommandServer } from '@/lib/utils/server-script-executor';

interface VisualizationFile {
  name: string;
  path: string;
  url: string;
}

// Create visualization output directory
const VISUALIZATION_DIR = '/tmp/metaopt_visualizations';

// Ensure the directory exists
try {
  if (!fs.existsSync(VISUALIZATION_DIR)) {
    fs.mkdirSync(VISUALIZATION_DIR, { recursive: true });
  }
} catch (error: any) {
  console.error('Error creating visualization directory:', error);
}

/**
 * Execute a command and return its output
 */
export async function POST(request: Request) {
  try {
    const { command } = await request.json();
    
    if (!command) {
      return NextResponse.json(
        { success: false, error: 'Command is required' },
        { status: 400 }
      );
    }

    console.log(`Executing command: ${command}`);
    
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

    // Add the output directory to the command
    const commandWithOutput = 
      command.includes('--output-dir') 
        ? command 
        : `${command} --output-dir=${runDir}`;

    // Execute the command and capture output
    const result = await executeCommandServer(
      commandWithOutput,
      { cwd: path.resolve(process.cwd(), '..') } // Run from project root
    );

    if (!result.success) {
      console.error('Command execution error:', result.error);
      console.error('stderr:', result.stderr);
      return NextResponse.json(
        {
          success: false,
          error: result.error,
          stderr: result.stderr,
          stdout: result.output,
        },
        { status: 500 }
      );
    }

    // Look for generated visualization files
    let visualizationFiles: VisualizationFile[] = [];
    try {
      visualizationFiles = fs.readdirSync(runDir)
        .filter(file => file.endsWith('.png') || file.endsWith('.svg') || file.endsWith('.jpg'))
        .map(file => ({
          name: file,
          path: path.join(runDir, file),
          url: `/api/file?path=${encodeURIComponent(path.join(runDir, file))}`
        }));
    } catch (err) {
      console.warn('Error reading visualization directory:', err);
    }

    // Parse JSON from stdout if possible
    let jsonResult = null;
    try {
      // Look for JSON content in stdout
      const jsonMatch = result.output.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        jsonResult = JSON.parse(jsonMatch[0]);
      }
    } catch (jsonError) {
      console.warn('Could not parse JSON from output:', jsonError);
    }

    return NextResponse.json({
      success: true,
      output: result.output,
      visualizationFiles,
      result: jsonResult,
      outputDir: runDir
    });
  } catch (error: any) {
    console.error('Error processing request:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
} 