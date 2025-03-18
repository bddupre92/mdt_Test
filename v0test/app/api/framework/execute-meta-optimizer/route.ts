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
const VISUALIZATION_DIR = '/tmp/meta_optimizer_results';

// Ensure the directory exists
try {
  if (!fs.existsSync(VISUALIZATION_DIR)) {
    fs.mkdirSync(VISUALIZATION_DIR, { recursive: true });
  }
} catch (error: any) {
  console.error('Error creating meta_optimizer directory:', error);
}

/**
 * Execute a meta-optimizer command and return its output
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

    console.log(`Executing meta-optimizer command: ${command}`);
    
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

    // Build the full command for the meta-optimizer - don't include --output-dir as the script doesn't accept it
    const fullCommand = await buildPythonCommandServer("main_v2.py", command);
    console.log(`Built full command: ${fullCommand}`);

    // Execute the command and capture output
    const result = await executeCommandServer(
      fullCommand,
      { cwd: path.resolve(process.cwd(), '..') } // Run from project root
    );

    if (!result.success) {
      console.error('Meta-optimizer execution error:', result.error);
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

    // Look for generated visualization files in common output directories
    let visualizationFiles: VisualizationFile[] = [];
    const possibleOutputDirs = [
      path.resolve(process.cwd(), '..', 'results'),
      path.resolve(process.cwd(), '..', 'output'),
      path.resolve(process.cwd(), '..', 'visualizations'),
      path.resolve(process.cwd(), '..', 'figures'),
      '/tmp/meta_optimizer_output'
    ];
    
    try {
      // Try to extract output directory information from command output
      let extractedOutputDirs: string[] = [];
      
      // Look for patterns like "Output saved to: /path/to/output"
      const outputDirPatterns = [
        /Output saved to[:\s]+([^\s]+)/i,
        /Saving (results|output|data) to[:\s]+([^\s]+)/i,
        /Visualizations? saved to[:\s]+([^\s]+)/i,
        /Writing (output|results) to[:\s]+([^\s]+)/i,
        /Created (output|result) directory[:\s]+([^\s]+)/i
      ];
      
      for (const pattern of outputDirPatterns) {
        const match = result.output.match(pattern);
        if (match) {
          // Get the captured group (the path)
          const outputPath = match[1] || match[2];
          if (outputPath && !extractedOutputDirs.includes(outputPath)) {
            extractedOutputDirs.push(outputPath);
            console.log(`Extracted output path from command output: ${outputPath}`);
            
            // Add to possible output dirs if it exists
            if (fs.existsSync(outputPath)) {
              possibleOutputDirs.push(outputPath);
            }
          }
        }
      }
      
      // Also look for file paths mentioned in the output
      const filePatterns = [
        /(Generated|Created|Saved) .*? (file|image)[:\s]+([^\s]+\.(png|jpg|svg))/gi,
        /Plot saved to[:\s]+([^\s]+\.(png|jpg|svg))/gi,
        /Figure saved to[:\s]+([^\s]+\.(png|jpg|svg))/gi
      ];
      
      const mentionedFiles: string[] = [];
      
      for (const pattern of filePatterns) {
        let match;
        while ((match = pattern.exec(result.output)) !== null) {
          const filePath = match[1] || match[3];
          if (filePath && !mentionedFiles.includes(filePath)) {
            mentionedFiles.push(filePath);
            console.log(`Found file mentioned in output: ${filePath}`);
            
            // Try to copy the file if it exists
            if (fs.existsSync(filePath)) {
              const fileName = path.basename(filePath);
              const destPath = path.join(runDir, fileName);
              
              try {
                fs.copyFileSync(filePath, destPath);
                
                // Add to visualization files
                visualizationFiles.push({
                  name: fileName,
                  path: destPath,
                  url: `/api/file?path=${encodeURIComponent(destPath)}`
                });
              } catch (err) {
                console.warn(`Failed to copy file ${filePath}:`, err);
              }
            }
          }
        }
      }
    
      // After extracting possible output directories from command output, check them
      // Copy output files from possible locations to our managed directory
      for (const outputDir of possibleOutputDirs) {
        if (fs.existsSync(outputDir)) {
          console.log(`Checking for visualization files in: ${outputDir}`);
          
          const files = fs.readdirSync(outputDir)
            .filter(file => file.endsWith('.png') || file.endsWith('.svg') || file.endsWith('.jpg'));
          
          if (files.length > 0) {
            console.log(`Found ${files.length} visualization files in ${outputDir}`);
            
            for (const file of files) {
              const srcPath = path.join(outputDir, file);
              const destPath = path.join(runDir, file);
              fs.copyFileSync(srcPath, destPath);
            }
            
            // Add to visualization files array
            const newFiles = files.map(file => ({
              name: file,
              path: path.join(runDir, file),
              url: `/api/file?path=${encodeURIComponent(path.join(runDir, file))}`
            }));
            
            visualizationFiles = [...visualizationFiles, ...newFiles];
          }
        }
      }
      
      // If no files found, check in current directory
      if (visualizationFiles.length === 0) {
        const cwd = path.resolve(process.cwd(), '..');
        console.log(`Checking current directory for visualization files: ${cwd}`);
        
        const files = fs.readdirSync(cwd)
          .filter(file => file.endsWith('.png') || file.endsWith('.svg') || file.endsWith('.jpg'));
        
        if (files.length > 0) {
          console.log(`Found ${files.length} visualization files in current directory`);
          
          for (const file of files) {
            const srcPath = path.join(cwd, file);
            const destPath = path.join(runDir, file);
            fs.copyFileSync(srcPath, destPath);
          }
          
          // Add to visualization files array
          visualizationFiles = files.map(file => ({
            name: file,
            path: path.join(runDir, file),
            url: `/api/file?path=${encodeURIComponent(path.join(runDir, file))}`
          }));
        }
      }
    } catch (err) {
      console.warn('Error handling visualization files:', err);
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