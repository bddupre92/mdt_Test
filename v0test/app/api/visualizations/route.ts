import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    // Extract the file name from the search parameters
    const searchParams = request.nextUrl.searchParams;
    const fileName = searchParams.get('file');

    if (!fileName) {
      return NextResponse.json(
        { error: 'File name is required' },
        { status: 400 }
      );
    }

    // Define the directory where the visualizations are stored
    // This would typically be a directory in your project where you store results
    const visualizationsDirectory = path.resolve(process.cwd(), '../../results/paper_visuals');
    
    // Find the most recent results directory (they're timestamped)
    const directories = fs.readdirSync(visualizationsDirectory)
      .filter(dir => fs.statSync(path.join(visualizationsDirectory, dir)).isDirectory())
      .filter(dir => /^\d{8}_\d{6}$/.test(dir)) // Filter for timestamp directories
      .sort()
      .reverse(); // Most recent first

    if (directories.length === 0) {
      return NextResponse.json(
        { error: 'No visualization results found' },
        { status: 404 }
      );
    }

    const mostRecentDir = directories[0];
    const visualizationsPath = path.join(visualizationsDirectory, mostRecentDir, 'visualizations');
    const filePath = path.join(visualizationsPath, fileName);

    // Check if the file exists
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { error: 'Visualization file not found' },
        { status: 404 }
      );
    }

    // Read the file as a buffer
    const fileBuffer = fs.readFileSync(filePath);
    
    // Determine the content type based on file extension
    const ext = path.extname(fileName).toLowerCase();
    let contentType = 'application/octet-stream';
    
    if (ext === '.png') {
      contentType = 'image/png';
    } else if (ext === '.jpg' || ext === '.jpeg') {
      contentType = 'image/jpeg';
    } else if (ext === '.svg') {
      contentType = 'image/svg+xml';
    }

    // Return the file with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `inline; filename="${fileName}"`,
      },
    });
  } catch (error) {
    console.error('Error serving visualization:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// For listing available visualizations
export async function POST() {
  try {
    const visualizationsDirectory = path.resolve(process.cwd(), '../../results/paper_visuals');
    
    // Find the most recent results directory
    const directories = fs.readdirSync(visualizationsDirectory)
      .filter(dir => fs.statSync(path.join(visualizationsDirectory, dir)).isDirectory())
      .filter(dir => /^\d{8}_\d{6}$/.test(dir))
      .sort()
      .reverse();

    if (directories.length === 0) {
      return NextResponse.json({ files: [] });
    }

    const mostRecentDir = directories[0];
    const visualizationsPath = path.join(visualizationsDirectory, mostRecentDir, 'visualizations');
    
    // Check if the directory exists
    if (!fs.existsSync(visualizationsPath)) {
      return NextResponse.json({ files: [] });
    }

    // Get all files in the directory
    const files = fs.readdirSync(visualizationsPath)
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return ['.png', '.jpg', '.jpeg', '.svg'].includes(ext);
      })
      .map(file => ({
        name: file,
        path: `/api/visualizations?file=${file}`,
        lastModified: fs.statSync(path.join(visualizationsPath, file)).mtime,
      }));

    return NextResponse.json({ files });
  } catch (error) {
    console.error('Error listing visualizations:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 