import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import mime from 'mime-types';

export async function GET(request: Request) {
  // Parse the URL and search parameters
  const url = new URL(request.url);
  const filePath = url.searchParams.get('path');
  const listDir = url.searchParams.get('list') === 'true';

  if (!filePath) {
    return NextResponse.json(
      { error: 'File path is required' },
      { status: 400 }
    );
  }

  try {
    // Normalize path to prevent directory traversal
    const normalizedPath = path.normalize(filePath).replace(/^(\.\.(\/|\\|$))+/, '');
    
    // Check if the path is a directory and list is requested
    if (listDir) {
      try {
        const stats = fs.statSync(normalizedPath);
        if (stats.isDirectory()) {
          const files = fs.readdirSync(normalizedPath);
          
          // Get details for each file
          const fileDetails = files.map(file => {
            const filePath = path.join(normalizedPath, file);
            const stats = fs.statSync(filePath);
            return {
              name: file,
              path: filePath,
              isDirectory: stats.isDirectory(),
              size: stats.size,
              modified: stats.mtime
            };
          });
          
          return NextResponse.json({ files: fileDetails });
        }
      } catch (error) {
        console.error(`Error listing directory: ${error}`);
      }
    }
    
    // Check if file exists
    if (!fs.existsSync(normalizedPath)) {
      // If file not found and it's a visualization path, check for timestamped subdirectories
      if (normalizedPath.includes('/metaopt_') && !normalizedPath.endsWith('.png')) {
        const baseDir = path.dirname(normalizedPath);
        if (fs.existsSync(baseDir)) {
          // Look for timestamp subdirectories and use the most recent one
          try {
            const dirs = fs.readdirSync(baseDir)
              .filter(dir => fs.statSync(path.join(baseDir, dir)).isDirectory())
              .filter(dir => /\d{4}-\d{2}-\d{2}T/.test(dir))
              .sort()
              .reverse();
              
            if (dirs.length > 0) {
              const latestDir = path.join(baseDir, dirs[0]);
              return NextResponse.redirect(new URL(`/api/file?path=${encodeURIComponent(latestDir)}&list=true`, request.url));
            }
          } catch (error) {
            console.error(`Error finding timestamp directories: ${error}`);
          }
        }
      }
      
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    // Read the file
    const file = fs.readFileSync(normalizedPath);
    
    // Determine the content type
    const contentType = mime.lookup(normalizedPath) || 'application/octet-stream';
    
    // Return the file with appropriate headers
    return new NextResponse(file, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `inline; filename="${path.basename(normalizedPath)}"`,
      },
    });
  } catch (error) {
    console.error(`Error serving file: ${error}`);
    return NextResponse.json(
      { error: 'Failed to serve file' },
      { status: 500 }
    );
  }
} 