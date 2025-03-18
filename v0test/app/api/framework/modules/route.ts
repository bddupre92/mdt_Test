import { NextResponse } from 'next/server';

// This is a mock list of modules
// In a real application, you would fetch this from your actual framework
const mockModules = [
  'optimization',
  'visualization',
  'preprocessing',
  'statistics',
  'models'
];

export async function GET() {
  try {
    // Simulate a delay to mimic a real API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return NextResponse.json(mockModules);
  } catch (error) {
    console.error('Error fetching modules:', error);
    return NextResponse.json({ error: 'Failed to fetch modules' }, { status: 500 });
  }
} 