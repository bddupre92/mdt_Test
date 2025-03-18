"use server"

import { NextResponse } from "next/server"
import { getDatasetStatistics } from "../../statistics-service"

// Generate statistics for a dataset
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Access params in a way that's compatible with Next.js App Router
    const datasetId = params.id
    console.log(`Statistics route: Received request for dataset ID: ${datasetId}`)
    
    // Get statistics from the service
    const statistics = await getDatasetStatistics(datasetId)
    
    if (!statistics) {
      console.log(`Statistics route: No statistics found for dataset ${datasetId}`)
      return NextResponse.json(
        { error: "Dataset not found" },
        { status: 404 }
      )
    }
    
    console.log(`Statistics route: Successfully returning statistics for dataset ${datasetId}`)
    return NextResponse.json(statistics)
  } catch (error) {
    console.error(`Statistics route: Error generating statistics:`, error)
    return NextResponse.json(
      { error: "Failed to generate statistics" },
      { status: 500 }
    )
  }
} 