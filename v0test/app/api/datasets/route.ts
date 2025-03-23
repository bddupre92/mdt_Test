"use server"

import { NextResponse } from "next/server"
import { getAllDatasets, createDataset } from "./data-service"

// Handler for GET requests to fetch all datasets
export async function GET() {
  // Get datasets from the service
  const datasets = await getAllDatasets()
  
  return NextResponse.json({ datasets })
}

// Handler for POST requests to create a new dataset
export async function POST(request: Request) {
  try {
    const data = await request.json()
    
    // Create dataset using the service
    const newDataset = await createDataset(data)
    
    return NextResponse.json(newDataset)
  } catch (error) {
    console.error("Error creating dataset:", error)
    return NextResponse.json(
      { error: "Failed to create dataset" },
      { status: 500 }
    )
  }
} 