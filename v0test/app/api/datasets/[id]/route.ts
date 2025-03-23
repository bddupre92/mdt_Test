"use server"

import { NextResponse } from "next/server"
import { getDatasetById } from "../data-service"

// Handler for GET requests to fetch a specific dataset by ID
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800))
    
    // Access params in a way that's compatible with Next.js App Router
    const datasetId = params.id
    console.log(`Dataset route: Fetching dataset with ID: ${datasetId}`)
    
    // Get the dataset from the service
    const dataset = await getDatasetById(datasetId)
    
    if (!dataset) {
      console.log(`Dataset route: Dataset with ID ${datasetId} not found`)
      return NextResponse.json(
        { error: "Dataset not found" },
        { status: 404 }
      )
    }
    
    console.log(`Dataset route: Successfully returning dataset: ${dataset.name} (${dataset.id})`)
    return NextResponse.json(dataset)
  } catch (error) {
    console.error(`Dataset route: Error fetching dataset:`, error)
    return NextResponse.json(
      { error: "Failed to fetch dataset" },
      { status: 500 }
    )
  }
}

// Handler for PATCH requests to update a dataset
export async function PATCH(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Get the dataset ID from params
    const datasetId = params.id
    
    // Implementation omitted for brevity
    // In a real application, this would update the dataset in the database
    
    return NextResponse.json({ 
      success: true, 
      message: `Dataset ${datasetId} updated` 
    })
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to update dataset" },
      { status: 500 }
    )
  }
}

// Handler for DELETE requests to delete a dataset
export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Get the dataset ID from params
    const datasetId = params.id
    
    // Implementation omitted for brevity
    // In a real application, this would delete the dataset from the database
    
    return NextResponse.json({ 
      success: true, 
      message: `Dataset ${datasetId} deleted` 
    })
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to delete dataset" },
      { status: 500 }
    )
  }
} 