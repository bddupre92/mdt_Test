"use server"

import { NextResponse } from "next/server"
import { createDataset } from "../data-service"

export async function POST(request: Request) {
  try {
    const params = await request.json()
    
    // Validate required parameters
    if (!params.name || !params.type) {
      return NextResponse.json(
        { error: "Missing required parameters" },
        { status: 400 }
      )
    }

    // Use the data service to create and store the dataset
    const dataset = await createDataset({
      name: params.name,
      description: params.description,
      type: params.type,
      category: params.category,
      features: params.features || 10,
      samples: params.samples || 1000,
      metadata: {
        generatedWith: "SyntheticDatasetGenerator",
        parameters: params,
        status: "completed"
      }
    })

    // Log the created dataset for debugging
    console.log(`Created synthetic dataset with ID: ${dataset.id}`)
    
    return NextResponse.json(dataset)
  } catch (error) {
    console.error("Error generating synthetic dataset:", error)
    return NextResponse.json(
      { error: "Failed to generate dataset" },
      { status: 500 }
    )
  }
} 