"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Upload, FileText, X } from "lucide-react"

interface FileUploaderProps {
  onFileUpload: (file: File) => void
  acceptedTypes?: string
}

export function FileUploader({ onFileUpload, acceptedTypes = ".csv,.json,.xlsx,.xls" }: FileUploaderProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setSelectedFile(file)
      onFileUpload(file)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()

    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      onFileUpload(file)
    }
  }

  const handleButtonClick = () => {
    inputRef.current?.click()
  }

  const handleRemoveFile = () => {
    setSelectedFile(null)
    if (inputRef.current) {
      inputRef.current.value = ""
    }
  }

  return (
    <div className="flex flex-col items-center">
      <input ref={inputRef} type="file" accept={acceptedTypes} onChange={handleChange} className="hidden" />

      {!selectedFile ? (
        <div
          className={`w-full p-4 border-2 border-dashed rounded-md flex flex-col items-center justify-center cursor-pointer transition-colors ${
            dragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={handleButtonClick}
        >
          <Upload className="h-6 w-6 mb-2 text-muted-foreground" />
          <p className="text-sm text-center text-muted-foreground">Drag and drop a file here, or click to select</p>
        </div>
      ) : (
        <div className="w-full p-3 border rounded-md flex items-center justify-between">
          <div className="flex items-center">
            <FileText className="h-5 w-5 mr-2 text-muted-foreground" />
            <span className="text-sm truncate max-w-[180px]">{selectedFile.name}</span>
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleRemoveFile}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

