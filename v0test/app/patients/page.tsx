"use client"

import { useState } from "react"
import { Button } from "../../components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "../../components/ui/card"
import { Input } from "../../components/ui/input"
import { Search, UserRound, MoreHorizontal, ArrowUpRight } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../../components/ui/table"
import { Badge } from "../../components/ui/badge"
import { Avatar, AvatarFallback } from "../../components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "../../components/ui/dropdown-menu"
import { LineChart } from "../../components/charts"

// Sample patient data
const samplePatients = [
  {
    id: "patient-001",
    name: "Jane Doe",
    age: 32,
    gender: "Female",
    lastRecording: "2023-03-15T14:30:00",
    migraine: { status: "low", value: 15 },
    contact: "janedoe@example.com",
    lastVisit: "2023-03-10"
  },
  {
    id: "patient-002",
    name: "John Smith",
    age: 45,
    gender: "Male",
    lastRecording: "2023-03-14T09:15:00",
    migraine: { status: "high", value: 85 },
    contact: "johnsmith@example.com",
    lastVisit: "2023-02-28"
  },
  {
    id: "patient-003",
    name: "Emma Wilson",
    age: 28,
    gender: "Female",
    lastRecording: "2023-03-15T16:45:00",
    migraine: { status: "medium", value: 45 },
    contact: "emmawilson@example.com",
    lastVisit: "2023-03-05"
  },
  {
    id: "patient-004",
    name: "Michael Johnson",
    age: 52,
    gender: "Male",
    lastRecording: "2023-03-13T11:20:00",
    migraine: { status: "low", value: 25 },
    contact: "michaeljohnson@example.com",
    lastVisit: "2023-02-20"
  },
  {
    id: "patient-005",
    name: "Sarah Brown",
    age: 39,
    gender: "Female",
    lastRecording: "2023-03-15T10:30:00",
    migraine: { status: "high", value: 90 },
    contact: "sarahbrown@example.com",
    lastVisit: "2023-03-12"
  }
]

interface DataPoint {
  x: string | number
  y: number
}

// Helper function to get badge variant based on risk level
const getRiskVariant = (status: string) => {
  if (status === "high") return "destructive"
  if (status === "medium") return "secondary"
  return "secondary"
}

export default function PatientsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedPatient, setSelectedPatient] = useState<any>(null)
  
  // Filter patients based on search query
  const filteredPatients = searchQuery 
    ? samplePatients.filter(patient => 
        patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        patient.id.toLowerCase().includes(searchQuery.toLowerCase()))
    : samplePatients
    
  // Generate sample trend data for a patient
  const generateTrendData = (patientId: string): DataPoint[] => {
    // Generate fake data points for the past 14 days
    return Array.from({ length: 14 }, (_, i) => ({
      x: new Date(Date.now() - (13 - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      }),
      y: Math.random() * 100
    }))
  }
  
  // Handle patient selection
  const handlePatientSelect = (patient: any) => {
    setSelectedPatient(patient)
  }

  return (
    <div className="container py-10">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patient Records</h1>
          <p className="text-muted-foreground">
            Manage and view patient data and migraine prediction records.
          </p>
        </div>
        
        <div className="flex w-full md:w-auto gap-2">
          <div className="relative w-full md:w-64">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search patients..."
              className="w-full pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <Button>Add Patient</Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Patient Directory</CardTitle>
            <CardDescription>View and manage patient records</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Gender</TableHead>
                  <TableHead>Migraine Risk</TableHead>
                  <TableHead>Last Recording</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredPatients.map(patient => (
                  <TableRow key={patient.id} onClick={() => handlePatientSelect(patient)} className="cursor-pointer">
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarFallback>{patient.name.split(' ').map((n: string) => n[0]).join('')}</AvatarFallback>
                        </Avatar>
                        <div>
                          <div className="font-medium">{patient.name}</div>
                          <div className="text-xs text-muted-foreground">{patient.id}</div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>{patient.age}</TableCell>
                    <TableCell>{patient.gender}</TableCell>
                    <TableCell>
                      <Badge variant={getRiskVariant(patient.migraine.status)}>
                        {patient.migraine.value}%
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {new Date(patient.lastRecording).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handlePatientSelect(patient)}>
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>Edit Profile</DropdownMenuItem>
                          <DropdownMenuItem>View Recordings</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Patient Details</CardTitle>
            <CardDescription>
              {selectedPatient ? `Data for ${selectedPatient.name}` : "Select a patient to view details"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedPatient ? (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <Avatar className="h-20 w-20">
                    <AvatarFallback className="text-xl">
                      {selectedPatient.name.split(' ').map((n: string) => n[0]).join('')}
                    </AvatarFallback>
                  </Avatar>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-muted-foreground">Patient ID</div>
                  <div className="text-right font-medium">{selectedPatient.id}</div>
                  
                  <div className="text-muted-foreground">Age</div>
                  <div className="text-right font-medium">{selectedPatient.age}</div>
                  
                  <div className="text-muted-foreground">Gender</div>
                  <div className="text-right font-medium">{selectedPatient.gender}</div>
                  
                  <div className="text-muted-foreground">Contact</div>
                  <div className="text-right font-medium">{selectedPatient.contact}</div>
                  
                  <div className="text-muted-foreground">Last Visit</div>
                  <div className="text-right font-medium">{selectedPatient.lastVisit}</div>
                  
                  <div className="text-muted-foreground">Migraine Risk</div>
                  <div className="text-right font-medium">
                    <Badge variant={getRiskVariant(selectedPatient.migraine.status)}>
                      {selectedPatient.migraine.value}%
                    </Badge>
                  </div>
                </div>
                
                <div>
                  <div className="mb-2 font-medium">14-Day Migraine Trend</div>
                  <div className="h-[150px]">
                    <LineChart data={generateTrendData(selectedPatient.id)} height={150} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-[300px] flex items-center justify-center border border-dashed rounded-md">
                <div className="text-center">
                  <UserRound className="mx-auto h-12 w-12 text-muted-foreground opacity-50" />
                  <p className="text-muted-foreground mt-2">Select a patient from the list</p>
                </div>
              </div>
            )}
          </CardContent>
          {selectedPatient && (
            <CardFooter>
              <Button className="w-full">
                <ArrowUpRight className="mr-2 h-4 w-4" />
                View Full Profile
              </Button>
            </CardFooter>
          )}
        </Card>
      </div>
    </div>
  )
} 