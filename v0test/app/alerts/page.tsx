"use client"

import { Button } from "../../components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
import { Badge } from "../../components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { AlertCircle, Bell, BellOff } from "lucide-react"

export default function AlertsPage() {
  return (
    <div className="container py-10">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Alerts & Notifications</h1>
          <p className="text-muted-foreground">
            Manage system alerts and patient monitoring notifications.
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <BellOff className="mr-2 h-4 w-4" />
            Mute All
          </Button>
          <Button size="sm">
            <Bell className="mr-2 h-4 w-4" />
            Configure Alerts
          </Button>
        </div>
      </div>
      
      <Tabs defaultValue="all">
        <TabsList className="mb-6">
          <TabsTrigger value="all">All Alerts</TabsTrigger>
          <TabsTrigger value="high">
            High Priority
            <Badge variant="destructive" className="ml-2">3</Badge>
          </TabsTrigger>
          <TabsTrigger value="medium">Medium Priority</TabsTrigger>
          <TabsTrigger value="low">Low Priority</TabsTrigger>
        </TabsList>
        
        <TabsContent value="all">
          <Card>
            <CardHeader>
              <CardTitle>System Alerts</CardTitle>
              <CardDescription>Alerts placeholder page - content coming soon</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] flex items-center justify-center border border-dashed rounded-md">
                <div className="text-center">
                  <AlertCircle className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground mb-4">Alert notifications will be displayed here</p>
                  <Button variant="outline">View Alert Settings</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="high">
          <Card>
            <CardContent className="pt-6">
              <div className="h-[200px] flex items-center justify-center border border-dashed rounded-md">
                <p className="text-muted-foreground">High priority alerts will appear here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="medium">
          <Card>
            <CardContent className="pt-6">
              <div className="h-[200px] flex items-center justify-center border border-dashed rounded-md">
                <p className="text-muted-foreground">Medium priority alerts will appear here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="low">
          <Card>
            <CardContent className="pt-6">
              <div className="h-[200px] flex items-center justify-center border border-dashed rounded-md">
                <p className="text-muted-foreground">Low priority alerts will appear here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 