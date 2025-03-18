"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../../components/ui/table"

export default function AboutPage() {
  return (
    <div className="container py-10">
      <h1 className="text-3xl font-bold tracking-tight mb-2">About MigraineDT</h1>
      <p className="text-muted-foreground mb-8">
        Learn more about the Migraine Detection & Tracking platform.
      </p>
      
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technology">Technology</TabsTrigger>
          <TabsTrigger value="team">Team</TabsTrigger>
          <TabsTrigger value="privacy">Data & Privacy</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>MigraineDT: Advanced Migraine Detection Platform</CardTitle>
              <CardDescription>
                A comprehensive solution for migraine prediction, detection, and tracking
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                MigraineDT is a sophisticated platform designed to revolutionize migraine management through 
                advanced data analytics and machine learning. By continuously monitoring physiological signals,
                environmental factors, and behavioral patterns, our system aims to predict migraine onset with 
                unprecedented accuracy.
              </p>
              
              <h3 className="text-lg font-semibold mt-6">Key Features</h3>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  <span className="font-medium">Real-time Physiological Monitoring:</span> Track ECG, HRV, and other vital signs to detect early warning signs of migraine.
                </li>
                <li>
                  <span className="font-medium">Environmental Analysis:</span> Monitor weather conditions, barometric pressure, light levels, and other environmental triggers.
                </li>
                <li>
                  <span className="font-medium">Behavioral Tracking:</span> Log sleep patterns, stress levels, medication usage, and dietary habits to identify personal triggers.
                </li>
                <li>
                  <span className="font-medium">Predictive Analytics:</span> Leverage machine learning to provide personalized migraine forecasts and early warnings.
                </li>
                <li>
                  <span className="font-medium">Comprehensive Dashboard:</span> Visualize all data in one place for patients and healthcare providers.
                </li>
              </ul>
              
              <h3 className="text-lg font-semibold mt-6">Our Mission</h3>
              <p>
                Our mission is to empower migraine sufferers with actionable insights that enable proactive management of their condition.
                By providing early warnings and identifying personal triggers, we aim to reduce the frequency and severity of migraine attacks,
                improving quality of life for millions of people worldwide.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="technology">
          <Card>
            <CardHeader>
              <CardTitle>Technology Stack</CardTitle>
              <CardDescription>The technologies powering the MigraineDT platform</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Frontend</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Technology</TableHead>
                        <TableHead>Purpose</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-medium">Next.js</TableCell>
                        <TableCell>React framework for building the user interface</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Chart.js</TableCell>
                        <TableCell>Data visualization library for physiological signal display</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">TailwindCSS</TableCell>
                        <TableCell>Utility-first CSS framework for styling</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Backend</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Technology</TableHead>
                        <TableHead>Purpose</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-medium">FastAPI</TableCell>
                        <TableCell>High-performance Python web framework for API development</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">PostgreSQL</TableCell>
                        <TableCell>Relational database for structured data storage</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Redis</TableCell>
                        <TableCell>In-memory database for caching and real-time data</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3">Data Science & Machine Learning</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Technology</TableHead>
                        <TableHead>Purpose</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-medium">PyTorch</TableCell>
                        <TableCell>Deep learning for time-series physiological data analysis</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Scikit-learn</TableCell>
                        <TableCell>Classical machine learning algorithms for prediction models</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Pandas & NumPy</TableCell>
                        <TableCell>Data manipulation and numerical computing</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">BioSPPy</TableCell>
                        <TableCell>Biosignal processing in Python for ECG/HRV analysis</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="team">
          <Card>
            <CardHeader>
              <CardTitle>Our Team</CardTitle>
              <CardDescription>The experts behind the MigraineDT platform</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="border rounded-lg p-4 text-center">
                  <div className="w-24 h-24 rounded-full bg-primary/10 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-primary">JD</span>
                  </div>
                  <h3 className="font-bold text-lg">Jane Doe, MD, PhD</h3>
                  <p className="text-sm text-muted-foreground mb-2">Neurology Specialist</p>
                  <p className="text-sm">Clinical advisor and migraine research lead</p>
                </div>
                
                <div className="border rounded-lg p-4 text-center">
                  <div className="w-24 h-24 rounded-full bg-primary/10 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-primary">JS</span>
                  </div>
                  <h3 className="font-bold text-lg">John Smith, MSc</h3>
                  <p className="text-sm text-muted-foreground mb-2">Lead Data Scientist</p>
                  <p className="text-sm">Developing predictive models and signal processing algorithms</p>
                </div>
                
                <div className="border rounded-lg p-4 text-center">
                  <div className="w-24 h-24 rounded-full bg-primary/10 mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-bold text-primary">AT</span>
                  </div>
                  <h3 className="font-bold text-lg">Alex Taylor</h3>
                  <p className="text-sm text-muted-foreground mb-2">Full-Stack Developer</p>
                  <p className="text-sm">Building the user interface and backend systems</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="privacy">
          <Card>
            <CardHeader>
              <CardTitle>Data Privacy & Security</CardTitle>
              <CardDescription>How we protect your sensitive health information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                At MigraineDT, we take data privacy and security extremely seriously. All physiological and health data 
                is treated with the utmost confidentiality and protected using industry-leading security measures.
              </p>
              
              <h3 className="text-lg font-semibold mt-6">Our Commitments</h3>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  <span className="font-medium">HIPAA Compliance:</span> All data storage and processing adheres to HIPAA requirements.
                </li>
                <li>
                  <span className="font-medium">End-to-End Encryption:</span> All data is encrypted during transmission and at rest.
                </li>
                <li>
                  <span className="font-medium">Anonymized Analysis:</span> Data used for research is fully anonymized.
                </li>
                <li>
                  <span className="font-medium">Limited Data Retention:</span> Only keeping data for as long as necessary.
                </li>
                <li>
                  <span className="font-medium">Transparent Policies:</span> Clear documentation on how your data is used.
                </li>
                <li>
                  <span className="font-medium">Data Ownership:</span> You maintain ownership of your health data.
                </li>
              </ul>
              
              <p className="mt-6">
                For more detailed information about our data handling practices, please refer to our 
                comprehensive Privacy Policy or contact our Data Protection Officer.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 