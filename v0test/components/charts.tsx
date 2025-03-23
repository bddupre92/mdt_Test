"use client"
import { useState, useEffect, useRef } from "react"
import * as d3 from "d3"

interface DataPoint {
  x: string | number
  y: number
}

interface SeriesData {
  id: string
  name: string
  data: DataPoint[]
}

interface ChartProps {
  data: DataPoint[] | SeriesData[]
  xLabel?: string
  yLabel?: string
  height?: number
  width?: number
  showLegend?: boolean
}

interface RadarChartProps extends ChartProps {
  metrics?: string[]
}

// Utility to check if data is SeriesData[]
const isSeriesData = (data: DataPoint[] | SeriesData[]): data is SeriesData[] => {
  return data.length > 0 && 'data' in data[0] && Array.isArray(data[0].data)
}

export function LineChart({
  data,
  xLabel = "X Axis",
  yLabel = "Y Axis",
  height = 400,
  width = 800,
  showLegend = false,
}: ChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || !svgRef.current || !data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const margin = { top: 20, right: 20, bottom: 50, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create scales
    const seriesData = isSeriesData(data) ? data : [{ id: "default", name: "Default", data }]
    
    // Extract all data points from all series
    const allPoints = seriesData.flatMap(series => series.data)
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([
        d3.min(allPoints, (d: DataPoint) => typeof d.x === 'number' ? d.x : 0) as number,
        d3.max(allPoints, (d: DataPoint) => typeof d.x === 'number' ? d.x : 0) as number
      ])
      .range([0, innerWidth])
      .nice()
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(allPoints, (d: DataPoint) => d.y) as number])
      .range([innerHeight, 0])
      .nice()

    // Create axes
    const xAxis = d3.axisBottom(xScale)
    const yAxis = d3.axisLeft(yScale)

    // Create line generator
    const line = d3.line<DataPoint>()
      .x(d => xScale(typeof d.x === 'number' ? d.x : 0))
      .y(d => yScale(d.y))
      .curve(d3.curveMonotoneX)

    // Create container
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("x", innerWidth / 2)
      .attr("y", 35)
      .attr("text-anchor", "middle")
      .text(xLabel)

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .text(yLabel)

    // Create color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)

    // Draw lines
    seriesData.forEach((series, i) => {
      g.append("path")
        .datum(series.data)
        .attr("fill", "none")
        .attr("stroke", colorScale(series.id))
        .attr("stroke-width", 2)
        .attr("d", line)
    })

    // Add legend if needed
    if (showLegend && isSeriesData(data)) {
      const legend = svg.append("g")
        .attr("transform", `translate(${margin.left + innerWidth - 120}, ${margin.top + 10})`)

      seriesData.forEach((series, i) => {
        const legendRow = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`)

        legendRow.append("rect")
          .attr("width", 10)
          .attr("height", 10)
          .attr("fill", colorScale(series.id))

        legendRow.append("text")
          .attr("x", 15)
          .attr("y", 9)
          .attr("class", "text-xs")
          .text(series.name)
      })
    }
  }, [data, xLabel, yLabel, height, width, showLegend, mounted])

  return (
    <div className="w-full h-full flex items-center justify-center">
      {!mounted || !data || data.length === 0 ? (
        <p className="text-muted-foreground">No data to display</p>
      ) : (
        <svg 
          ref={svgRef} 
          width="100%" 
          height="100%" 
          viewBox={`0 0 ${width} ${height}`} 
          preserveAspectRatio="xMidYMid meet"
        />
      )}
    </div>
  )
}

export function BarChart({
  data,
  xLabel = "X Axis",
  yLabel = "Y Axis",
  height = 400,
  width = 800,
  showLegend = false,
}: ChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || !svgRef.current || !data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const margin = { top: 20, right: 20, bottom: 70, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Extract data points from all series if necessary
    const dataPoints = isSeriesData(data) 
      ? data.flatMap(series => series.data) 
      : data as DataPoint[]

    // Create scales
    const xScale = d3.scaleBand()
      .domain(dataPoints.map((d: DataPoint) => d.x.toString()))
      .range([0, innerWidth])
      .padding(0.2)

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(dataPoints, (d: DataPoint) => d.y) as number])
      .range([innerHeight, 0])
      .nice()

    // Create axes
    const xAxis = d3.axisBottom(xScale)
    const yAxis = d3.axisLeft(yScale)

    // Create container
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .attr("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")

    g.append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 50)
      .attr("text-anchor", "middle")
      .text(xLabel)

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .text(yLabel)

    // Draw bars
    g.selectAll(".bar")
      .data(dataPoints)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d: DataPoint) => xScale(d.x.toString()) as number)
      .attr("y", (d: DataPoint) => yScale(d.y))
      .attr("width", xScale.bandwidth())
      .attr("height", (d: DataPoint) => innerHeight - yScale(d.y))
      .attr("fill", "#3b82f6")
      .attr("rx", 2)
  }, [data, xLabel, yLabel, height, width, showLegend, mounted])

  return (
    <div className="w-full h-full flex items-center justify-center">
      {!mounted || !data || data.length === 0 ? (
        <p className="text-muted-foreground">No data to display</p>
      ) : (
        <svg 
          ref={svgRef} 
          width="100%" 
          height="100%" 
          viewBox={`0 0 ${width} ${height}`} 
          preserveAspectRatio="xMidYMid meet"
        />
      )}
    </div>
  )
}

export function ScatterChart({
  data,
  xLabel = "X Axis",
  yLabel = "Y Axis",
  height = 400,
  width = 800,
  showLegend = false,
}: ChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || !svgRef.current || !data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const margin = { top: 20, right: 20, bottom: 50, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create scales
    const seriesData = isSeriesData(data) ? data : [{ id: "default", name: "Default", data }]
    
    // Extract all data points from all series
    const allPoints = seriesData.flatMap(series => series.data)
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([
        d3.min(allPoints, (d: DataPoint) => typeof d.x === 'number' ? d.x : 0) as number,
        d3.max(allPoints, (d: DataPoint) => typeof d.x === 'number' ? d.x : 0) as number
      ])
      .range([0, innerWidth])
      .nice()
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(allPoints, (d: DataPoint) => d.y) as number])
      .range([innerHeight, 0])
      .nice()

    // Create axes
    const xAxis = d3.axisBottom(xScale)
    const yAxis = d3.axisLeft(yScale)

    // Create container
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("x", innerWidth / 2)
      .attr("y", 35)
      .attr("text-anchor", "middle")
      .text(xLabel)

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("fill", "currentColor")
      .attr("class", "text-sm font-medium")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .text(yLabel)

    // Create color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)

    // Draw scatter points for each series
    seriesData.forEach((series, i) => {
      g.selectAll(".dot-" + series.id)
        .data(series.data)
        .enter()
        .append("circle")
        .attr("class", "dot-" + series.id)
        .attr("cx", (d: DataPoint) => xScale(typeof d.x === 'number' ? d.x : 0))
        .attr("cy", (d: DataPoint) => yScale(d.y))
        .attr("r", 5)
        .attr("fill", colorScale(series.id))
        .attr("opacity", 0.8)
    })

    // Add legend if needed
    if (showLegend && isSeriesData(data)) {
      const legend = svg.append("g")
        .attr("transform", `translate(${margin.left + innerWidth - 120}, ${margin.top + 10})`)

      seriesData.forEach((series, i) => {
        const legendRow = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`)

        legendRow.append("circle")
          .attr("cx", 5)
          .attr("cy", 5)
          .attr("r", 5)
          .attr("fill", colorScale(series.id))

        legendRow.append("text")
          .attr("x", 15)
          .attr("y", 9)
          .attr("class", "text-xs")
          .text(series.name)
      })
    }
  }, [data, xLabel, yLabel, height, width, showLegend, mounted])

  return (
    <div className="w-full h-full flex items-center justify-center">
      {!mounted || !data || data.length === 0 ? (
        <p className="text-muted-foreground">No data to display</p>
      ) : (
        <svg 
          ref={svgRef} 
          width="100%" 
          height="100%" 
          viewBox={`0 0 ${width} ${height}`} 
          preserveAspectRatio="xMidYMid meet"
        />
      )}
    </div>
  )
}

export function RadarChart({
  data,
  xLabel = "",
  yLabel = "",
  height = 400,
  width = 800,
  showLegend = false,
  metrics = []
}: RadarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || !svgRef.current || !data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const margin = { top: 50, right: 80, bottom: 50, left: 80 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // We expect data to be series data for radar chart
    const seriesData = isSeriesData(data) ? data : [{ id: "default", name: "Default", data }]
    
    // Create container
    const g = svg.append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`)

    // Radar chart requires a specific data structure
    // Calculate angles for each metric
    const angleSlice = (Math.PI * 2) / metrics.length
    
    // Create radial scale
    const rScale = d3.scaleLinear()
      .domain([0, d3.max(seriesData.flatMap(s => s.data.map((d: DataPoint) => d.y))) || 1])
      .range([0, Math.min(innerWidth, innerHeight) / 2])
      
    // Create color scale for the series
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
    
    // Draw radar grid
    const radarGrid = g.append("g").attr("class", "radar-grid")
    
    // Draw the background circles
    const gridLevels = 5
    const gridCircles = radarGrid.selectAll(".grid-circle")
      .data(d3.range(1, gridLevels + 1).reverse())
      .enter()
      .append("circle")
      .attr("class", "grid-circle")
      .attr("r", (d: number) => (rScale.range()[1] / gridLevels) * d)
      .attr("fill", "none")
      .attr("stroke", "#ccc")
      .attr("stroke-dasharray", "2,2")
    
    // Draw the axes (spokes)
    const axes = radarGrid.selectAll(".axis")
      .data(metrics)
      .enter()
      .append("g")
      .attr("class", "axis")
    
    axes.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", (d, i) => rScale.range()[1] * Math.cos(angleSlice * i - Math.PI / 2))
      .attr("y2", (d, i) => rScale.range()[1] * Math.sin(angleSlice * i - Math.PI / 2))
      .attr("stroke", "#ccc")
      .attr("stroke-width", 1)
    
    // Add labels for each axis
    axes.append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("x", (d, i) => (rScale.range()[1] + 20) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr("y", (d, i) => (rScale.range()[1] + 20) * Math.sin(angleSlice * i - Math.PI / 2))
      .text(d => d)
      .attr("fill", "currentColor")
      .attr("font-size", "12px")
    
    // Draw the radar chart areas for each series
    seriesData.forEach((series, seriesIndex) => {
      // Map data points to radar coordinates
      const radarPoints = metrics.map((metric, i) => {
        // Find the data point for this metric
        const dataPoint = series.data.find((d: DataPoint) => d.x === metric)
        const value = dataPoint ? dataPoint.y : 0
        
        // Convert to x,y coordinates
        return {
          x: rScale(value) * Math.cos(angleSlice * i - Math.PI / 2),
          y: rScale(value) * Math.sin(angleSlice * i - Math.PI / 2)
        }
      })
      
      // Create line generator for radar
      const radarLine = d3.lineRadial<{x: number, y: number}>()
        .angle((d, i) => angleSlice * i)
        .radius(d => Math.sqrt(d.x * d.x + d.y * d.y))
        .curve(d3.curveLinearClosed)
      
      // Draw the radar area
      g.append("path")
        .datum(radarPoints)
        .attr("class", "radar-area")
        .attr("d", (d: any) => {
          // Convert points to a format the lineRadial can use
          const radialPoints = d.map((p: any, i: number) => [
            angleSlice * i,
            Math.sqrt(p.x * p.x + p.y * p.y)
          ])
          
          // Create a closed path
          let path = "M "
          radialPoints.forEach((p: any, i: number) => {
            const x = p[1] * Math.cos(p[0])
            const y = p[1] * Math.sin(p[0])
            if (i === 0) path += `${x},${y} `
            else path += `L ${x},${y} `
          })
          path += "Z"
          return path
        })
        .attr("fill", colorScale(series.id))
        .attr("fill-opacity", 0.2)
        .attr("stroke", colorScale(series.id))
        .attr("stroke-width", 2)
      
      // Add points at each data point
      g.selectAll(".radar-point-" + seriesIndex)
        .data(radarPoints)
        .enter()
        .append("circle")
        .attr("class", "radar-point")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", 4)
        .attr("fill", colorScale(series.id))
    })
    
    // Add legend if required
    if (showLegend && seriesData.length > 1) {
      const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - margin.right + 10}, ${margin.top})`)
      
      seriesData.forEach((series, i) => {
        const legendRow = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`)
        
        legendRow.append("rect")
          .attr("width", 10)
          .attr("height", 10)
          .attr("fill", colorScale(series.id))
        
        legendRow.append("text")
          .attr("x", 15)
          .attr("y", 10)
          .attr("font-size", "12px")
          .attr("fill", "currentColor")
          .text(series.name)
      })
    }

  }, [data, xLabel, yLabel, height, width, showLegend, mounted, metrics])

  return (
    <div className="w-full h-full flex items-center justify-center">
      {!mounted || !data || data.length === 0 ? (
        <p className="text-muted-foreground">No data to display</p>
      ) : (
        <svg 
          ref={svgRef} 
          width="100%" 
          height="100%" 
          viewBox={`0 0 ${width} ${height}`} 
          preserveAspectRatio="xMidYMid meet"
        />
      )}
    </div>
  )
}

