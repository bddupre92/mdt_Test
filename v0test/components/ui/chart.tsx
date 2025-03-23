"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface ChartContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

const ChartContainer = React.forwardRef<HTMLDivElement, ChartContainerProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("w-full h-[300px] md:h-[400px]", className)}
      {...props}
    >
      {children}
    </div>
  )
)
ChartContainer.displayName = "ChartContainer"

interface ChartTooltipProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  visible?: boolean;
  x?: number;
  y?: number;
}

const ChartTooltip = React.forwardRef<HTMLDivElement, ChartTooltipProps>(
  ({ className, children, visible, x, y, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "absolute pointer-events-none transition-opacity duration-200",
        visible ? "opacity-100" : "opacity-0",
        className
      )}
      style={{
        transform: `translate(${x || 0}px, ${y || 0}px)`,
      }}
      {...props}
    >
      {children}
    </div>
  )
)
ChartTooltip.displayName = "ChartTooltip"

interface ChartTooltipContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

const ChartTooltipContent = React.forwardRef<HTMLDivElement, ChartTooltipContentProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "bg-background p-2 rounded-md shadow border text-sm",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
)
ChartTooltipContent.displayName = "ChartTooltipContent"

export { ChartContainer, ChartTooltip, ChartTooltipContent } 