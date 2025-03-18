"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import {
  Home,
  BarChart,
  Settings,
  Menu,
  X,
  Database
} from "lucide-react"

export function MainNav() {
  const pathname = usePathname()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const navItems = [
    {
      name: "Dashboard",
      href: "/",
      icon: Home,
      isActive: pathname === "/"
    },
    {
      name: "Results",
      href: "/results",
      icon: BarChart,
      isActive: pathname === "/results"
    },
    {
      name: "Datasets",
      href: "/datasets",
      icon: Database,
      isActive: pathname === "/datasets"
    },
    {
      name: "Settings",
      href: "/settings",
      icon: Settings,
      isActive: pathname === "/settings"
    },
    {
      name: "Meta-Optimizer",
      href: "/meta-optimizer",
      icon: Home,
      isActive: pathname === "/meta-optimizer"
    }
  ]

  return (
    <div className="flex items-center">
      <Link href="/" className="mr-6 flex items-center space-x-2">
        <span className="hidden font-bold sm:inline-block">
          Meta-Optimizer Framework
        </span>
      </Link>
      <nav className="hidden md:flex items-center space-x-6 text-sm font-medium">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center transition-colors hover:text-foreground/80",
              item.isActive ? "text-foreground" : "text-foreground/60"
            )}
          >
            <item.icon className="h-4 w-4 mr-2" />
            {item.name}
          </Link>
        ))}
      </nav>
      <button
        className="md:hidden ml-auto"
        onClick={toggleMobileMenu}
        aria-label="Toggle Menu"
      >
        {isMobileMenuOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <Menu className="h-6 w-6" />
        )}
      </button>
      {isMobileMenuOpen && (
        <div className="absolute top-16 left-0 right-0 bg-background border-b z-50 md:hidden">
          <nav className="flex flex-col p-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center py-2 transition-colors hover:text-foreground/80",
                  item.isActive ? "text-foreground" : "text-foreground/60"
                )}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <item.icon className="h-4 w-4 mr-2" />
                {item.name}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </div>
  )
} 