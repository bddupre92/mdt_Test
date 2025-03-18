import Link from "next/link"
import { Button } from "./ui/button"
import { MainNav } from "./main-nav"
import { ThemeToggle } from "./theme-toggle"
import { Badge } from "./ui/badge"

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background">
      <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center space-x-2">
            <span className="inline-block font-bold text-xl">MigraineDT</span>
            <Badge variant="secondary" className="text-xs font-normal">Beta</Badge>
          </Link>
          <MainNav />
        </div>
        <div className="flex flex-1 items-center justify-end space-x-4">
          <div className="flex items-center space-x-1">
            <ThemeToggle />
            <Button variant="ghost" size="icon" className="ml-2">
              <span className="sr-only">Notifications</span>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5"
              >
                <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
                <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
              </svg>
            </Button>
            <Button variant="ghost" size="sm" className="ml-2 gap-1 hidden sm:flex">
              <span className="h-6 w-6 rounded-full bg-primary text-primary-foreground grid place-items-center text-xs font-bold">JD</span>
              <span>Account</span>
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
} 