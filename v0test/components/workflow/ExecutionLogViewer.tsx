"use client"

import React, { useState, useEffect, useRef } from 'react';
import { fetchExecutionLogs } from '@/lib/api/execution';
import { getLogLevelClasses } from '@/lib/utils/execution-status';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, AlertTriangle, Info, Copy, Download, Search, Terminal, X } from 'lucide-react';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  source: string;
  data?: Record<string, any>;
}

interface LogLevelFilters {
  info: boolean;
  warning: boolean;
  error: boolean;
  debug: boolean;
}

interface ExecutionLogViewerProps {
  executionId: string;
  autoRefresh?: boolean;
  maxHeight?: string;
  className?: string;
  showControls?: boolean;
}

export function ExecutionLogViewer({
  executionId,
  autoRefresh = true,
  maxHeight = '400px',
  className = '',
  showControls = true,
}: ExecutionLogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilters, setLevelFilters] = useState<LogLevelFilters>({
    info: true,
    warning: true,
    error: true,
    debug: false,
  });
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  
  // Auto-scroll state
  const [autoScroll, setAutoScroll] = useState(true);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  
  // Refresh interval
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Load logs
  useEffect(() => {
    const loadLogs = async () => {
      if (!executionId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const fetchedLogs = await fetchExecutionLogs(executionId);
        setLogs(fetchedLogs);
      } catch (err) {
        console.error('Failed to fetch logs:', err);
        setError('Failed to load execution logs');
      } finally {
        setLoading(false);
      }
    };
    
    loadLogs();
    
    // Set up auto-refresh if enabled
    if (autoRefresh) {
      refreshIntervalRef.current = setInterval(loadLogs, 3000);
    }
    
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [executionId, autoRefresh]);
  
  // Apply filters to logs
  useEffect(() => {
    let filtered = [...logs];
    
    // Apply level filters
    filtered = filtered.filter(log => {
      return levelFilters[log.level as keyof LogLevelFilters];
    });
    
    // Apply source filter
    if (sourceFilter !== 'all') {
      filtered = filtered.filter(log => log.source === sourceFilter);
    }
    
    // Apply search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(term) || 
        (log.data && JSON.stringify(log.data).toLowerCase().includes(term))
      );
    }
    
    setFilteredLogs(filtered);
  }, [logs, levelFilters, sourceFilter, searchTerm]);
  
  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);
  
  // Get unique log sources for filtering
  const logSources = React.useMemo(() => {
    const sources = new Set(logs.map(log => log.source));
    return [...sources];
  }, [logs]);
  
  // Copy logs to clipboard
  const copyLogs = () => {
    const logsText = filteredLogs
      .map(log => `[${log.timestamp}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`)
      .join('\n');
    
    navigator.clipboard.writeText(logsText)
      .then(() => {
        alert('Logs copied to clipboard');
      })
      .catch(err => {
        console.error('Failed to copy logs:', err);
      });
  };
  
  // Download logs
  const downloadLogs = () => {
    const logsText = filteredLogs
      .map(log => `[${log.timestamp}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}${log.data ? ` ${JSON.stringify(log.data)}` : ''}`)
      .join('\n');
    
    const blob = new Blob([logsText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `execution-${executionId}-logs.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Clear search
  const clearSearch = () => {
    setSearchTerm('');
  };
  
  // Reset filters
  const resetFilters = () => {
    setLevelFilters({
      info: true,
      warning: true,
      error: true,
      debug: false,
    });
    setSourceFilter('all');
    setSearchTerm('');
  };
  
  // Render log level icon
  const renderLogLevelIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <AlertCircle className="mr-1 h-3 w-3 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="mr-1 h-3 w-3 text-amber-500" />;
      case 'info':
        return <Info className="mr-1 h-3 w-3 text-blue-500" />;
      case 'debug':
        return <Terminal className="mr-1 h-3 w-3 text-gray-500" />;
      default:
        return null;
    }
  };
  
  return (
    <div className={`border rounded-md ${className}`}>
      {/* Log Controls */}
      {showControls && (
        <div className="border-b p-2 bg-gray-50 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Execution Logs</h3>
            <div className="flex items-center gap-1">
              <Button 
                variant="outline" 
                size="sm"
                onClick={copyLogs} 
                title="Copy logs to clipboard"
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                onClick={downloadLogs} 
                title="Download logs as text file"
              >
                <Download className="h-3 w-3 mr-1" />
                Download
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                onClick={resetFilters} 
                title="Reset all filters"
              >
                Reset
              </Button>
            </div>
          </div>
          
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 relative">
              <Input
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="Search logs..."
                className="pr-8 text-sm h-8"
              />
              {searchTerm ? (
                <button 
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  onClick={clearSearch}
                >
                  <X className="h-3 w-3" />
                </button>
              ) : (
                <Search className="absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-gray-400" />
              )}
            </div>
            
            <Select value={sourceFilter} onValueChange={setSourceFilter}>
              <SelectTrigger className="w-[150px] h-8 text-xs">
                <SelectValue placeholder="All sources" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All sources</SelectItem>
                {logSources.map(source => (
                  <SelectItem key={source} value={source}>{source}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
            <div className="flex items-center space-x-1">
              <Checkbox 
                id="filter-info" 
                checked={levelFilters.info} 
                onCheckedChange={(checked: boolean) => setLevelFilters({...levelFilters, info: checked})}
              />
              <Label htmlFor="filter-info" className="text-xs cursor-pointer">
                <div className="flex items-center">
                  <Info className="mr-1 h-3 w-3 text-blue-500" />
                  Info
                </div>
              </Label>
            </div>
            <div className="flex items-center space-x-1">
              <Checkbox 
                id="filter-warning" 
                checked={levelFilters.warning} 
                onCheckedChange={(checked: boolean) => setLevelFilters({...levelFilters, warning: checked})}
              />
              <Label htmlFor="filter-warning" className="text-xs cursor-pointer">
                <div className="flex items-center">
                  <AlertTriangle className="mr-1 h-3 w-3 text-amber-500" />
                  Warning
                </div>
              </Label>
            </div>
            <div className="flex items-center space-x-1">
              <Checkbox 
                id="filter-error" 
                checked={levelFilters.error} 
                onCheckedChange={(checked: boolean) => setLevelFilters({...levelFilters, error: checked})}
              />
              <Label htmlFor="filter-error" className="text-xs cursor-pointer">
                <div className="flex items-center">
                  <AlertCircle className="mr-1 h-3 w-3 text-red-500" />
                  Error
                </div>
              </Label>
            </div>
            <div className="flex items-center space-x-1">
              <Checkbox 
                id="filter-debug" 
                checked={levelFilters.debug} 
                onCheckedChange={(checked: boolean) => setLevelFilters({...levelFilters, debug: checked})}
              />
              <Label htmlFor="filter-debug" className="text-xs cursor-pointer">
                <div className="flex items-center">
                  <Terminal className="mr-1 h-3 w-3 text-gray-500" />
                  Debug
                </div>
              </Label>
            </div>
            <div className="flex items-center space-x-1 ml-auto">
              <Checkbox 
                id="auto-scroll" 
                checked={autoScroll} 
                onCheckedChange={(checked: boolean) => setAutoScroll(checked)}
              />
              <Label htmlFor="auto-scroll" className="text-xs cursor-pointer">
                Auto-scroll
              </Label>
            </div>
          </div>
        </div>
      )}
      
      {/* Log Content */}
      <div 
        ref={logsContainerRef}
        className="p-2 overflow-auto font-mono text-xs"
        style={{ maxHeight }}
      >
        {loading && logs.length === 0 && (
          <div className="flex justify-center items-center h-20 text-gray-500">
            Loading logs...
          </div>
        )}
        
        {error && (
          <div className="text-red-500 p-2 border border-red-200 rounded bg-red-50">
            {error}
          </div>
        )}
        
        {!loading && !error && filteredLogs.length === 0 && (
          <div className="text-gray-500 p-2 text-center">
            No logs match the current filters.
          </div>
        )}
        
        {filteredLogs.map((log, index) => (
          <div 
            key={index}
            className={`py-1 px-2 border-l-2 mb-1 rounded ${getLogLevelClasses(log.level)}`}
          >
            <div className="flex items-start">
              <span className="text-gray-500 mr-2">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
              <div className="flex-1">
                <div className="flex items-center">
                  {renderLogLevelIcon(log.level)}
                  <span className="font-medium">{log.source}</span>
                </div>
                <div className="mt-0.5">{log.message}</div>
                {log.data && (
                  <pre className="mt-1 p-1 bg-black/5 rounded text-[10px] overflow-x-auto">
                    {JSON.stringify(log.data, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Log Status Bar */}
      <div className="border-t p-1 bg-gray-50 flex justify-between items-center text-xs text-gray-500">
        <div>
          Showing {filteredLogs.length} of {logs.length} log entries
        </div>
        {autoRefresh && (
          <div>
            Auto-refreshing every 3 seconds
          </div>
        )}
      </div>
    </div>
  );
} 