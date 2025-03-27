import click
from rich.console import Console
from rich.markdown import Markdown
from codegen import Codebase
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Set up logging and console
logging.basicConfig(level=logging.INFO)
console = Console()

def initialize_codebase(repo_path: str) -> Codebase:
    """Initialize a codebase with progress indicator."""
    with console.status("[bold blue]Initializing codebase...[/bold blue]") as status:
        try:
            codebase = Codebase(repo_path)
            status.update("[bold green]✓ Codebase initialized successfully![/bold green]")
            return codebase
        except Exception as e:
            console.print(f"[bold red]Error initializing codebase:[/bold red] {e}")
            raise

def find_unused_files(codebase: Codebase) -> list:
    """Find files that have no incoming references."""
    unused_files = []
    for file in codebase.files:
        # Skip non-Python files and test files
        if not str(file.filepath).endswith('.py') or 'test' in str(file.filepath).lower():
            continue
            
        # Check if file has any incoming references
        has_references = False
        try:
            # Check if any functions or classes in the file are used
            file_symbols = codebase.get_symbols_in_file(file.filepath)
            for symbol in file_symbols:
                if list(symbol.usages):
                    has_references = True
                    break
                    
            if not has_references:
                unused_files.append(file.filepath)
        except Exception as e:
            logging.warning(f"Could not analyze file {file.filepath}: {e}")
    
    return unused_files

def analyze_usage_patterns(codebase: Codebase) -> tuple:
    """Analyze usage patterns in the codebase."""
    # Track file usage patterns
    file_usage = defaultdict(int)
    file_symbols = defaultdict(list)
    
    # Track directory usage patterns
    dir_usage = defaultdict(int)
    
    # Analyze each function's usage
    for func in codebase.functions:
        usage_count = len(list(func.usages))
        filepath = str(func.filepath)
        file_usage[filepath] += usage_count
        file_symbols[filepath].append((func.name, usage_count))
        
        # Track directory usage
        dir_path = str(Path(filepath).parent)
        dir_usage[dir_path] += usage_count
    
    # Sort files by usage
    sorted_files = sorted(
        [(path, count) for path, count in file_usage.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Sort directories by usage
    sorted_dirs = sorted(
        [(path, count) for path, count in dir_usage.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Find low-usage files (files with less than 3 references)
    low_usage_files = [
        (path, symbols) 
        for path, symbols in file_symbols.items() 
        if sum(count for _, count in symbols) < 3 and 'test' not in path.lower()
    ]
    
    return sorted_files[:10], sorted_dirs[:5], low_usage_files

def analyze_codebase_structure(codebase: Codebase) -> str:
    """Analyze and return codebase structure information."""
    stats = {
        "files": len(codebase.files),
        "functions": len(codebase.functions),
        "classes": len(codebase.classes),
    }
    
    # Get usage patterns
    top_files, top_dirs, low_usage_files = analyze_usage_patterns(codebase)
    
    # Find most referenced symbols
    most_used = sorted(
        [(f, len(list(f.usages))) for f in codebase.functions],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    output = f"""
## Codebase Overview

- Total Files: {stats['files']}
- Total Functions: {stats['functions']}
- Total Classes: {stats['classes']}

### Most Active Files (by reference count):
{chr(10).join(f'- `{path}`: {count} references' for path, count in top_files)}

### Most Active Directories:
{chr(10).join(f'- `{path}`: {count} total references' for path, count in top_dirs)}

### Most Referenced Functions:
{chr(10).join(f'- `{f.name}`: {count} references' for f, count in most_used)}

### Low Usage Files (< 3 references, excluding tests):
"""
    
    if low_usage_files:
        for path, symbols in low_usage_files:
            output += f"\n#### `{path}`:\n"
            for symbol_name, count in symbols:
                output += f"- `{symbol_name}`: {count} references\n"
    else:
        output += "\n- No low-usage files found"
    
    return output

def search_codebase(codebase: Codebase, query: str) -> str:
    """Search the codebase for relevant code."""
    # First try semantic search
    results = []
    
    with console.status("[bold blue]Searching codebase...[/bold blue]"):
        # Use codebase_search for semantic matching
        semantic_results = codebase.codebase_search(query=query)
        if semantic_results:
            results.extend(semantic_results)
        
        # If we don't have enough results, try grep search
        if len(results) < 5:
            grep_results = codebase.grep_search(query=query)
            if grep_results:
                results.extend(grep_results)
    
    if not results:
        return "No matching code found."
    
    output = ["## Search Results\n"]
    for result in results[:5]:  # Limit to top 5 results
        output.append(f"### {result.filepath}")
        output.append(f"```python\n{result.content}\n```\n")
    
    return "\n".join(output)

def analyze_documentation_coverage(codebase: Codebase) -> dict:
    """Analyze documentation coverage across the codebase."""
    stats = {
        'total_functions': 0,
        'functions_with_docs': 0,
        'total_classes': 0,
        'classes_with_docs': 0,
        'dir_stats': defaultdict(lambda: {'total': 0, 'documented': 0}),
        'low_coverage_files': []
    }
    
    # Check functions
    for function in codebase.functions:
        if any(x in str(function.filepath).lower() for x in ['test', 'sql', 'alembic']):
            continue
            
        stats['total_functions'] += 1
        dir_path = str(Path(function.filepath).parent)
        stats['dir_stats'][dir_path]['total'] += 1
        
        if function.docstring:
            stats['functions_with_docs'] += 1
            stats['dir_stats'][dir_path]['documented'] += 1
        else:
            stats['low_coverage_files'].append((function.filepath, function.name, 'function'))

    # Check classes
    for cls in codebase.classes:
        if any(x in str(cls.filepath).lower() for x in ['test', 'sql', 'alembic']):
            continue
            
        stats['total_classes'] += 1
        dir_path = str(Path(cls.filepath).parent)
        stats['dir_stats'][dir_path]['total'] += 1
        
        if cls.docstring:
            stats['classes_with_docs'] += 1
            stats['dir_stats'][dir_path]['documented'] += 1
        else:
            stats['low_coverage_files'].append((cls.filepath, cls.name, 'class'))
    
    return stats

def format_documentation_report(stats: dict) -> str:
    """Format documentation coverage statistics into a markdown report."""
    # Calculate percentages
    func_coverage = (stats['functions_with_docs'] / stats['total_functions'] * 100) if stats['total_functions'] > 0 else 0
    class_coverage = (stats['classes_with_docs'] / stats['total_classes'] * 100) if stats['total_classes'] > 0 else 0
    overall_coverage = ((stats['functions_with_docs'] + stats['classes_with_docs']) / 
                       (stats['total_functions'] + stats['total_classes']) * 100) if (stats['total_functions'] + stats['total_classes']) > 0 else 0
    
    # Sort directories by coverage
    dir_coverage = {}
    for dir_path, dir_stats in stats['dir_stats'].items():
        if dir_stats['total'] > 0:
            coverage = (dir_stats['documented'] / dir_stats['total'] * 100)
            dir_coverage[dir_path] = (coverage, dir_stats)
    
    sorted_dirs = sorted(dir_coverage.items(), key=lambda x: x[1][0])
    
    output = f"""
## 📊 Documentation Coverage Report

### 📝 Functions:
- Total: {stats['total_functions']}
- Documented: {stats['functions_with_docs']}
- Coverage: {func_coverage:.1f}%

### 📚 Classes:
- Total: {stats['total_classes']}
- Documented: {stats['classes_with_docs']}
- Coverage: {class_coverage:.1f}%

### 🎯 Overall Coverage: {overall_coverage:.1f}%

### 📉 Directory Coverage (>10 symbols):"""
    
    for dir_path, (coverage, dir_stats) in sorted_dirs:
        if dir_stats['total'] > 10:
            output += f"\n- `{dir_path}`: {coverage:.1f}% ({dir_stats['documented']}/{dir_stats['total']} symbols)"
    
    if stats['low_coverage_files']:
        output += "\n\n### ⚠️ Symbols Missing Documentation:\n"
        for filepath, name, symbol_type in stats['low_coverage_files'][:10]:
            output += f"- {symbol_type.title()} `{name}` in `{filepath}`\n"
        if len(stats['low_coverage_files']) > 10:
            output += f"\n... and {len(stats['low_coverage_files']) - 10} more"
    
    return output

def generate_docstring(codebase: Codebase, symbol) -> str:
    """Generate a docstring for a function or class."""
    try:
        timestamp = datetime.now().strftime("%B %d, %Y")
        
        if hasattr(symbol, 'parameters'):
            # Build parameter documentation
            param_docs = []
            for param in symbol.parameters:
                param_type = param.type.source if hasattr(param, 'type') and param.type else "Any"
                param_docs.append(f"    {param.name} ({param_type}): Description of {param.name}")
            
            # Get return type if present
            return_type = symbol.return_type.source if hasattr(symbol, 'return_type') and symbol.return_type else "None"
            
            # Create Google-style docstring
            docstring = f'''"""
    Description of {symbol.name}.

    Args:
{chr(10).join(param_docs)}

    Returns:
        {return_type}: Description of return value
        
    Generated on: {timestamp}
    """'''
        else:
            # Simple docstring for classes
            docstring = f'''"""
    Description of {symbol.name} class.
    
    Generated on: {timestamp}
    """'''
        
        return docstring
    except Exception as e:
        logging.warning(f"Could not generate docstring for {symbol.name}: {e}")
        return None

@click.group()
def cli():
    """🔍 Codegen Code Research CLI"""
    pass

@cli.command()
@click.argument("path", type=str)
@click.option("--query", "-q", help="Search query")
def analyze(path: str, query: str = None):
    """Analyze a codebase or search for specific code."""
    try:
        # Initialize codebase
        codebase = initialize_codebase(path)
        
        if query:
            # Search mode
            results = search_codebase(codebase, query)
            console.print(Markdown(results))
        else:
            # Analysis mode
            overview = analyze_codebase_structure(codebase)
            console.print(Markdown(overview))
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@cli.command()
@click.argument("path", type=str)
@click.option("--generate-docs", is_flag=True, help="Generate missing docstrings")
def docs(path: str, generate_docs: bool):
    """Analyze and optionally generate documentation."""
    try:
        codebase = initialize_codebase(path)
        stats = analyze_documentation_coverage(codebase)
        
        # Print documentation coverage report
        report = format_documentation_report(stats)
        console.print(Markdown(report))
        
        if generate_docs:
            with console.status("[bold blue]Generating missing docstrings...[/bold blue]") as status:
                generated = 0
                for filepath, name, symbol_type in stats['low_coverage_files']:
                    try:
                        # Find the symbol
                        if symbol_type == 'function':
                            symbol = next((f for f in codebase.functions if f.name == name and str(f.filepath) == str(filepath)), None)
                        else:
                            symbol = next((c for c in codebase.classes if c.name == name and str(c.filepath) == str(filepath)), None)
                        
                        if symbol and not symbol.docstring:
                            docstring = generate_docstring(codebase, symbol)
                            if docstring:
                                symbol.set_docstring(docstring)
                                generated += 1
                                status.update(f"[bold blue]Generated {generated} docstrings...[/bold blue]")
                    except Exception as e:
                        logging.warning(f"Error generating docstring for {name}: {e}")
                
                console.print(f"\n[bold green]✓ Generated {generated} docstrings![/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    cli() 