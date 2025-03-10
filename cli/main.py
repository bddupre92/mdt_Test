#!/usr/bin/env python3
"""
Main entry point for the CLI interface

This module provides the main() function that serves as the entry point
for the CLI interface.
"""
import sys
import logging
from typing import Optional, List
import traceback

from .argument_parser import parse_args
from .commands import COMMAND_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args: Optional[List[str]] = None) -> int:
    """
    Main function
    
    Args:
        args: Command-line arguments (if None, use sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Get command name
        command = parsed_args.pop("command")
        
        # Get command class
        command_class = COMMAND_MAP.get(command)
        if command_class is None:
            logger.error(f"Unknown command: {command}")
            return 1
        
        # Create and execute command
        command_instance = command_class(parsed_args)
        return command_instance.execute()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
