"""
Base Command Definition

This module provides the base Command class to avoid circular imports.
"""
import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

class Command(ABC):
    """
    Base class for all CLI commands
    """
    def __init__(self, args):
        """
        Initialize the command with parsed arguments
        
        Parameters:
        -----------
        args : argparse.Namespace or dict
            Parsed command-line arguments
        """
        # If args is already a dictionary, convert it to a Namespace
        if isinstance(args, dict):
            from argparse import Namespace
            namespace_args = Namespace()
            for key, value in args.items():
                setattr(namespace_args, key, value)
            self.args = namespace_args
        else:
            # Otherwise, just pass it through
            self.args = args
            
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self) -> int:
        """
        Execute the command
        
        Returns:
        --------
        int
            Exit code (0 for success, non-zero for failure)
        """
        pass
