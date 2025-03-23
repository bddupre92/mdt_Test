import logging
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(level=logging.INFO, log_to_file=True, log_dir='logs', log_format=None):
    """
    Configure logging for the application
    
    Parameters:
    -----------
    level : int, optional
        Logging level (default: logging.INFO)
    log_to_file : bool, optional
        Whether to log to file in addition to console (default: True)
    log_dir : str, optional
        Directory to store log files (default: 'logs')
    log_format : str, optional
        Custom log format. If None, a default format will be used.
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'app_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to {log_file}")
    
    return root_logger

def get_logger(name, level=None):
    """
    Get a logger with a specific name
    
    Parameters:
    -----------
    name : str
        Logger name, typically __name__ of the module
    level : int, optional
        Logging level. If None, inherits from parent logger.
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    
    return logger

class LoggingContext:
    """
    Context manager for temporarily changing logging level
    
    Examples:
    ---------
    >>> with LoggingContext('my_module', logging.DEBUG):
    ...     # This code will run with DEBUG level for my_module
    ...     pass
    """
    def __init__(self, logger_name, level):
        self.logger = logging.getLogger(logger_name)
        self.old_level = self.logger.level
        self.level = level
        
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
