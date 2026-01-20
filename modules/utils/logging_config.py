import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = 'INFO',
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    colored_output: bool = True
):
    """
    Setup logging configuration for the entire project
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        format_string: Custom format string
        log_file: Optional file path to write logs
        console_output: Whether to output to console
        colored_output: Whether to use colors (if colorlog is available)
    """
    if format_string is None:
        if colored_output:
            # Colored format if colorlog is available
            try:
                import colorlog
                format_string = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            except ImportError:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Try to use colored formatter
        if colored_output:
            try:
                import colorlog
                console_formatter = colorlog.ColoredFormatter(
                    format_string,
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
            except ImportError:
                console_formatter = logging.Formatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)
            
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        # File doesn't need colors
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Set level for specific loggers to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specified name
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_file_logging(log_dir: str = 'logs'):
    """
    Setup file logging with timestamp
    
    Args:
        log_dir: Directory to store log files
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'medical_datasets_{timestamp}.log')
    
    setup_logging(
        level='INFO',
        log_file=log_file,
        console_output=True,
        colored_output=True
    )
    
    logger = get_logger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return log_file


def setup_debug_logging():
    """
    Setup debug level logging for development
    """
    setup_logging(
        level='DEBUG',
        console_output=True,
        colored_output=True
    )
    
    logger = get_logger(__name__)
    logger.debug("Debug logging enabled")


def setup_production_logging(log_dir: str = '/var/log/medical_datasets'):
    """
    Setup production logging with file rotation
    """
    from logging.handlers import RotatingFileHandler
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup rotating file handler
    log_file = os.path.join(log_dir, 'medical_datasets.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Setup console handler with WARNING level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    logger = get_logger(__name__)
    logger.info(f"Production logging setup. Log file: {log_file}")


# Setup default logging when module is imported
setup_logging()


# Convenience functions for different environments
def enable_debug():
    """Enable debug logging"""
    setup_debug_logging()


def enable_file_logging(log_dir: str = 'logs'):
    """Enable file logging"""
    return setup_file_logging(log_dir)


def enable_production_logging(log_dir: str = '/var/log/medical_datasets'):
    """Enable production logging"""
    setup_production_logging(log_dir)


if __name__ == "__main__":
    # Demo logging configurations
    logger = get_logger(__name__)
    
    logger.info("Testing default logging")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    print("\n" + "="*50)
    print("Testing file logging...")
    log_file = setup_file_logging('test_logs')
    logger.info("File logging enabled")
    
    print("\n" + "="*50)
    print("Testing debug logging...")
    setup_debug_logging()
    logger.debug("Debug message")
    logger.info("Info message")