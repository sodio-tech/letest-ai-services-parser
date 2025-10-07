"""
Logging Module
Configures and manages logging for the CSV parser
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from .config import ConfigManager
from .env_manager import EnvManager


class LoggerSetup:
    """Sets up and configures logging for the application"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, 
                 env_manager: Optional[EnvManager] = None):
        """
        Initialize logger setup
        
        Args:
            config_manager: Configuration manager instance
            env_manager: Environment manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.env_manager = env_manager or EnvManager()
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """
        Set up logging configuration
        """
        try:
            # Get logging configuration
            log_config = self.env_manager.get_logging_config()
            config_log_config = self.config_manager.get_logging_config()
            
            # Merge configurations (env takes precedence)
            log_config.update({k: v for k, v in config_log_config.items() if k not in log_config})
            
            # Set up logger
            self.logger = logging.getLogger('csv_parser')
            self.logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
            
            # Clear existing handlers
            self.logger.handlers.clear()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (if log file is specified)
            log_file = log_config.get('file')
            if log_file:
                # Create log directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Create rotating file handler
                max_size = self._parse_size(log_config.get('max_size', '10MB'))
                backup_count = log_config.get('backup_count', 5)
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            # Set up root logger to prevent duplicate messages
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.WARNING)
            
            self.logger.info("Logging configured successfully")
            
        except Exception as e:
            # Fallback to basic logging if setup fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('csv_parser')
            self.logger.error(f"Error setting up logging: {str(e)}")
    
    def _parse_size(self, size_str: str) -> int:
        """
        Parse size string to bytes
        
        Args:
            size_str: Size string (e.g., "10MB", "1GB")
            
        Returns:
            Size in bytes
        """
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance
        
        Args:
            name: Logger name (optional)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f'csv_parser.{name}')
        return self.logger
    
    def set_level(self, level: str) -> None:
        """
        Set logging level
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            log_level = getattr(logging, level.upper())
            self.logger.setLevel(log_level)
            
            # Update all handlers
            for handler in self.logger.handlers:
                handler.setLevel(log_level)
                
            self.logger.info(f"Logging level set to {level}")
            
        except AttributeError:
            self.logger.error(f"Invalid logging level: {level}")
    
    def add_file_handler(self, file_path: str, level: str = "DEBUG") -> None:
        """
        Add file handler to logger
        
        Args:
            file_path: Path to log file
            level: Logging level for file handler
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            self.logger.info(f"File handler added: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding file handler: {str(e)}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary containing logging statistics
        """
        stats = {
            "logger_name": self.logger.name,
            "logger_level": self.logger.level,
            "handlers_count": len(self.logger.handlers),
            "handlers": []
        }
        
        for handler in self.logger.handlers:
            handler_info = {
                "type": type(handler).__name__,
                "level": handler.level,
                "formatter": type(handler.formatter).__name__ if handler.formatter else None
            }
            
            if hasattr(handler, 'baseFilename'):
                handler_info["filename"] = handler.baseFilename
            
            stats["handlers"].append(handler_info)
        
        return stats


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance (convenience function)
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'csv_parser.{name}' if name else 'csv_parser')
