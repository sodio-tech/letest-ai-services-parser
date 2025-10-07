"""
Configuration Management Module
Handles loading and managing configuration settings
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration settings for the CSV parser"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Configuration loaded from {self.config_path}")
                    return config or {}
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration settings
        
        Returns:
            Default configuration dictionary
        """
        return {
            "csv_parser": {
                "sample_percentage": 0.2,
                "max_file_size": 10485760,  # 10MB
                "supported_encodings": ["utf-8", "latin1", "cp1252"],
                "output_format": "json"
            },
            "llm_settings": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30,
                "retry_attempts": 3
            },
            "pattern_recognition": {
                "confidence_threshold": 0.85,
                "max_patterns": 10,
                "validation_sample_size": 50
            },
            "logging": {
                "level": "INFO",
                "file": "logs/csv_parser.log",
                "max_size": "10MB",
                "backup_count": 5
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value for key '{key}': {str(e)}")
            return default
    
    def get_csv_parser_config(self) -> Dict[str, Any]:
        """
        Get CSV parser specific configuration
        
        Returns:
            CSV parser configuration dictionary
        """
        return self.get("csv_parser", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM specific configuration
        
        Returns:
            LLM configuration dictionary
        """
        return self.get("llm_settings", {})
    
    def get_pattern_config(self) -> Dict[str, Any]:
        """
        Get pattern recognition configuration
        
        Returns:
            Pattern recognition configuration dictionary
        """
        return self.get("pattern_recognition", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration
        
        Returns:
            Logging configuration dictionary
        """
        return self.get("logging", {})
    
    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value to set
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            logger.info(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Error updating config value for key '{key}': {str(e)}")
    
    def save_config(self) -> bool:
        """
        Save current configuration to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def reload_config(self) -> None:
        """
        Reload configuration from file
        """
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration settings
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
