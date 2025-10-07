"""
Environment Manager Module
Handles environment variable loading and management
"""

import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EnvManager:
    """Manages environment variables and .env file loading"""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize environment manager
        
        Args:
            env_file: Path to the .env file
        """
        self.env_file = env_file
        self._load_env()
    
    def _load_env(self) -> None:
        """
        Load environment variables from .env file
        """
        try:
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"Environment variables loaded from {self.env_file}")
            else:
                logger.warning(f"Environment file {self.env_file} not found")
                
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable value
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def get_required(self, key: str) -> str:
        """
        Get required environment variable (raises error if not found)
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If environment variable is not set
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value
    
    def get_openai_config(self) -> Dict[str, Any]:
        """
        Get OpenAI configuration from environment variables
        
        Returns:
            Dictionary containing OpenAI configuration
        """
        return {
            "api_key": self.get_required("OPENAI_API_KEY"),
            "model": self.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": float(self.get("OPENAI_TEMPERATURE", "0.1")),
            "max_tokens": int(self.get("OPENAI_MAX_TOKENS", "2000")),
            "timeout": int(self.get("OPENAI_TIMEOUT", "30")),
            "retry_attempts": int(self.get("OPENAI_RETRY_ATTEMPTS", "3"))
        }
    
    def is_openai_configured(self) -> bool:
        """
        Check if OpenAI is properly configured
        
        Returns:
            True if OpenAI API key is set, False otherwise
        """
        return self.get("OPENAI_API_KEY") is not None
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration from environment variables
        
        Returns:
            Dictionary containing database configuration
        """
        return {
            "host": self.get("DB_HOST", "localhost"),
            "port": int(self.get("DB_PORT", "5432")),
            "name": self.get("DB_NAME", ""),
            "user": self.get("DB_USER", ""),
            "password": self.get("DB_PASSWORD", "")
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration from environment variables
        
        Returns:
            Dictionary containing logging configuration
        """
        return {
            "level": self.get("LOG_LEVEL", "INFO"),
            "file": self.get("LOG_FILE", "logs/csv_parser.log"),
            "max_size": self.get("LOG_MAX_SIZE", "10MB"),
            "backup_count": int(self.get("LOG_BACKUP_COUNT", "5"))
        }
    
    def set(self, key: str, value: str) -> None:
        """
        Set environment variable
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        os.environ[key] = value
        logger.info(f"Environment variable set: {key}")
    
    def reload(self) -> None:
        """
        Reload environment variables from .env file
        """
        self._load_env()
        logger.info("Environment variables reloaded")
    
    def create_env_template(self, template_path: str = "env_template.txt") -> bool:
        """
        Create environment template file
        
        Args:
            template_path: Path to create the template file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            template_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model settings
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_TEMPERATURE=0.1
# OPENAI_MAX_TOKENS=2000
# OPENAI_TIMEOUT=30
# OPENAI_RETRY_ATTEMPTS=3

# Logging Configuration (Optional)
# LOG_LEVEL=INFO
# LOG_FILE=logs/csv_parser.log
# LOG_MAX_SIZE=10MB
# LOG_BACKUP_COUNT=5

# Database Configuration (Optional)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=your_database_name
# DB_USER=your_username
# DB_PASSWORD=your_password
"""
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            logger.info(f"Environment template created at {template_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating environment template: {str(e)}")
            return False
