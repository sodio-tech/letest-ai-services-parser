"""
API Dependencies - Dependency injection for FastAPI
"""

from functools import lru_cache
from typing import Generator
from .services import ProcessingService, FileService, ConfigurationService
from .external_api_service import ExternalAPIService
from src.utils.config import ConfigManager
from src.utils.env_manager import EnvManager

# Global instances (singleton pattern)
_config_manager = None
_env_manager = None
_processing_service = None
_file_service = None
_config_service = None
_external_api_service = None


def get_config_manager() -> ConfigManager:
    """Get configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_env_manager() -> EnvManager:
    """Get environment manager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvManager()
    return _env_manager


def get_processing_service() -> ProcessingService:
    """Get processing service instance"""
    global _processing_service
    if _processing_service is None:
        config_manager = get_config_manager()
        env_manager = get_env_manager()
        _processing_service = ProcessingService(config_manager, env_manager)
    return _processing_service


def get_file_service() -> FileService:
    """Get file service instance"""
    global _file_service
    if _file_service is None:
        _file_service = FileService()
    return _file_service


def get_config_service() -> ConfigurationService:
    """Get configuration service instance"""
    global _config_service
    if _config_service is None:
        config_manager = get_config_manager()
        _config_service = ConfigurationService(config_manager)
    return _config_service


def get_external_api_service() -> ExternalAPIService:
    """Get external API service instance"""
    global _external_api_service
    if _external_api_service is None:
        env_manager = get_env_manager()
        _external_api_service = ExternalAPIService(env_manager)
    return _external_api_service
