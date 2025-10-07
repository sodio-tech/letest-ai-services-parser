"""
API Exceptions - Custom exception classes for the API
"""

from typing import Optional, Dict, Any


class CSVParserAPIException(Exception):
    """Base exception for CSV Parser API"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class FileNotFoundError(CSVParserAPIException):
    """Exception raised when a file is not found"""
    
    def __init__(self, file_id: str):
        super().__init__(
            message=f"File {file_id} not found",
            error_code="FILE_NOT_FOUND",
            details={"file_id": file_id}
        )


class FileUploadError(CSVParserAPIException):
    """Exception raised when file upload fails"""
    
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to upload file {filename}: {reason}",
            error_code="FILE_UPLOAD_ERROR",
            details={"filename": filename, "reason": reason}
        )


class ProcessingError(CSVParserAPIException):
    """Exception raised when CSV processing fails"""
    
    def __init__(self, file_id: str, reason: str):
        super().__init__(
            message=f"Failed to process file {file_id}: {reason}",
            error_code="PROCESSING_ERROR",
            details={"file_id": file_id, "reason": reason}
        )


class ConfigurationError(CSVParserAPIException):
    """Exception raised when configuration is invalid"""
    
    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Configuration error for {config_key}: {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key, "reason": reason}
        )


class OpenAIError(CSVParserAPIException):
    """Exception raised when OpenAI API fails"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"OpenAI API error: {reason}",
            error_code="OPENAI_ERROR",
            details={"reason": reason}
        )


class ValidationError(CSVParserAPIException):
    """Exception raised when data validation fails"""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Validation error for {field}: {reason}",
            error_code="VALIDATION_ERROR",
            details={"field": field, "reason": reason}
        )


class RateLimitError(CSVParserAPIException):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, limit: int, window: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            error_code="RATE_LIMIT_ERROR",
            details={"limit": limit, "window": window}
        )
