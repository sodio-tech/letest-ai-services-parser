"""
API Models - Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    message: str
    file_id: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None


class ProcessingRequest(BaseModel):
    """Request model for CSV processing"""
    file_id: str
    output_format: str = Field(default="json", description="Output format (json)")
    include_metadata: bool = Field(default=False, description="Include processing metadata")
    sample_percentage: float = Field(default=0.2, ge=0.1, le=1.0, description="Percentage of data to sample for analysis")


class ProcessingResponse(BaseModel):
    """Response model for processing results"""
    success: bool
    message: str
    processing_id: Optional[str] = None
    status: Optional[ProcessingStatus] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""
    file_ids: List[str] = Field(..., min_items=1, description="List of file IDs to process")
    output_format: str = Field(default="json", description="Output format (json)")
    include_metadata: bool = Field(default=False, description="Include processing metadata")
    sample_percentage: float = Field(default=0.2, ge=0.1, le=1.0, description="Percentage of data to sample for analysis")


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing results"""
    success: bool
    message: str
    batch_id: Optional[str] = None
    total_files: int
    processed_files: int
    failed_files: int
    results: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[str]] = None


class StatusResponse(BaseModel):
    """Response model for processing status"""
    processing_id: str
    status: ProcessingStatus
    progress: Optional[float] = Field(ge=0.0, le=100.0, description="Processing progress percentage")
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class CSVInfoResponse(BaseModel):
    """Response model for CSV file information"""
    success: bool
    file_info: Optional[Dict[str, Any]] = None
    column_info: Optional[Dict[str, Any]] = None
    sample_data: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None


class PatternAnalysisResponse(BaseModel):
    """Response model for pattern analysis"""
    success: bool
    pattern_info: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    examples: Optional[List[str]] = None
    error: Optional[str] = None


class JSONOutputResponse(BaseModel):
    """Response model for JSON output"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    total_objects: Optional[int] = None
    validation_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ConfigurationRequest(BaseModel):
    """Request model for configuration updates"""
    csv_parser: Optional[Dict[str, Any]] = None
    llm_settings: Optional[Dict[str, Any]] = None
    pattern_recognition: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None


class ConfigurationResponse(BaseModel):
    """Response model for configuration"""
    success: bool
    configuration: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class FileInfo(BaseModel):
    """Model for file information"""
    file_id: str
    filename: str
    file_size: int
    upload_time: str
    content_type: str
    status: ProcessingStatus = ProcessingStatus.PENDING


class ProcessingJob(BaseModel):
    """Model for processing job"""
    job_id: str
    file_id: str
    status: ProcessingStatus
    progress: float = 0.0
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


# New models for single API endpoint
class SingleProcessingRequest(BaseModel):
    """Request model for single-step CSV processing"""
    output_format: str = Field(default="json", description="Output format (json)")
    include_metadata: bool = Field(default=False, description="Include processing metadata")
    sample_percentage: float = Field(default=0.2, ge=0.1, le=1.0, description="Percentage of data to sample for analysis")
    user_id: int = Field(..., description="User ID")
    project_id: int = Field(..., description="Project ID")
    environment_id: int = Field(..., description="Environment ID")
    category_id: int = Field(..., description="Category ID")
    columns: str = Field(default="{}", description="JSON string for column mapping")


class SingleProcessingResponse(BaseModel):
    """Response model for single-step processing results"""
    user_id: int
    project_id: int
    environment_id: int
    category_id: int
    total_objects: int
    data: List[Dict[str, Any]]


# New models for external API upload
class UploadTestCasesRequest(BaseModel):
    """Request model for uploading test cases to external API"""
    user_id: int = Field(..., description="User ID")
    project_id: int = Field(..., description="Project ID")
    environment_id: int = Field(..., description="Environment ID")
    category_id: int = Field(..., description="Category ID")
    total_objects: int = Field(..., description="Total number of test case objects")
    data: List[Dict[str, Any]] = Field(..., description="List of test case data")


class UploadTestCasesResponse(BaseModel):
    """Response model for test cases upload"""
    success: bool
    message: str
    status_code: Optional[int] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProcessAndUploadRequest(BaseModel):
    """Request model for processing CSV and uploading to external API"""
    user_id: int = Field(..., description="User ID")
    project_id: int = Field(..., description="Project ID")
    environment_id: int = Field(..., description="Environment ID")
    category_id: int = Field(..., description="Category ID")
    output_format: str = Field(default="json", description="Output format (json)")
    include_metadata: bool = Field(default=False, description="Include processing metadata")
    sample_percentage: float = Field(default=0.2, ge=0.1, le=1.0, description="Percentage of data to sample for analysis")
    columns: str = Field(default="{}", description="JSON string for column mapping")
    upload_to_external: bool = Field(default=True, description="Whether to upload to external API after processing")


class ProcessAndUploadResponse(BaseModel):
    """Response model for process and upload operation"""
    success: bool
    message: str
    processing_results: Optional[Dict[str, Any]] = None
    upload_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None