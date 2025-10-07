"""
API Routes - FastAPI route handlers for CSV processing
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
import aiofiles
import json

from .models import (
    FileUploadResponse, ProcessingRequest, ProcessingResponse, BatchProcessingRequest,
    BatchProcessingResponse, StatusResponse, HealthResponse, ErrorResponse,
    CSVInfoResponse, PatternAnalysisResponse, JSONOutputResponse,
    ConfigurationRequest, ConfigurationResponse, FileInfo, ProcessingJob,
    ProcessingStatus, SingleProcessingRequest, SingleProcessingResponse,
    UploadTestCasesRequest, UploadTestCasesResponse, ProcessAndUploadRequest, ProcessAndUploadResponse
)
from .services import ProcessingService, FileService, ConfigurationService
from .external_api_service import ExternalAPIService
from .websocket_manager import websocket_manager
from .dependencies import get_processing_service, get_file_service, get_config_service, get_external_api_service

# Create router
router = APIRouter()

# Setup logger
logger = logging.getLogger(__name__)

# In-memory storage for demo (in production, use Redis or database)
processing_jobs = {}
uploaded_files = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services={
            "openai": "connected",
            "file_storage": "active",
            "processing": "ready"
        }
    )


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service)
):
    """Upload a CSV file for processing"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save file
        file_path = await file_service.save_uploaded_file(file, file_id)
        
        # Store file info
        file_info = FileInfo(
            file_id=file_id,
            filename=file.filename,
            file_size=file.size,
            upload_time=datetime.utcnow().isoformat(),
            content_type=file.content_type or "text/csv",
            status=ProcessingStatus.PENDING
        )
        uploaded_files[file_id] = file_info
        
        return FileUploadResponse(
            success=True,
            message="File uploaded successfully",
            file_id=file_id,
            filename=file.filename,
            file_size=file.size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/files/{file_id}/info", response_model=CSVInfoResponse)
async def get_file_info(
    file_id: str,
    file_service: FileService = Depends(get_file_service),
    processing_service: ProcessingService = Depends(get_processing_service)
):
    """Get information about an uploaded CSV file"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = uploaded_files[file_id]
        file_path = file_service.get_file_path(file_id)
        
        # Analyze CSV file
        analysis_result = await processing_service.analyze_csv_file(file_path)
        
        return CSVInfoResponse(
            success=True,
            file_info=analysis_result.get("file_info"),
            column_info=analysis_result.get("column_info"),
            sample_data=analysis_result.get("sample_data")
        )
        
    except Exception as e:
        return CSVInfoResponse(
            success=False,
            error=str(e)
        )


@router.post("/process", response_model=ProcessingResponse)
async def process_csv(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    processing_service: ProcessingService = Depends(get_processing_service),
    file_service: FileService = Depends(get_file_service)
):
    """Process a CSV file and convert to JSON"""
    try:
        if request.file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Generate processing job ID
        job_id = str(uuid.uuid4())
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            file_id=request.file_id,
            status=ProcessingStatus.PENDING,
            created_at=datetime.utcnow().isoformat(),
            config={
                "output_format": request.output_format,
                "include_metadata": request.include_metadata,
                "sample_percentage": request.sample_percentage
            }
        )
        processing_jobs[job_id] = job
        
        # Start background processing
        background_tasks.add_task(
            process_csv_background,
            job_id,
            request,
            processing_service,
            file_service
        )
        
        return ProcessingResponse(
            success=True,
            message="Processing started",
            processing_id=job_id,
            status=ProcessingStatus.PROCESSING
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.get("/process/{job_id}/status", response_model=StatusResponse)
async def get_processing_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = processing_jobs[job_id]
    return StatusResponse(
        processing_id=job_id,
        status=job.status,
        progress=job.progress,
        message=job.error if job.status == ProcessingStatus.FAILED else None,
        results=job.results,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at
    )


@router.get("/process/{job_id}/result", response_model=JSONOutputResponse)
async def get_processing_result(job_id: str):
    """Get the result of a completed processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = processing_jobs[job_id]
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    return JSONOutputResponse(
        success=True,
        data=job.results.get("json_objects") if job.results else None,
        total_objects=job.results.get("total_objects") if job.results else 0,
        validation_results=job.results.get("validation_results") if job.results else None
    )


@router.post("/process/batch", response_model=BatchProcessingResponse)
async def process_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    processing_service: ProcessingService = Depends(get_processing_service),
    file_service: FileService = Depends(get_file_service)
):
    """Process multiple CSV files in batch"""
    try:
        # Validate all files exist
        for file_id in request.file_ids:
            if file_id not in uploaded_files:
                raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Create individual processing jobs
        job_ids = []
        for file_id in request.file_ids:
            job_id = str(uuid.uuid4())
            job = ProcessingJob(
                job_id=job_id,
                file_id=file_id,
                status=ProcessingStatus.PENDING,
                created_at=datetime.utcnow().isoformat(),
                config={
                    "output_format": request.output_format,
                    "include_metadata": request.include_metadata,
                    "sample_percentage": request.sample_percentage
                }
            )
            processing_jobs[job_id] = job
            job_ids.append(job_id)
        
        # Start background batch processing
        background_tasks.add_task(
            process_batch_background,
            batch_id,
            job_ids,
            request,
            processing_service,
            file_service
        )
        
        return BatchProcessingResponse(
            success=True,
            message="Batch processing started",
            batch_id=batch_id,
            total_files=len(request.file_ids),
            processed_files=0,
            failed_files=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")


@router.get("/files", response_model=List[FileInfo])
async def list_uploaded_files():
    """List all uploaded files"""
    return list(uploaded_files.values())


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    """Delete an uploaded file"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete file from storage
        await file_service.delete_file(file_id)
        
        # Remove from memory
        del uploaded_files[file_id]
        
        return {"success": True, "message": "File deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/config", response_model=ConfigurationResponse)
async def get_configuration(config_service: ConfigurationService = Depends(get_config_service)):
    """Get current configuration"""
    try:
        config = config_service.get_configuration()
        return ConfigurationResponse(
            success=True,
            configuration=config,
            message="Configuration retrieved successfully"
        )
    except Exception as e:
        return ConfigurationResponse(
            success=False,
            error=str(e)
        )


@router.put("/config", response_model=ConfigurationResponse)
async def update_configuration(
    request: ConfigurationRequest,
    config_service: ConfigurationService = Depends(get_config_service)
):
    """Update configuration"""
    try:
        config_service.update_configuration(request.dict(exclude_unset=True))
        return ConfigurationResponse(
            success=True,
            message="Configuration updated successfully"
        )
    except Exception as e:
        return ConfigurationResponse(
            success=False,
            error=str(e)
        )


# Background task functions
async def process_csv_background(
    job_id: str,
    request: ProcessingRequest,
    processing_service: ProcessingService,
    file_service: FileService
):
    """Background task for processing a single CSV file"""
    try:
        # Update job status
        job = processing_jobs[job_id]
        job.status = ProcessingStatus.PROCESSING
        job.started_at = datetime.utcnow().isoformat()
        job.progress = 10.0
        
        # Get file path
        file_path = file_service.get_file_path(request.file_id)
        
        # Process the file
        job.progress = 30.0
        result = await processing_service.process_csv_file(
            file_path,
            request.output_format,
            request.include_metadata,
            request.sample_percentage
        )
        
        # Update job with results
        job.progress = 100.0
        job.status = ProcessingStatus.COMPLETED
        job.completed_at = datetime.utcnow().isoformat()
        job.results = result
        
    except Exception as e:
        # Update job with error
        job = processing_jobs[job_id]
        job.status = ProcessingStatus.FAILED
        job.completed_at = datetime.utcnow().isoformat()
        job.error = str(e)


async def process_batch_background(
    batch_id: str,
    job_ids: List[str],
    request: BatchProcessingRequest,
    processing_service: ProcessingService,
    file_service: FileService
):
    """Background task for batch processing"""
    processed_count = 0
    failed_count = 0
    
    for job_id in job_ids:
        try:
            job = processing_jobs[job_id]
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow().isoformat()
            
            # Process the file
            file_path = file_service.get_file_path(job.file_id)
            result = await processing_service.process_csv_file(
                file_path,
                request.output_format,
                request.include_metadata,
                request.sample_percentage
            )
            
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow().isoformat()
            job.results = result
            processed_count += 1
            
        except Exception as e:
            job = processing_jobs[job_id]
            job.status = ProcessingStatus.FAILED
            job.completed_at = datetime.utcnow().isoformat()
            job.error = str(e)
            failed_count += 1


# WebSocket endpoint for real-time processing updates
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time CSV processing updates"""
    await websocket_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages if needed
            if message.get("type") == "ping":
                await websocket_manager.send_message(session_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket_manager.disconnect(session_id)

# New single API endpoint with WebSocket support
@router.post("/process-csv")
async def process_csv_single(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    project_id: int = Form(...),
    environment_id: int = Form(...),
    category_id: int = Form(...),
    columns: str = Form("{}"),
    output_format: str = Form("json"),
    include_metadata: bool = Form(False),
    sample_percentage: float = Form(0.2),
    upload_to_external: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    processing_service: ProcessingService = Depends(get_processing_service),
    file_service: FileService = Depends(get_file_service),
    external_api_service: ExternalAPIService = Depends(get_external_api_service)
):
    """
    Single API endpoint for complete CSV processing with WebSocket support
    Returns WebSocket URL for real-time updates
    """
    import time
    import tempfile
    import os
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Check file size (limit to 10MB for single processing)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            raise HTTPException(status_code=400, detail="File too large for single processing. Use /process endpoint for larger files.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Return WebSocket URL immediately
        websocket_url = f"/api/v1/ws/{session_id}"
        
        # Start background processing
        background_tasks.add_task(
            process_csv_with_websocket,
            session_id,
            temp_file_path,
            user_id,
            project_id,
            environment_id,
            category_id,
            columns,
            output_format,
            include_metadata,
            sample_percentage,
            upload_to_external,
            processing_service,
            external_api_service
        )
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "websocket_url": websocket_url,
            "message": "Processing started. Connect to WebSocket for real-time updates.",
            "status": "processing_started"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_csv_with_websocket(
    session_id: str,
    temp_file_path: str,
    user_id: int,
    project_id: int,
    environment_id: int,
    category_id: int,
    columns: str,
    output_format: str,
    include_metadata: bool,
    sample_percentage: float,
    upload_to_external: bool,
    processing_service: ProcessingService,
    external_api_service: ExternalAPIService
):
    """Background task for processing CSV with WebSocket updates"""
    import time
    import os
    import json
    
    start_time = time.time()
    
    try:
        # Wait for WebSocket connection
        await websocket_manager.send_log(session_id, "info", "Waiting for WebSocket connection...")
        
        # Wait up to 30 seconds for client to connect
        max_wait_time = 30
        wait_time = 0
        while session_id not in websocket_manager.active_connections and wait_time < max_wait_time:
            await asyncio.sleep(1)
            wait_time += 1
        
        if session_id not in websocket_manager.active_connections:
            await websocket_manager.send_error(session_id, "WebSocket connection timeout. Client did not connect within 30 seconds.")
            return
        
        await websocket_manager.send_log(session_id, "info", "WebSocket connected. Starting CSV processing...")
        
        # Parse columns parameter
        try:
            columns_mapping = json.loads(columns) if columns != "{}" else {}
        except json.JSONDecodeError:
            columns_mapping = {}
        
        # Step 1: Reading CSV file
        await websocket_manager.send_progress(session_id, "reading", 1, 5, "Reading CSV file...")
        await websocket_manager.send_log(session_id, "info", f"Reading CSV file: {temp_file_path}")
        
        # Step 2: Processing CSV file
        await websocket_manager.send_progress(session_id, "processing", 2, 5, "Processing CSV file...")
        await websocket_manager.send_log(session_id, "info", "Starting CSV processing with AI analysis...")
        
        result = await processing_service.process_csv_file(
            temp_file_path,
            output_format,
            include_metadata,
            sample_percentage,
            columns_mapping
        )
        
        processing_time = time.time() - start_time
        
        # Step 3: Processing complete
        await websocket_manager.send_progress(session_id, "processing", 3, 5, "CSV processing completed")
        await websocket_manager.send_log(session_id, "success", f"CSV processing completed in {processing_time:.2f} seconds")
        
        if result.get("success", False):
            total_objects = result.get("total_objects", 0)
            await websocket_manager.send_log(session_id, "info", f"Successfully processed {total_objects} test cases")
            
            # Step 4: Upload to external API (if requested)
            upload_result = None
            if upload_to_external:
                await websocket_manager.send_progress(session_id, "uploading", 4, 5, "Uploading to external API...")
                await websocket_manager.send_log(session_id, "info", "Uploading processed data to external API...")
                
                try:
                    upload_result = external_api_service.upload_test_cases(
                        user_id=user_id,
                        project_id=project_id,
                        environment_id=environment_id,
                        category_id=category_id,
                        total_objects=total_objects,
                        data=result.get("json_objects", [])
                    )
                    
                    if upload_result.get("success", False):
                        await websocket_manager.send_log(session_id, "success", f"Successfully uploaded {total_objects} test cases to external API")
                    else:
                        await websocket_manager.send_log(session_id, "error", f"Failed to upload to external API: {upload_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    await websocket_manager.send_log(session_id, "error", f"Error uploading to external API: {str(e)}")
                    upload_result = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                await websocket_manager.send_log(session_id, "info", "External API upload skipped")
            
            # Step 5: Final result
            await websocket_manager.send_progress(session_id, "complete", 5, 5, "Processing completed successfully")
            
            final_result = {
                "success": True,
                "user_id": user_id,
                "project_id": project_id,
                "environment_id": environment_id,
                "category_id": category_id,
                "total_objects": total_objects,
                "data": result.get("json_objects", []),
                "processing_time": processing_time,
                "upload_result": upload_result
            }
            
            await websocket_manager.send_final_result(session_id, final_result)
            await websocket_manager.send_log(session_id, "success", "Processing completed successfully!")
            
        else:
            error_msg = result.get("error", "Unknown processing error")
            await websocket_manager.send_error(session_id, f"CSV processing failed: {error_msg}")
            
    except Exception as e:
        await websocket_manager.send_error(session_id, f"Unexpected error during processing: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            await websocket_manager.send_log(session_id, "info", "Temporary file cleaned up")
        
        # Close WebSocket connection after a short delay
        await asyncio.sleep(2)
        await websocket_manager.disconnect(session_id)


# New endpoints for external API integration
@router.post("/upload-test-cases", response_model=UploadTestCasesResponse)
async def upload_test_cases(
    request: UploadTestCasesRequest,
    external_api_service: ExternalAPIService = Depends(get_external_api_service)
):
    """
    Upload processed test cases to external backend API
    """
    try:
        result = external_api_service.upload_test_cases(
            user_id=request.user_id,
            project_id=request.project_id,
            environment_id=request.environment_id,
            category_id=request.category_id,
            total_objects=request.total_objects,
            data=request.data
        )
        
        return UploadTestCasesResponse(
            success=result.get("success", False),
            message=result.get("message", "Upload completed"),
            status_code=result.get("status_code"),
            response=result.get("response"),
            error=result.get("error")
        )
        
    except Exception as e:
        return UploadTestCasesResponse(
            success=False,
            message="Upload failed",
            error=str(e)
        )


@router.post("/process-csv-and-upload", response_model=ProcessAndUploadResponse)
async def process_csv_and_upload(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    project_id: int = Form(...),
    environment_id: int = Form(...),
    category_id: int = Form(...),
    output_format: str = Form("json"),
    include_metadata: bool = Form(False),
    sample_percentage: float = Form(0.2),
    columns: str = Form("{}"),
    upload_to_external: bool = Form(True),
    processing_service: ProcessingService = Depends(get_processing_service),
    external_api_service: ExternalAPIService = Depends(get_external_api_service)
):
    """
    Process CSV file and optionally upload to external API in one request
    """
    import time
    import tempfile
    import os
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Check file size (limit to 10MB for single processing)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            raise HTTPException(status_code=400, detail="File too large for single processing. Use /process endpoint for larger files.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse columns parameter
            import json
            try:
                columns_mapping = json.loads(columns) if columns != "{}" else {}
            except json.JSONDecodeError:
                columns_mapping = {}
            
            # Process CSV file
            processing_result = await processing_service.process_csv_file(
                temp_file_path,
                output_format,
                include_metadata,
                sample_percentage,
                columns_mapping
            )
            
            processing_time = time.time() - start_time
            
            if not processing_result.get("success", False):
                return ProcessAndUploadResponse(
                    success=False,
                    message="CSV processing failed",
                    processing_results=processing_result,
                    error=processing_result.get("error", "Unknown processing error")
                )
            
            # Prepare upload results
            upload_results = None
            if upload_to_external:
                # Upload to external API
                upload_result = external_api_service.upload_test_cases(
                    user_id=user_id,
                    project_id=project_id,
                    environment_id=environment_id,
                    category_id=category_id,
                    total_objects=processing_result.get("total_objects", 0),
                    data=processing_result.get("json_objects", [])
                )
                upload_results = upload_result
                
                if not upload_result.get("success", False):
                    return ProcessAndUploadResponse(
                        success=False,
                        message="CSV processing succeeded but upload failed",
                        processing_results=processing_result,
                        upload_results=upload_result,
                        error=upload_result.get("error", "Upload failed")
                    )
            
            return ProcessAndUploadResponse(
                success=True,
                message="Processing and upload completed successfully",
                processing_results=processing_result,
                upload_results=upload_results
            )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        return ProcessAndUploadResponse(
            success=False,
            message="Processing and upload failed",
            error=str(e)
        )


@router.get("/external-api/test-connection")
async def test_external_api_connection(
    external_api_service: ExternalAPIService = Depends(get_external_api_service)
):
    """
    Test connection to external backend API
    """
    try:
        result = external_api_service.test_connection()
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Connection test completed"),
            "error": result.get("error")
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Connection test failed",
            "error": str(e)
        }
