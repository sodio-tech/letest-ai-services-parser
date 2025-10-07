"""
Simplified API Server for AI CSV Parser Agent
A simpler version that avoids complex dependency injection issues
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.core.csv_reader import CSVReader
from src.core.header_detector import HeaderDetector
from src.core.llm_analyzer import LLMAnalyzer
from src.transformers.json_generator import JSONGenerator
from src.ai.openai_client import OpenAIClient
from src.utils.config import ConfigManager
from src.utils.env_manager import EnvManager
from src.utils.logger import LoggerSetup

# Initialize configuration and logging
config_manager = ConfigManager()
env_manager = EnvManager()
logger_setup = LoggerSetup(config_manager, env_manager)
logger = logger_setup.get_logger("simple_api")

# Create FastAPI app
app = FastAPI(
    title="AI CSV Parser Agent API",
    description="Intelligent CSV to JSON conversion using GPT-4o-mini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (in production, use Redis or database)
uploaded_files = {}
processing_jobs = {}

# Initialize LLM client
try:
    if not env_manager.is_openai_configured():
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
    
    llm_config = env_manager.get_openai_config()
    llm_client = OpenAIClient(**llm_config)
    logger.info("OpenAI client initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    llm_client = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI CSV Parser Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "openai": "connected" if llm_client else "disconnected",
            "file_storage": "active",
            "processing": "ready"
        }
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file for processing"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = uploads_dir / f"{file_id}.csv"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store file info
        file_info = {
            "file_id": file_id,
            "filename": file.filename,
            "file_size": file.size,
            "upload_time": datetime.utcnow().isoformat(),
            "file_path": str(file_path)
        }
        uploaded_files[file_id] = file_info
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": file.filename,
            "file_size": file.size
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/files/{file_id}/info")
async def get_file_info(file_id: str):
    """Get information about an uploaded CSV file"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = uploaded_files[file_id]
        file_path = file_info["file_path"]
        
        # Analyze CSV file
        csv_reader = CSVReader(file_path, config_manager.get_csv_parser_config())
        
        if not csv_reader.is_loaded():
            return {"success": False, "error": "Failed to load CSV file"}
        
        # Get file and column information
        file_info_data = csv_reader.get_file_info()
        column_info = csv_reader.get_column_info()
        sample_data = csv_reader.get_dataframe().head(3).to_dict('records')
        
        return {
            "success": True,
            "file_info": file_info_data,
            "column_info": column_info,
            "sample_data": sample_data
        }
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/process")
async def process_csv(
    file_id: str,
    output_format: str = "json",
    include_metadata: bool = False,
    sample_percentage: float = 0.2,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process a CSV file and convert to JSON"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        if not llm_client:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized")
        
        # Generate processing job ID
        job_id = str(uuid.uuid4())
        
        # Create processing job
        job = {
            "job_id": job_id,
            "file_id": file_id,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "config": {
                "output_format": output_format,
                "include_metadata": include_metadata,
                "sample_percentage": sample_percentage
            }
        }
        processing_jobs[job_id] = job
        
        # Start background processing
        background_tasks.add_task(process_csv_background, job_id)
        
        return {
            "success": True,
            "message": "Processing started",
            "processing_id": job_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to start processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@app.get("/process/{job_id}/status")
async def get_processing_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = processing_jobs[job_id]
    return {
        "processing_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job.get("error") if job["status"] == "failed" else None,
        "results": job.get("results"),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at")
    }


@app.get("/process/{job_id}/result")
async def get_processing_result(job_id: str):
    """Get the result of a completed processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    return {
        "success": True,
        "data": job["results"].get("json_objects") if job["results"] else None,
        "total_objects": job["results"].get("total_objects") if job["results"] else 0,
        "validation_results": job["results"].get("validation_results") if job["results"] else None
    }


@app.get("/files")
async def list_uploaded_files():
    """List all uploaded files"""
    return list(uploaded_files.values())


async def process_csv_background(job_id: str):
    """Background task for processing a single CSV file"""
    try:
        # Update job status
        job = processing_jobs[job_id]
        job["status"] = "processing"
        job["progress"] = 10.0
        
        # Get file path
        file_id = job["file_id"]
        file_info = uploaded_files[file_id]
        file_path = file_info["file_path"]
        
        # Update config with request parameters
        config = config_manager.get_csv_parser_config()
        config.update({
            "sample_percentage": job["config"]["sample_percentage"],
            "output_format": job["config"]["output_format"],
            "include_metadata": job["config"]["include_metadata"]
        })
        
        # Step 1: Read CSV file
        job["progress"] = 20.0
        csv_reader = CSVReader(file_path, config)
        
        if not csv_reader.is_loaded():
            raise Exception("Failed to load CSV file")
        
        # Step 2: Detect headers using LLM
        job["progress"] = 40.0
        header_detector = HeaderDetector(
            csv_reader.get_dataframe(), 
            llm_client,
            config
        )
        
        if not header_detector.is_ready():
            raise Exception("Failed to identify required columns")
        
        headers = header_detector.get_all_headers()
        
        # Step 3: Analyze delimiter patterns using LLM
        job["progress"] = 60.0
        test_steps_column = header_detector.get_test_steps_column()
        sample_data = csv_reader.get_text_samples(test_steps_column, 10)
        
        if not sample_data:
            raise Exception("No sample data available for pattern analysis")
        
        llm_analyzer = LLMAnalyzer(
            llm_client, 
            sample_data, 
            config_manager.get_pattern_config()
        )
        
        if not llm_analyzer.is_ready():
            raise Exception("Failed to identify delimiter patterns")
        
        pattern_info = llm_analyzer.get_pattern_info()
        
        # Step 4: Generate JSON output
        job["progress"] = 80.0
        json_generator = JSONGenerator(
            csv_reader.get_dataframe(),
            headers,
            llm_client,
            pattern_info,
            config
        )
        
        json_objects = json_generator.transform_to_json()
        
        if not json_objects:
            raise Exception("Failed to generate JSON output")
        
        # Step 5: Validate output
        job["progress"] = 90.0
        validation_results = json_generator.validate_json_output(json_objects)
        
        # Update job with results
        job["progress"] = 100.0
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["results"] = {
            "total_objects": len(json_objects),
            "json_objects": json_objects,
            "validation_results": validation_results,
            "pattern_info": pattern_info,
            "headers": headers,
            "file_info": csv_reader.get_file_info(),
            "transformation_stats": json_generator.get_transformation_stats()
        }
        
        logger.info(f"Processing completed successfully for job {job_id}")
        
    except Exception as e:
        # Update job with error
        job = processing_jobs[job_id]
        job["status"] = "failed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["error"] = str(e)
        logger.error(f"Processing failed for job {job_id}: {str(e)}")


if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting simplified API server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "simple_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
