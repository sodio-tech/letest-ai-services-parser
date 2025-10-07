"""
API Services - Business logic for CSV processing operations
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import aiofiles
import pandas as pd
from pathlib import Path

# Import from main application
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.csv_reader import CSVReader
from src.core.header_detector import HeaderDetector
from src.core.llm_analyzer import LLMAnalyzer
from src.transformers.json_generator import JSONGenerator
from src.ai.openai_client import OpenAIClient
from src.utils.config import ConfigManager
from src.utils.env_manager import EnvManager
from src.utils.logger import LoggerSetup


class FileService:
    """Service for file operations"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file, file_id: str) -> str:
        """Save uploaded file to disk"""
        file_path = self.upload_dir / f"{file_id}.csv"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return str(file_path)
    
    def get_file_path(self, file_id: str) -> str:
        """Get file path by ID"""
        file_path = self.upload_dir / f"{file_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_id} not found")
        return str(file_path)
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID"""
        file_path = self.upload_dir / f"{file_id}.csv"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_file_size(self, file_id: str) -> int:
        """Get file size by ID"""
        file_path = self.upload_dir / f"{file_id}.csv"
        if file_path.exists():
            return file_path.stat().st_size
        return 0


class ProcessingService:
    """Service for CSV processing operations"""
    
    def __init__(self, config_manager: ConfigManager, env_manager: EnvManager):
        self.config_manager = config_manager
        self.env_manager = env_manager
        self.logger_setup = LoggerSetup(config_manager, env_manager)
        self.logger = self.logger_setup.get_logger("processing")
        
        # Initialize LLM client
        self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize OpenAI client"""
        try:
            if not self.env_manager.is_openai_configured():
                raise ValueError("OpenAI API key not configured")
            
            llm_config = self.env_manager.get_openai_config()
            self.llm_client = OpenAIClient(**llm_config)
            self.logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    async def analyze_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file and return basic information"""
        try:
            # Read CSV file
            csv_reader = CSVReader(file_path, self.config_manager.get_csv_parser_config())
            
            if not csv_reader.is_loaded():
                return {"error": "Failed to load CSV file"}
            
            # Get file and column information
            file_info = csv_reader.get_file_info()
            column_info = csv_reader.get_column_info()
            
            # Get sample data
            sample_data = csv_reader.get_dataframe().head(3).to_dict('records')
            
            return {
                "file_info": file_info,
                "column_info": column_info,
                "sample_data": sample_data
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing CSV file: {str(e)}")
            return {"error": str(e)}
    
    async def process_csv_file(
        self, 
        file_path: str, 
        output_format: str = "json",
        include_metadata: bool = False,
        sample_percentage: float = 0.2,
        columns_mapping: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Process CSV file and convert to JSON"""
        try:
            self.logger.info(f"Starting CSV processing: {file_path}")
            
            # Update config with request parameters
            config = self.config_manager.get_csv_parser_config()
            config.update({
                "sample_percentage": sample_percentage,
                "output_format": output_format,
                "include_metadata": include_metadata
            })
            
            # Step 1: Read CSV file
            self.logger.info("Step 1: Reading CSV file...")
            csv_reader = CSVReader(file_path, config)
            
            if not csv_reader.is_loaded():
                return {"error": "Failed to load CSV file", "success": False}
            
            # Step 2: Detect headers using LLM or use provided mapping
            self.logger.info("Step 2: Detecting headers using LLM...")
            header_detector = HeaderDetector(
                csv_reader.get_dataframe(), 
                self.llm_client,
                config,
                columns_mapping
            )
            
            if not header_detector.is_ready():
                return {"error": "Failed to identify required columns", "success": False}
            
            headers = header_detector.get_all_headers()
            
            # Step 3: Analyze delimiter patterns using LLM
            self.logger.info("Step 3: Analyzing delimiter patterns using LLM...")
            test_steps_column = header_detector.get_test_steps_column()
            sample_data = csv_reader.get_text_samples(test_steps_column, 10)
            
            if not sample_data:
                return {"error": "No sample data available for pattern analysis", "success": False}
            
            llm_analyzer = LLMAnalyzer(
                self.llm_client, 
                sample_data, 
                self.config_manager.get_pattern_config()
            )
            
            if not llm_analyzer.is_ready():
                return {"error": "Failed to identify delimiter patterns", "success": False}
            
            pattern_info = llm_analyzer.get_pattern_info()
            
            # Step 4: Generate JSON output
            self.logger.info("Step 4: Generating JSON output...")
            json_generator = JSONGenerator(
                csv_reader.get_dataframe(),
                headers,
                self.llm_client,
                pattern_info,
                config
            )
            
            json_objects = json_generator.transform_to_json()
            
            if not json_objects:
                return {"error": "Failed to generate JSON output", "success": False}
            
            # Step 5: Validate output
            self.logger.info("Step 5: Validating JSON output...")
            validation_results = json_generator.validate_json_output(json_objects)
            
            # Prepare results
            results = {
                "success": True,
                "total_objects": len(json_objects),
                "json_objects": json_objects,
                "validation_results": validation_results,
                "pattern_info": pattern_info,
                "headers": headers,
                "file_info": csv_reader.get_file_info(),
                "transformation_stats": json_generator.get_transformation_stats()
            }
            
            self.logger.info("CSV processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def get_pattern_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get pattern analysis for a CSV file"""
        try:
            # Read CSV file
            csv_reader = CSVReader(file_path, self.config_manager.get_csv_parser_config())
            
            if not csv_reader.is_loaded():
                return {"error": "Failed to load CSV file"}
            
            # Detect headers
            header_detector = HeaderDetector(
                csv_reader.get_dataframe(), 
                self.llm_client,
                self.config_manager.get_csv_parser_config()
            )
            
            if not header_detector.is_ready():
                return {"error": "Failed to identify required columns"}
            
            # Get sample data for pattern analysis
            test_steps_column = header_detector.get_test_steps_column()
            sample_data = csv_reader.get_text_samples(test_steps_column, 10)
            
            if not sample_data:
                return {"error": "No sample data available for pattern analysis"}
            
            # Analyze patterns
            llm_analyzer = LLMAnalyzer(
                self.llm_client, 
                sample_data, 
                self.config_manager.get_pattern_config()
            )
            
            if not llm_analyzer.is_ready():
                return {"error": "Failed to identify delimiter patterns"}
            
            pattern_info = llm_analyzer.get_pattern_info()
            
            return {
                "success": True,
                "pattern_info": pattern_info,
                "confidence": llm_analyzer.get_confidence(),
                "examples": llm_analyzer.get_examples_found()
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}")
            return {"error": str(e)}


class ConfigurationService:
    """Service for configuration management"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config_manager.get_all_config()
    
    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        for key, value in updates.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.config_manager.update_config(f"{key}.{sub_key}", sub_value)
            else:
                self.config_manager.update_config(key, value)
        
        # Save configuration
        self.config_manager.save_config()
    
    def get_csv_parser_config(self) -> Dict[str, Any]:
        """Get CSV parser specific configuration"""
        return self.config_manager.get_csv_parser_config()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM specific configuration"""
        return self.config_manager.get_llm_config()
    
    def get_pattern_config(self) -> Dict[str, Any]:
        """Get pattern recognition configuration"""
        return self.config_manager.get_pattern_config()
