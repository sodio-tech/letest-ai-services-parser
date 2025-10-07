"""
AI CSV Parser Agent - Main Pipeline
Processes CSV files using gpt-4o-mini for intelligent pattern recognition
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.csv_reader import CSVReader
from src.core.header_detector import HeaderDetector
from src.core.llm_analyzer import LLMAnalyzer
from src.transformers.json_generator import JSONGenerator
from src.ai.openai_client import OpenAIClient
from src.utils.config import ConfigManager
from src.utils.env_manager import EnvManager
from src.utils.logger import LoggerSetup


class CSVParserAgent:
    """Main CSV Parser Agent using LLM for intelligent processing"""
    
    def __init__(self, config_path: str = "config.yaml", env_file: str = ".env"):
        """
        Initialize CSV Parser Agent
        
        Args:
            config_path: Path to configuration file
            env_file: Path to environment file
        """
        # Initialize managers
        self.config_manager = ConfigManager(config_path)
        self.env_manager = EnvManager(env_file)
        
        # Setup logging
        self.logger_setup = LoggerSetup(self.config_manager, self.env_manager)
        self.logger = self.logger_setup.get_logger("main")
        
        # Initialize components
        self.llm_client = None
        self.csv_reader = None
        self.header_detector = None
        self.llm_analyzer = None
        self.json_generator = None
        
        # Initialize LLM client
        self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> None:
        """Initialize OpenAI client"""
        try:
            if not self.env_manager.is_openai_configured():
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
            
            llm_config = self.env_manager.get_openai_config()
            self.llm_client = OpenAIClient(**llm_config)
            
            self.logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def process_csv(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process CSV file and convert to JSON format
        
        Args:
            file_path: Path to input CSV file
            output_path: Path to output JSON file (optional)
            
        Returns:
            Dictionary containing processing results
        """
        try:
            self.logger.info(f"Starting CSV processing: {file_path}")
            
            # Step 1: Read CSV file
            self.logger.info("Step 1: Reading CSV file...")
            self.csv_reader = CSVReader(file_path, self.config_manager.get_csv_parser_config())
            
            if not self.csv_reader.is_loaded():
                return {"error": "Failed to load CSV file", "success": False}
            
            self.logger.info(f"CSV loaded successfully: {self.csv_reader.get_file_info()}")
            
            # Step 2: Detect headers using LLM
            self.logger.info("Step 2: Detecting headers using LLM...")
            self.header_detector = HeaderDetector(
                self.csv_reader.get_dataframe(), 
                self.llm_client,
                self.config_manager.get_csv_parser_config()
            )
            
            if not self.header_detector.is_ready():
                return {"error": "Failed to identify required columns", "success": False}
            
            headers = self.header_detector.get_all_headers()
            self.logger.info(f"Headers identified: {headers}")
            
            # Step 3: Analyze delimiter patterns using LLM
            self.logger.info("Step 3: Analyzing delimiter patterns using LLM...")
            test_steps_column = self.header_detector.get_test_steps_column()
            sample_data = self.csv_reader.get_text_samples(test_steps_column, 10)
            
            if not sample_data:
                return {"error": "No sample data available for pattern analysis", "success": False}
            
            self.llm_analyzer = LLMAnalyzer(
                self.llm_client, 
                sample_data, 
                self.config_manager.get_pattern_config()
            )
            
            if not self.llm_analyzer.is_ready():
                return {"error": "Failed to identify delimiter patterns", "success": False}
            
            pattern_info = self.llm_analyzer.get_pattern_info()
            self.logger.info(f"Pattern analysis completed: {pattern_info}")
            
            # Step 4: Generate JSON output
            self.logger.info("Step 4: Generating JSON output...")
            self.json_generator = JSONGenerator(
                self.csv_reader.get_dataframe(),
                headers,
                self.llm_client,
                pattern_info,
                self.config_manager.get_csv_parser_config()
            )
            
            json_objects = self.json_generator.transform_to_json()
            
            if not json_objects:
                return {"error": "Failed to generate JSON output", "success": False}
            
            self.logger.info(f"JSON generation completed: {len(json_objects)} objects created")
            
            # Step 5: Validate output
            self.logger.info("Step 5: Validating JSON output...")
            validation_results = self.json_generator.validate_json_output(json_objects)
            self.logger.info(f"Validation results: {validation_results}")
            
            # Step 6: Save output if path provided
            if output_path:
                self.logger.info(f"Saving output to: {output_path}")
                if not self.json_generator.save_json_output(json_objects, output_path):
                    return {"error": "Failed to save JSON output", "success": False}
            
            # Prepare results
            results = {
                "success": True,
                "input_file": file_path,
                "output_file": output_path,
                "total_objects": len(json_objects),
                "validation_results": validation_results,
                "pattern_info": pattern_info,
                "headers": headers,
                "file_info": self.csv_reader.get_file_info(),
                "transformation_stats": self.json_generator.get_transformation_stats()
            }
            
            self.logger.info("CSV processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            return {"error": str(e), "success": False}
    
    def process_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process multiple CSV files in a directory
        
        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save JSON outputs
            
        Returns:
            Dictionary containing batch processing results
        """
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            if not input_path.exists():
                return {"error": f"Input directory does not exist: {input_dir}", "success": False}
            
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find CSV files
            csv_files = list(input_path.glob("*.csv"))
            if not csv_files:
                return {"error": f"No CSV files found in {input_dir}", "success": False}
            
            self.logger.info(f"Found {len(csv_files)} CSV files to process")
            
            results = {
                "success": True,
                "total_files": len(csv_files),
                "processed_files": 0,
                "failed_files": 0,
                "file_results": []
            }
            
            # Process each file
            for csv_file in csv_files:
                self.logger.info(f"Processing file: {csv_file.name}")
                
                output_file = output_path / f"{csv_file.stem}.json"
                
                file_result = self.process_csv(str(csv_file), str(output_file))
                
                if file_result.get("success", False):
                    results["processed_files"] += 1
                    self.logger.info(f"Successfully processed: {csv_file.name}")
                else:
                    results["failed_files"] += 1
                    self.logger.error(f"Failed to process: {csv_file.name} - {file_result.get('error', 'Unknown error')}")
                
                results["file_results"].append({
                    "file": csv_file.name,
                    "success": file_result.get("success", False),
                    "error": file_result.get("error"),
                    "objects_count": file_result.get("total_objects", 0)
                })
            
            self.logger.info(f"Batch processing completed: {results['processed_files']}/{results['total_files']} files processed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return {"error": str(e), "success": False}


def main():
    """Main entry point for the CSV Parser Agent"""
    parser = argparse.ArgumentParser(description="AI CSV Parser Agent - Intelligent CSV to JSON conversion")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("-o", "--output", help="Output JSON file or directory")
    parser.add_argument("-c", "--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("-e", "--env", default=".env", help="Environment file path")
    parser.add_argument("--batch", action="store_true", help="Process all CSV files in input directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = CSVParserAgent(args.config, args.env)
        
        # Set verbose logging if requested
        if args.verbose:
            agent.logger_setup.set_level("DEBUG")
        
        # Process files
        if args.batch:
            # Batch processing
            if not args.output:
                args.output = "output"
            
            results = agent.process_batch(args.input, args.output)
        else:
            # Single file processing
            results = agent.process_csv(args.input, args.output)
        
        # Print results
        if results.get("success", False):
            print("✅ Processing completed successfully!")
            print(f"📊 Results: {json.dumps(results, indent=2)}")
        else:
            print("❌ Processing failed!")
            print(f"🚨 Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
