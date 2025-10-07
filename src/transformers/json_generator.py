"""
JSON Generator Module
Transforms CSV data into structured JSON format with LLM validation
"""

import pandas as pd
import json
import logging
from typing import Dict, Any, List, Optional
from ..ai.openai_client import OpenAIClient
from ..ai.prompt_engineer import PromptEngineer

logger = logging.getLogger(__name__)


class JSONGenerator:
    """Transforms CSV data to JSON format with LLM validation"""
    
    def __init__(self, csv_data: pd.DataFrame, headers: Dict[str, str], 
                 llm_client: OpenAIClient, pattern_info: Dict[str, Any], 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON generator
        
        Args:
            csv_data: DataFrame containing CSV data
            headers: Dictionary mapping column types to column names
            llm_client: OpenAI client for LLM interactions
            pattern_info: Information about delimiter patterns
            config: Configuration dictionary
        """
        self.csv_data = csv_data
        self.headers = headers
        self.llm_client = llm_client
        self.pattern_info = pattern_info
        self.config = config or {}
        self.prompt_engineer = PromptEngineer()
        self.output_format = self.config.get('output_format', 'json')
    
    def transform_to_json(self) -> List[Dict[str, Any]]:
        """
        Transform CSV data to JSON format
        
        Returns:
            List of JSON objects representing test cases
        """
        if self.csv_data is None or self.csv_data.empty:
            logger.error("No data available for transformation")
            return []
        
        if not self.headers.get("test_steps"):
            logger.error("No test steps column identified")
            return []
        
        try:
            json_objects = []
            
            for index, row in self.csv_data.iterrows():
                try:
                    # Generate JSON object for this row
                    json_obj = self.generate_output_format(row)
                    if json_obj:
                        json_objects.append(json_obj)
                        
                except Exception as e:
                    logger.warning(f"Error processing row {index}: {str(e)}")
                    continue
            
            logger.info(f"Successfully transformed {len(json_objects)} rows to JSON")
            return json_objects
            
        except Exception as e:
            logger.error(f"Error transforming data to JSON: {str(e)}")
            return []
    
    def generate_output_format(self, row_data: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Generate the required JSON structure for a single row
        
        Args:
            row_data: Pandas Series representing a single row
            
        Returns:
            JSON object or None if generation fails
        """
        try:
            # Extract basic information
            test_title = self._get_column_value(row_data, self.headers.get("test_title"))
            test_steps_text = self._get_column_value(row_data, self.headers.get("test_steps"))
            expected_outcome = self._get_column_value(row_data, self.headers.get("expected_outcome"))
            
            # Extract individual steps using LLM
            steps = self._extract_steps(test_steps_text)
            
            # Create JSON structure (keeping everything lowercase as requested)
            json_obj = {
                    "test_title": test_title.lower() if test_title else "",
                    "steps": [step.lower() for step in steps] if steps else [],
                    "expected_outcome": expected_outcome.lower() if expected_outcome else ""
            }
            # Add metadata if configured
            if self.config.get('include_metadata', False):
                json_obj["metadata"] = {
                    "row_index": int(row_data.name) if hasattr(row_data, 'name') else 0,
                    "pattern_used": self.pattern_info.get("pattern_type", "unknown"),
                    "confidence": self.pattern_info.get("confidence", 0.0),
                    "steps_count": len(steps)
                }
            
            return json_obj
            
        except Exception as e:
            logger.error(f"Error generating JSON for row: {str(e)}")
            return None
    
    def _get_column_value(self, row_data: pd.Series, column_name: Optional[str]) -> str:
        """
        Get value from a specific column, handling missing columns
        
        Args:
            row_data: Pandas Series representing a single row
            column_name: Name of the column to extract
            
        Returns:
            Column value as string or empty string if not found
        """
        if not column_name or column_name not in row_data.index:
            return ""
        
        value = row_data[column_name]
        
        # Handle NaN values
        if pd.isna(value):
            return ""
        
        # Convert to string and strip whitespace
        return str(value).strip()
    
    def _extract_steps(self, steps_text: str) -> List[str]:
        """
        Extract individual steps using identified pattern and LLM
        
        Args:
            steps_text: Text containing multiple steps
            
        Returns:
            List of extracted steps
        """
        if not steps_text or not steps_text.strip():
            return []
        
        try:
            # Use LLM for intelligent step extraction
            extracted_steps = self.llm_client.extract_steps_with_llm(steps_text, self.pattern_info)
            
            if extracted_steps:
                # Validate extraction with LLM
                validation = self.llm_client.validate_extraction(steps_text, extracted_steps)
                
                if validation.get("is_accurate", False):
                    return extracted_steps
                else:
                    logger.warning(f"LLM validation failed: {validation.get('suggestions', 'Unknown issue')}")
            
            # Fallback to regex-based extraction
            return self._extract_steps_with_regex(steps_text)
            
        except Exception as e:
            logger.error(f"Error extracting steps with LLM: {str(e)}")
            return self._extract_steps_with_regex(steps_text)
    
    def _extract_steps_with_regex(self, steps_text: str) -> List[str]:
        """
        Extract steps using regex pattern as fallback
        
        Args:
            steps_text: Text containing multiple steps
            
        Returns:
            List of extracted steps
        """
        if not steps_text or not self.pattern_info:
            return [steps_text] if steps_text else []
        
        import re
        
        regex_pattern = self.pattern_info.get("regex_pattern", r'\n')
        
        try:
            # Split text using the regex pattern
            steps = re.split(regex_pattern, steps_text)
            
            # Clean up steps
            cleaned_steps = []
            for step in steps:
                step = step.strip()
                if step:  # Only add non-empty steps
                    cleaned_steps.append(step)
            
            return cleaned_steps if cleaned_steps else [steps_text]
            
        except Exception as e:
            logger.error(f"Error in regex extraction: {str(e)}")
            return [steps_text]
    
    def validate_json_output(self, json_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the generated JSON output
        
        Args:
            json_objects: List of generated JSON objects
            
        Returns:
            Dictionary containing validation results
        """
        if not json_objects:
            return {"error": "No JSON objects to validate"}
        
        validation_results = {
            "total_objects": len(json_objects),
            "valid_objects": 0,
            "invalid_objects": 0,
            "errors": [],
            "warnings": []
        }
        
        for i, json_obj in enumerate(json_objects):
            try:
                # Check required structure
                if not isinstance(json_obj, dict):
                    validation_results["errors"].append(f"Object {i}: Not a dictionary")
                    validation_results["invalid_objects"] += 1
                    continue
                data = json_obj
                
                # Check required fields
                required_fields = ["test_title", "steps", "expected_outcome"]
                for field in required_fields:
                    if field not in data:
                        validation_results["warnings"].append(f"Object {i}: Missing '{field}' field")
                
                # Check steps is a list
                if "steps" in data and not isinstance(data["steps"], list):
                    validation_results["errors"].append(f"Object {i}: 'steps' is not a list")
                    validation_results["invalid_objects"] += 1
                    continue
                
                # Check if steps are empty
                if "steps" in data and not data["steps"]:
                    validation_results["warnings"].append(f"Object {i}: Empty steps list")
                
                validation_results["valid_objects"] += 1
                
            except Exception as e:
                validation_results["errors"].append(f"Object {i}: Validation error - {str(e)}")
                validation_results["invalid_objects"] += 1
        
        return validation_results
    
    def save_json_output(self, json_objects: List[Dict[str, Any]], 
                        output_path: str) -> bool:
        """
        Save JSON output to file
        
        Args:
            json_objects: List of JSON objects to save
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save JSON objects
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_objects, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON output saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON output: {str(e)}")
            return False
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the transformation process
        
        Returns:
            Dictionary containing transformation statistics
        """
        if self.csv_data is None:
            return {"error": "No data available"}
        
        total_rows = len(self.csv_data)
        test_steps_column = self.headers.get("test_steps")
        
        stats = {
            "total_rows": total_rows,
            "test_steps_column": test_steps_column,
            "pattern_type": self.pattern_info.get("pattern_type", "unknown"),
            "pattern_confidence": self.pattern_info.get("confidence", 0.0),
            "output_format": self.output_format
        }
        
        if test_steps_column and test_steps_column in self.csv_data.columns:
            # Count non-empty test steps
            non_empty_steps = self.csv_data[test_steps_column].notna() & (self.csv_data[test_steps_column] != '')
            stats["rows_with_steps"] = int(non_empty_steps.sum())
            stats["rows_without_steps"] = int((~non_empty_steps).sum())
        
        return stats
