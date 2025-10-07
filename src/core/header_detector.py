"""
Header Detector Module
Uses LLM to intelligently identify column purposes in CSV files
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from ..ai.openai_client import OpenAIClient
from ..ai.prompt_engineer import PromptEngineer

logger = logging.getLogger(__name__)


class HeaderDetector:
    """Uses LLM to identify test title, steps, and expected outcome columns"""
    
    def __init__(self, df: pd.DataFrame, llm_client: OpenAIClient, config: Optional[Dict[str, Any]] = None, columns_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize header detector
        
        Args:
            df: DataFrame containing CSV data
            llm_client: OpenAI client for LLM interactions
            config: Configuration dictionary
            columns_mapping: Optional pre-defined column mapping
        """
        self.df = df
        self.llm_client = llm_client
        self.config = config or {}
        self.columns_mapping = columns_mapping
        self.prompt_engineer = PromptEngineer()
        self.headers = {}
        self.column_mapping = {}
        self._analyze_headers()
    
    def _analyze_headers(self) -> None:
        """
        Analyze headers using LLM to identify column purposes
        """
        if self.df is None or self.df.empty:
            logger.error("No data available for header analysis")
            return
        
        try:
            # Use provided column mapping if available
            if self.columns_mapping:
                self.column_mapping = self.columns_mapping
                logger.info(f"Using provided column mapping: {self.column_mapping}")
                return
            
            # Prepare data for LLM analysis
            headers = list(self.df.columns)
            sample_rows = self.df.head(3).values.tolist()
            
            # Use LLM to identify columns
            self.column_mapping = self.llm_client.identify_columns(headers, sample_rows)
            
            if "error" in self.column_mapping:
                logger.error(f"Error in column identification: {self.column_mapping['error']}")
                self._fallback_column_detection()
                return
            
            # Process the LLM response
            self._process_column_mapping()
            
            logger.info(f"Column mapping identified: {self.column_mapping}")
            
        except Exception as e:
            logger.error(f"Error analyzing headers: {str(e)}")
            self._fallback_column_detection()
    
    def _process_column_mapping(self) -> None:
        """
        Process the LLM column mapping response
        """
        try:
            # Map columns by name or index
            self.headers = {
                "test_title": self._resolve_column(self.column_mapping.get("test_title_column")),
                "test_steps": self._resolve_column(self.column_mapping.get("test_steps_column")),
                "expected_outcome": self._resolve_column(self.column_mapping.get("expected_outcome_column"))
            }
            
            # Validate the mapping
            if not self._validate_column_mapping():
                logger.warning("Column mapping validation failed, using fallback detection")
                self._fallback_column_detection()
                
        except Exception as e:
            logger.error(f"Error processing column mapping: {str(e)}")
            self._fallback_column_detection()
    
    def _resolve_column(self, column_identifier: Any) -> Optional[str]:
        """
        Resolve column identifier to actual column name
        
        Args:
            column_identifier: Column name, index, or None
            
        Returns:
            Resolved column name or None
        """
        if column_identifier is None:
            return None
        
        # If it's already a column name
        if column_identifier in self.df.columns:
            return column_identifier
        
        # If it's an index
        try:
            if isinstance(column_identifier, (int, str)) and str(column_identifier).isdigit():
                index = int(column_identifier)
                if 0 <= index < len(self.df.columns):
                    return self.df.columns[index]
        except (ValueError, IndexError):
            pass
        
        # Try fuzzy matching on column names
        column_lower = str(column_identifier).lower()
        for col in self.df.columns:
            if column_lower in col.lower() or col.lower() in column_lower:
                return col
        
        return None
    
    def _validate_column_mapping(self) -> bool:
        """
        Validate the identified column mapping
        
        Returns:
            True if mapping is valid, False otherwise
        """
        if not self.headers:
            return False
        
        # Check if at least test_steps column is identified
        if not self.headers.get("test_steps"):
            logger.warning("No test steps column identified")
            return False
        
        # Check if identified columns exist in the dataframe
        for col_type, col_name in self.headers.items():
            if col_name and col_name not in self.df.columns:
                logger.warning(f"Identified column '{col_name}' for '{col_type}' does not exist")
                return False
        
        return True
    
    def _fallback_column_detection(self) -> None:
        """
        Fallback column detection using simple heuristics
        """
        logger.info("Using fallback column detection")
        
        self.headers = {
            "test_title": None,
            "test_steps": None,
            "expected_outcome": None
        }
        
        # Simple keyword-based detection
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Test title detection
            if any(keyword in col_lower for keyword in ['title', 'name', 'test', 'case']):
                if not self.headers["test_title"]:
                    self.headers["test_title"] = col
            
            # Test steps detection
            elif any(keyword in col_lower for keyword in ['step', 'instruction', 'procedure', 'action']):
                if not self.headers["test_steps"]:
                    self.headers["test_steps"] = col
            
            # Expected outcome detection
            elif any(keyword in col_lower for keyword in ['expected', 'outcome', 'result', 'verify']):
                if not self.headers["expected_outcome"]:
                    self.headers["expected_outcome"] = col
        
        # If no test_steps found, try to find the column with most text content
        if not self.headers["test_steps"]:
            text_lengths = {}
            for col in self.df.columns:
                # Calculate average text length in this column
                text_lengths[col] = self.df[col].astype(str).str.len().mean()
            
            # Select the column with the longest average text
            if text_lengths:
                self.headers["test_steps"] = max(text_lengths, key=text_lengths.get)
        
        logger.info(f"Fallback column mapping: {self.headers}")
    
    def find_columns_with_llm(self) -> Dict[str, Any]:
        """
        Use GPT-4o-mini to intelligently identify column purposes
        
        Returns:
            Dictionary containing column identification results
        """
        return self.column_mapping
    
    def validate_column_mapping(self, mapping: Optional[Dict[str, str]] = None) -> bool:
        """
        Validate the column mapping
        
        Args:
            mapping: Optional custom mapping to validate
            
        Returns:
            True if mapping is valid, False otherwise
        """
        mapping_to_validate = mapping or self.headers
        
        if not mapping_to_validate:
            return False
        
        # Check if all required columns exist
        for col_type, col_name in mapping_to_validate.items():
            if col_name and col_name not in self.df.columns:
                logger.warning(f"Column '{col_name}' for '{col_type}' does not exist in dataframe")
                return False
        
        return True
    
    def get_test_title_column(self) -> Optional[str]:
        """
        Get the identified test title column
        
        Returns:
            Column name or None
        """
        return self.headers.get("test_title")
    
    def get_test_steps_column(self) -> Optional[str]:
        """
        Get the identified test steps column
        
        Returns:
            Column name or None
        """
        return self.headers.get("test_steps")
    
    def get_expected_outcome_column(self) -> Optional[str]:
        """
        Get the identified expected outcome column
        
        Returns:
            Column name or None
        """
        return self.headers.get("expected_outcome")
    
    def get_all_headers(self) -> Dict[str, Optional[str]]:
        """
        Get all identified headers
        
        Returns:
            Dictionary of all headers
        """
        return self.headers.copy()
    
    def get_column_confidence(self) -> float:
        """
        Get confidence score for column identification
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.column_mapping:
            return 0.0
        
        return self.column_mapping.get("confidence", 0.0)
    
    def get_reasoning(self) -> str:
        """
        Get reasoning for column identification
        
        Returns:
            Reasoning text
        """
        if not self.column_mapping:
            return "No LLM analysis performed"
        
        return self.column_mapping.get("reasoning", "No reasoning provided")
    
    def get_alternative_mappings(self) -> Dict[str, List[str]]:
        """
        Get alternative column mappings suggested by LLM
        
        Returns:
            Dictionary of alternative mappings
        """
        if not self.column_mapping:
            return {}
        
        return self.column_mapping.get("alternative_mappings", {})
    
    def is_ready(self) -> bool:
        """
        Check if header detection is complete and valid
        
        Returns:
            True if ready, False otherwise
        """
        return (
            self.headers is not None and
            self.headers.get("test_steps") is not None and
            self.validate_column_mapping()
        )
