"""
CSV Reader Module
Handles CSV file reading with encoding detection and data validation
"""

import pandas as pd
import chardet
import os
import logging
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class CSVReader:
    """Handles CSV file reading with automatic encoding detection"""
    
    def __init__(self, file_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV reader
        
        Args:
            file_path: Path to the CSV file
            config: Configuration dictionary
        """
        self.file_path = file_path
        self.config = config or {}
        self.encoding = None
        self.df = None
        self.file_size = 0
        self._load_csv()
    
    def _detect_encoding(self) -> str:
        """
        Detect file encoding using chardet
        
        Returns:
            Detected encoding string
        """
        try:
            with open(self.file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Fallback to common encodings if confidence is low
                if confidence < 0.7:
                    logger.warning(f"Low confidence in encoding detection. Trying common encodings.")
                    for fallback_encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            with open(self.file_path, 'r', encoding=fallback_encoding) as test_file:
                                test_file.read(1000)
                            logger.info(f"Successfully tested fallback encoding: {fallback_encoding}")
                            return fallback_encoding
                        except UnicodeDecodeError:
                            continue
                
                return encoding or 'utf-8'
                
        except Exception as e:
            logger.error(f"Error detecting encoding: {str(e)}")
            return 'utf-8'  # Default fallback
    
    def _validate_file(self) -> bool:
        """
        Validate file before processing
        
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(self.file_path):
            logger.error(f"File does not exist: {self.file_path}")
            return False
        
        self.file_size = os.path.getsize(self.file_path)
        max_size = self.config.get('max_file_size', 10485760)  # 10MB default
        
        if self.file_size > max_size:
            logger.error(f"File size ({self.file_size} bytes) exceeds maximum allowed size ({max_size} bytes)")
            return False
        
        if self.file_size == 0:
            logger.error("File is empty")
            return False
        
        return True
    
    def _load_csv(self) -> None:
        """
        Load CSV file with proper encoding and delimiter detection
        """
        if not self._validate_file():
            return
        
        self.encoding = self._detect_encoding()
        
        try:
            # Try to read with detected encoding
            self.df = pd.read_csv(
                self.file_path,
                encoding=self.encoding,
                sep=None,  # Auto-detect delimiter
                engine='python',
                on_bad_lines='skip',  # Skip malformed lines
                dtype=str  # Read all columns as strings initially
            )
            
            logger.info(f"Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Clean the data
            self._clean_data()
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            # Try with different encodings
            for fallback_encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.info(f"Trying fallback encoding: {fallback_encoding}")
                    self.df = pd.read_csv(
                        self.file_path,
                        encoding=fallback_encoding,
                        sep=None,
                        engine='python',
                        on_bad_lines='skip',
                        dtype=str
                    )
                    self.encoding = fallback_encoding
                    logger.info(f"Successfully loaded with fallback encoding: {fallback_encoding}")
                    self._clean_data()
                    break
                except Exception as fallback_error:
                    logger.warning(f"Fallback encoding {fallback_encoding} failed: {str(fallback_error)}")
                    continue
            else:
                logger.error("All encoding attempts failed")
                self.df = None
    
    def _clean_data(self) -> None:
        """
        Clean the loaded CSV data
        """
        if self.df is None:
            return
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Remove rows where all columns are empty strings
        self.df = self.df[~(self.df == '').all(axis=1)]
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Replace NaN values with empty strings
        self.df = self.df.fillna('')
        
        logger.info(f"Data cleaned. Final shape: {self.df.shape}")
    
    def get_sample_data(self, percentage: float = 0.2) -> pd.DataFrame:
        """
        Extract random sample of data for analysis
        
        Args:
            percentage: Percentage of data to sample (0.0 to 1.0)
            
        Returns:
            Sampled DataFrame
        """
        if self.df is None or self.df.empty:
            logger.warning("No data available for sampling")
            return pd.DataFrame()
        
        sample_size = max(1, int(len(self.df) * percentage))
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        
        logger.info(f"Sampled {len(sample_df)} rows ({percentage*100:.1f}% of {len(self.df)} total rows)")
        return sample_df
    
    def get_column_info(self) -> Dict[str, Any]:
        """
        Get information about the CSV columns
        
        Returns:
            Dictionary containing column information
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        column_info = {
            "total_columns": len(self.df.columns),
            "total_rows": len(self.df),
            "columns": list(self.df.columns),
            "column_types": self.df.dtypes.to_dict(),
            "non_empty_counts": self.df.count().to_dict(),
            "sample_data": self.df.head(3).to_dict('records')
        }
        
        return column_info
    
    def get_text_samples(self, column_name: str, sample_size: int = 10) -> List[str]:
        """
        Get text samples from a specific column for pattern analysis
        
        Args:
            column_name: Name of the column to sample
            sample_size: Number of samples to return
            
        Returns:
            List of text samples
        """
        if self.df is None or column_name not in self.df.columns:
            logger.warning(f"Column '{column_name}' not found")
            return []
        
        # Get non-empty values from the column
        non_empty_values = self.df[column_name].dropna()
        non_empty_values = non_empty_values[non_empty_values != '']
        
        if non_empty_values.empty:
            logger.warning(f"No non-empty values found in column '{column_name}'")
            return []
        
        # Sample the values
        sample_values = non_empty_values.sample(n=min(sample_size, len(non_empty_values)), random_state=42)
        
        return sample_values.tolist()
    
    def is_loaded(self) -> bool:
        """
        Check if CSV was successfully loaded
        
        Returns:
            True if CSV is loaded, False otherwise
        """
        return self.df is not None and not self.df.empty
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded DataFrame
        
        Returns:
            DataFrame or None if not loaded
        """
        return self.df
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the file
        
        Returns:
            Dictionary containing file information
        """
        return {
            "file_path": self.file_path,
            "file_size": self.file_size,
            "encoding": self.encoding,
            "is_loaded": self.is_loaded(),
            "shape": self.df.shape if self.df is not None else (0, 0)
        }
