"""
OpenAI Client Module for gpt-4o-mini Integration
Handles all LLM interactions for CSV parsing tasks
"""

import json
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI gpt-4o-mini API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 temperature: float = 0.1, max_tokens: int = 2000, 
                 timeout: int = 30, retry_attempts: int = 3):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
    def _make_request(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Make API request with retry logic
        
        Args:
            messages: List of message dictionaries for the API
            
        Returns:
            API response or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                return response
                
            except Exception as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All {self.retry_attempts} API request attempts failed")
                    return None
                    
        return None
    
    def analyze_text_patterns(self, text_samples: List[str]) -> Dict[str, Any]:
        """
        Use gpt-4o-mini to analyze delimiter patterns in text samples
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        prompt = f"""
        Analyze the following text samples to identify delimiter patterns used to separate test steps.
        
        Text samples:
        {json.dumps(text_samples, indent=2)}
        
        Please identify:
        1. The delimiter pattern used (e.g., "1. ", "a. ", "i. ", "Step 1:", etc.)
        2. The pattern type (numeric, alphabetic, roman, custom, etc.)
        3. A regex pattern that can be used to split the text
        4. Confidence level (0.0 to 1.0)
        
        Return your analysis as a JSON object with the following structure:
        {{
            "pattern_type": "string",
            "delimiter_pattern": "string", 
            "regex_pattern": "string",
            "confidence": 0.0,
            "examples_found": ["list of examples"],
            "notes": "any additional observations"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing text patterns and identifying delimiters. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages)
        if not response:
            return {"error": "Failed to get response from OpenAI API"}
            
        try:
            content = response.choices[0].message.content
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from response"}
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}
    
    def identify_columns(self, headers: List[str], sample_rows: List[List[str]]) -> Dict[str, Any]:
        """
        Use LLM to identify column purposes from headers and sample data
        
        Args:
            headers: List of column headers
            sample_rows: Sample data rows
            
        Returns:
            Dictionary containing column identification results
        """
        prompt = f"""
        Analyze the following CSV data to identify which columns contain:
        1. Test titles/names
        2. Test steps (instructions)
        3. Expected outcomes/results
        
        Headers: {headers}
        
        Sample data (first 3 rows):
        {json.dumps(sample_rows[:3], indent=2)}
        
        Return your analysis as a JSON object:
        {{
            "test_title_column": "column_name_or_index",
            "test_steps_column": "column_name_or_index", 
            "expected_outcome_column": "column_name_or_index",
            "confidence": 0.0,
            "reasoning": "explanation of your choices"
        }}
        
        If a column type cannot be identified, use null for that field.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing CSV data structure. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages)
        if not response:
            return {"error": "Failed to get response from OpenAI API"}
            
        try:
            content = response.choices[0].message.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from response"}
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}
    
    def extract_steps_with_llm(self, text: str, pattern_info: Dict[str, Any]) -> List[str]:
        """
        Use LLM to intelligently extract individual steps from text
        
        Args:
            text: Text containing multiple steps
            pattern_info: Information about the delimiter pattern
            
        Returns:
            List of extracted steps
        """
        prompt = f"""
        Extract individual test steps from the following text using the identified pattern.
        
        Text to extract from:
        {text}
        
        Pattern information:
        {json.dumps(pattern_info, indent=2)}
        
        Return the extracted steps as a JSON array of strings:
        ["step 1", "step 2", "step 3", ...]
        
        Make sure each step is clean and properly formatted.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at extracting structured data from text. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages)
        if not response:
            return []
            
        try:
            content = response.choices[0].message.content
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return []
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return []
    
    def validate_extraction(self, original_text: str, extracted_steps: List[str]) -> Dict[str, Any]:
        """
        Use LLM to validate step extraction accuracy
        
        Args:
            original_text: Original text that was processed
            extracted_steps: Steps that were extracted
            
        Returns:
            Dictionary containing validation results
        """
        prompt = f"""
        Validate the accuracy of step extraction by comparing the original text with the extracted steps.
        
        Original text:
        {original_text}
        
        Extracted steps:
        {json.dumps(extracted_steps, indent=2)}
        
        Return validation results as JSON:
        {{
            "is_accurate": true/false,
            "accuracy_score": 0.0,
            "missing_steps": ["any steps that were missed"],
            "incorrect_extractions": ["any incorrect extractions"],
            "suggestions": "any suggestions for improvement"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at validating data extraction accuracy. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages)
        if not response:
            return {"error": "Failed to get response from OpenAI API"}
            
        try:
            content = response.choices[0].message.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from response"}
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}
