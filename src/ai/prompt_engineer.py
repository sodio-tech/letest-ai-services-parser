"""
Prompt Engineering Module
Creates specialized prompts for different CSV parsing tasks
"""

from typing import List, Dict, Any
import json


class PromptEngineer:
    """Creates optimized prompts for different LLM tasks"""
    
    def __init__(self):
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize base prompt templates"""
        return {
            "system_base": "You are an expert at analyzing CSV data and extracting structured information. Always respond with valid JSON format.",
            "column_identification": self._get_column_identification_template(),
            "pattern_analysis": self._get_pattern_analysis_template(),
            "step_extraction": self._get_step_extraction_template(),
            "validation": self._get_validation_template()
        }
    
    def _get_column_identification_template(self) -> str:
        """Get template for column identification prompt"""
        return """
        Analyze the following CSV data to identify which columns contain:
        1. Test titles/names (the main test case name)
        2. Test steps (detailed instructions or procedures)
        3. Expected outcomes/results (what should happen)
        
        Headers: {headers}
        
        Sample data (first 3 rows):
        {sample_data}
        
        Instructions:
        - Look for columns that contain test case names or titles
        - Look for columns with step-by-step instructions (may have delimiters like "1. ", "a. ", etc.)
        - Look for columns describing expected results or outcomes
        - Consider column names, content patterns, and data structure
        
        Return your analysis as a JSON object:
        {{
            "test_title_column": "column_name_or_index",
            "test_steps_column": "column_name_or_index", 
            "expected_outcome_column": "column_name_or_index",
            "confidence": 0.0,
            "reasoning": "explanation of your choices",
            "alternative_mappings": {{
                "test_title": ["alternative_column_1", "alternative_column_2"],
                "test_steps": ["alternative_column_1", "alternative_column_2"],
                "expected_outcome": ["alternative_column_1", "alternative_column_2"]
            }}
        }}
        
        If a column type cannot be identified with confidence > 0.5, use null for that field.
        """
    
    def _get_pattern_analysis_template(self) -> str:
        """Get template for pattern analysis prompt"""
        return """
        Analyze the following text samples to identify delimiter patterns used to separate test steps.
        
        Text samples:
        {text_samples}
        
        Your task:
        1. Identify the delimiter pattern used (e.g., "1. ", "a. ", "i. ", "Step 1:", "• ", etc.)
        2. Determine the pattern type (numeric, alphabetic, roman, bullet, custom, etc.)
        3. Create a regex pattern that can be used to split the text
        4. Assess confidence level (0.0 to 1.0)
        5. Find examples of the pattern in the text
        
        Common pattern types:
        - Numeric: "1. ", "2. ", "3. " or "1) ", "2) ", "3) "
        - Alphabetic: "a. ", "b. ", "c. " or "A. ", "B. ", "C. "
        - Roman: "i. ", "ii. ", "iii. " or "I. ", "II. ", "III. "
        - Bullet: "• ", "◦ ", "- ", "* "
        - Custom: "Step 1:", "Action 1:", "Task 1:", etc.
        
        Return your analysis as a JSON object:
        {{
            "pattern_type": "string",
            "delimiter_pattern": "string", 
            "regex_pattern": "string",
            "confidence": 0.0,
            "examples_found": ["list of examples"],
            "pattern_description": "human readable description",
            "notes": "any additional observations or edge cases"
        }}
        """
    
    def _get_step_extraction_template(self) -> str:
        """Get template for step extraction prompt"""
        return """
        Extract individual test steps from the following text using the identified pattern.
        
        Text to extract from:
        {text}
        
        Pattern information:
        {pattern_info}
        
        Instructions:
        1. Use the provided pattern information to identify step boundaries
        2. Extract each step as a separate, clean string
        3. Remove any delimiter markers from the beginning of each step
        4. Ensure each step is properly formatted and readable
        5. Handle any edge cases or variations in the pattern
        
        Return the extracted steps as a JSON array of strings:
        ["step 1 description", "step 2 description", "step 3 description", ...]
        
        Make sure each step is:
        - Complete and meaningful
        - Free of delimiter markers
        - Properly formatted
        - In the correct order
        """
    
    def _get_validation_template(self) -> str:
        """Get template for validation prompt"""
        return """
        Validate the accuracy of step extraction by comparing the original text with the extracted steps.
        
        Original text:
        {original_text}
        
        Extracted steps:
        {extracted_steps}
        
        Your validation should check:
        1. Are all steps from the original text captured?
        2. Are the extracted steps accurate and complete?
        3. Are there any false extractions or duplicates?
        4. Is the order of steps preserved?
        5. Are the steps properly formatted?
        
        Return validation results as JSON:
        {{
            "is_accurate": true/false,
            "accuracy_score": 0.0,
            "missing_steps": ["any steps that were missed"],
            "incorrect_extractions": ["any incorrect extractions"],
            "duplicate_steps": ["any duplicate steps found"],
            "order_correct": true/false,
            "suggestions": "any suggestions for improvement",
            "overall_quality": "excellent/good/fair/poor"
        }}
        """
    
    def create_column_identification_prompt(self, headers: List[str], sample_data: List[List[str]]) -> str:
        """
        Create prompt for identifying column purposes
        
        Args:
            headers: List of column headers
            sample_data: Sample data rows
            
        Returns:
            Formatted prompt string
        """
        return self.prompts["column_identification"].format(
            headers=json.dumps(headers, indent=2),
            sample_data=json.dumps(sample_data[:3], indent=2)
        )
    
    def create_pattern_analysis_prompt(self, text_samples: List[str]) -> str:
        """
        Create prompt for analyzing delimiter patterns
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            Formatted prompt string
        """
        return self.prompts["pattern_analysis"].format(
            text_samples=json.dumps(text_samples, indent=2)
        )
    
    def create_step_extraction_prompt(self, text: str, pattern_info: Dict[str, Any]) -> str:
        """
        Create prompt for extracting individual steps
        
        Args:
            text: Text containing multiple steps
            pattern_info: Information about the delimiter pattern
            
        Returns:
            Formatted prompt string
        """
        return self.prompts["step_extraction"].format(
            text=text,
            pattern_info=json.dumps(pattern_info, indent=2)
        )
    
    def create_validation_prompt(self, original_text: str, extracted_steps: List[str]) -> str:
        """
        Create prompt for validating extraction accuracy
        
        Args:
            original_text: Original text that was processed
            extracted_steps: Steps that were extracted
            
        Returns:
            Formatted prompt string
        """
        return self.prompts["validation"].format(
            original_text=original_text,
            extracted_steps=json.dumps(extracted_steps, indent=2)
        )
    
    def create_custom_prompt(self, task: str, data: Dict[str, Any]) -> str:
        """
        Create a custom prompt for specific tasks
        
        Args:
            task: Description of the task
            data: Data to include in the prompt
            
        Returns:
            Custom prompt string
        """
        base_prompt = f"""
        Task: {task}
        
        Data:
        {json.dumps(data, indent=2)}
        
        Please analyze the data and provide a structured response in JSON format.
        """
        
        return base_prompt
