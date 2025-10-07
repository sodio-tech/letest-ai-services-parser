"""
LLM Analyzer Module
Uses GPT-4o-mini for intelligent pattern recognition and text analysis
"""

import re
import logging
from typing import Dict, Any, List, Optional
from ..ai.openai_client import OpenAIClient
from ..ai.prompt_engineer import PromptEngineer

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Uses LLM to analyze delimiter patterns and extract test steps"""
    
    def __init__(self, llm_client: OpenAIClient, sample_data: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM analyzer
        
        Args:
            llm_client: OpenAI client for LLM interactions
            sample_data: Sample text data for pattern analysis
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.sample_data = sample_data
        self.config = config or {}
        self.prompt_engineer = PromptEngineer()
        self.patterns = {}
        self._extract_patterns_with_llm()
    
    def _extract_patterns_with_llm(self) -> None:
        """
        Use GPT-4o-mini to analyze and extract delimiter patterns
        """
        if not self.sample_data:
            logger.warning("No sample data available for pattern analysis")
            return
        
        try:
            # Use LLM to analyze patterns
            self.patterns = self.llm_client.analyze_text_patterns(self.sample_data)
            
            if "error" in self.patterns:
                logger.error(f"Error in pattern analysis: {self.patterns['error']}")
                self._fallback_pattern_detection()
                return
            
            logger.info(f"Pattern analysis completed: {self.patterns}")
            
        except Exception as e:
            logger.error(f"Error extracting patterns with LLM: {str(e)}")
            self._fallback_pattern_detection()
    
    def _fallback_pattern_detection(self) -> None:
        """
        Fallback pattern detection using simple regex patterns
        """
        logger.info("Using fallback pattern detection")
        
        # Common pattern regexes
        patterns = {
            'numeric_dot': r'^\d+\.\s*',
            'numeric_paren': r'^\d+\)\s*',
            'alphabetic_lower_dot': r'^[a-z]\.\s*',
            'alphabetic_upper_dot': r'^[A-Z]\.\s*',
            'roman_lower_dot': r'^[ivxlcdm]+\.\s*',
            'roman_upper_dot': r'^[IVXLCDM]+\.\s*',
            'bullet': r'^[•◦-*]\s*',
            'step_prefix': r'^step\s+\d+\s*[:\-\.]\s*',
            'action_prefix': r'^action\s+\d+\s*[:\-\.]\s*'
        }
        
        # Test patterns against sample data
        pattern_scores = {}
        for pattern_name, pattern_regex in patterns.items():
            matches = 0
            for text in self.sample_data:
                if re.search(pattern_regex, text, re.IGNORECASE):
                    matches += 1
            
            if matches > 0:
                pattern_scores[pattern_name] = matches / len(self.sample_data)
        
        # Select best pattern
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            best_score = pattern_scores[best_pattern]
            
            self.patterns = {
                "pattern_type": best_pattern,
                "delimiter_pattern": patterns[best_pattern],
                "regex_pattern": patterns[best_pattern],
                "confidence": best_score,
                "examples_found": [text for text in self.sample_data 
                                 if re.search(patterns[best_pattern], text, re.IGNORECASE)],
                "pattern_description": f"Detected {best_pattern} pattern",
                "notes": "Fallback detection used"
            }
        else:
            # Default to simple line splitting
            self.patterns = {
                "pattern_type": "line_split",
                "delimiter_pattern": "\\n",
                "regex_pattern": "\\n",
                "confidence": 0.5,
                "examples_found": [],
                "pattern_description": "No clear pattern detected, using line split",
                "notes": "Fallback to line splitting"
            }
        
        logger.info(f"Fallback pattern detection completed: {self.patterns}")
    
    def analyze_delimiter_patterns(self, text_samples: List[str]) -> Dict[str, Any]:
        """
        Use LLM to identify delimiter patterns in test step text
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        if not text_samples:
            return {"error": "No text samples provided"}
        
        try:
            # Use LLM to analyze patterns
            patterns = self.llm_client.analyze_text_patterns(text_samples)
            
            if "error" in patterns:
                logger.error(f"Error in pattern analysis: {patterns['error']}")
                return patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing delimiter patterns: {str(e)}")
            return {"error": f"Error analyzing patterns: {str(e)}"}
    
    def generate_regex_pattern(self, pattern_description: str) -> str:
        """
        Generate regex pattern based on LLM analysis
        
        Args:
            pattern_description: Description of the pattern
            
        Returns:
            Regex pattern string
        """
        if not self.patterns:
            return r'.*'  # Default pattern
        
        return self.patterns.get("regex_pattern", r'.*')
    
    def validate_pattern_with_llm(self, pattern: str, test_data: List[str]) -> Dict[str, Any]:
        """
        Use LLM to validate pattern accuracy and suggest improvements
        
        Args:
            pattern: Regex pattern to validate
            test_data: Test data to validate against
            
        Returns:
            Dictionary containing validation results
        """
        if not test_data:
            return {"error": "No test data provided"}
        
        try:
            # Test pattern against data
            matches = 0
            total = len(test_data)
            
            for text in test_data:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
            
            accuracy = matches / total if total > 0 else 0
            
            # Use LLM to validate if accuracy is low
            if accuracy < 0.8:
                validation_result = self.llm_client.validate_extraction(
                    str(test_data), 
                    [text for text in test_data if re.search(pattern, text, re.IGNORECASE)]
                )
                
                if "error" not in validation_result:
                    return {
                        "accuracy": accuracy,
                        "matches": matches,
                        "total": total,
                        "llm_validation": validation_result,
                        "pattern": pattern
                    }
            
            return {
                "accuracy": accuracy,
                "matches": matches,
                "total": total,
                "pattern": pattern,
                "is_valid": accuracy >= 0.8
            }
            
        except Exception as e:
            logger.error(f"Error validating pattern: {str(e)}")
            return {"error": f"Error validating pattern: {str(e)}"}
    
    def extract_steps_from_text(self, text: str) -> List[str]:
        """
        Extract individual steps from text using identified pattern
        
        Args:
            text: Text containing multiple steps
            
        Returns:
            List of extracted steps
        """
        if not text or not self.patterns:
            return [text] if text else []
        
        try:
            # Use LLM for intelligent extraction
            extracted_steps = self.llm_client.extract_steps_with_llm(text, self.patterns)
            
            if extracted_steps:
                return extracted_steps
            
            # Fallback to regex-based extraction
            return self._extract_steps_with_regex(text)
            
        except Exception as e:
            logger.error(f"Error extracting steps: {str(e)}")
            return self._extract_steps_with_regex(text)
    
    def _extract_steps_with_regex(self, text: str) -> List[str]:
        """
        Extract steps using regex pattern as fallback
        
        Args:
            text: Text to extract steps from
            
        Returns:
            List of extracted steps
        """
        if not self.patterns or not text:
            return [text] if text else []
        
        regex_pattern = self.patterns.get("regex_pattern", r'\n')
        
        try:
            # Split text using the regex pattern
            steps = re.split(regex_pattern, text)
            
            # Clean up steps
            cleaned_steps = []
            for step in steps:
                step = step.strip()
                if step:  # Only add non-empty steps
                    cleaned_steps.append(step)
            
            return cleaned_steps if cleaned_steps else [text]
            
        except Exception as e:
            logger.error(f"Error in regex extraction: {str(e)}")
            return [text]
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        Get information about the identified pattern
        
        Returns:
            Dictionary containing pattern information
        """
        return self.patterns.copy() if self.patterns else {}
    
    def get_confidence(self) -> float:
        """
        Get confidence score for pattern identification
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        return self.patterns.get("confidence", 0.0) if self.patterns else 0.0
    
    def is_ready(self) -> bool:
        """
        Check if pattern analysis is complete
        
        Returns:
            True if ready, False otherwise
        """
        return bool(self.patterns and "regex_pattern" in self.patterns)
    
    def update_patterns(self, new_patterns: Dict[str, Any]) -> None:
        """
        Update patterns with new information
        
        Args:
            new_patterns: New pattern information
        """
        self.patterns.update(new_patterns)
        logger.info(f"Patterns updated: {self.patterns}")
    
    def get_examples_found(self) -> List[str]:
        """
        Get examples of the pattern found in sample data
        
        Returns:
            List of example texts
        """
        return self.patterns.get("examples_found", []) if self.patterns else []
