"""
External API Service
Handles communication with external backend APIs
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
from src.utils.env_manager import EnvManager

logger = logging.getLogger(__name__)


class ExternalAPIService:
    """Service for communicating with external backend APIs"""
    
    def __init__(self, env_manager: EnvManager):
        """
        Initialize external API service
        
        Args:
            env_manager: Environment manager instance
        """
        self.env_manager = env_manager
        self.base_url = self.env_manager.get("BASE_BACKEND_API_URL")
        
        if not self.base_url:
            logger.warning("BASE_BACKEND_API_URL not configured in environment variables")
    
    def upload_test_cases(
        self, 
        user_id: int, 
        project_id: int, 
        environment_id: int, 
        category_id: int,
        total_objects: int,
        data: List[Dict[str, Any]],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Upload test cases to external backend API
        
        Args:
            user_id: User ID
            project_id: Project ID
            environment_id: Environment ID
            category_id: Category ID
            total_objects: Total number of test case objects
            data: List of test case data
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing upload results
        """
        if not self.base_url:
            return {
                "success": False,
                "error": "BASE_BACKEND_API_URL not configured"
            }
        
        try:
            # Prepare the payload
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "environment_id": environment_id,
                "category_id": category_id,
                "total_objects": total_objects,
                "data": data
            }
            
            # Make the API request
            url = f"{self.base_url}/api/v1/test-cases/ai-steps"
            
            logger.info(f"Uploading {total_objects} test cases to {url}")
            logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
            logger.info(f"Request headers: {json.dumps({'Content-Type': 'application/json', 'Accept': 'application/json'}, indent=2)}")
            
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=timeout
            )
            
            # Log detailed response information
            logger.info(f"External API Response Status: {response.status_code}")
            logger.info(f"External API Response Headers: {dict(response.headers)}")
            logger.info(f"External API Response Content-Type: {response.headers.get('content-type', 'Unknown')}")
            logger.info(f"External API Response Content Length: {len(response.content)}")
            logger.info(f"External API Response Raw Content: {response.text}")
            
            # Check response status
            if response.status_code == 200:
                try:
                    # Try to parse JSON response
                    if response.content:
                        json_response = response.json()
                        logger.info(f"Successfully uploaded {total_objects} test cases")
                        logger.info(f"External API JSON Response: {json_response}")
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "response": json_response,
                            "message": f"Successfully uploaded {total_objects} test cases"
                        }
                    else:
                        logger.warning("External API returned empty response")
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "response": {},
                            "message": f"Successfully uploaded {total_objects} test cases (empty response)"
                        }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Raw response content: {response.text}")
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "error": f"Invalid JSON response from external API: {str(e)}",
                        "response": response.text
                    }
            else:
                logger.error(f"Failed to upload test cases. Status: {response.status_code}")
                logger.error(f"Error response content: {response.text}")
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"Upload failed with status {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout while uploading test cases")
            return {
                "success": False,
                "error": "Request timeout while uploading test cases"
            }
        except requests.exceptions.ConnectionError:
            logger.error("Connection error while uploading test cases")
            return {
                "success": False,
                "error": "Connection error while uploading test cases"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while uploading test cases: {str(e)}")
            return {
                "success": False,
                "error": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error while uploading test cases: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to external API
        
        Returns:
            Dictionary containing connection test results
        """
        if not self.base_url:
            return {
                "success": False,
                "error": "BASE_BACKEND_API_URL not configured"
            }
        
        try:
            # Try to make a simple request to test connectivity
            response = requests.get(
                self.base_url,
                timeout=10
            )
            
            return {
                "success": True,
                "status_code": response.status_code,
                "message": "Connection test successful"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection test failed: {str(e)}"
            }
    
    def get_upload_status(self, upload_id: str) -> Dict[str, Any]:
        """
        Get status of a previous upload (if supported by external API)
        
        Args:
            upload_id: Upload ID to check status for
            
        Returns:
            Dictionary containing upload status
        """
        if not self.base_url:
            return {
                "success": False,
                "error": "BASE_BACKEND_API_URL not configured"
            }
        
        try:
            # This would depend on your external API's status endpoint
            # For now, return a placeholder
            return {
                "success": True,
                "message": "Upload status check not implemented",
                "upload_id": upload_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Status check failed: {str(e)}"
            }
