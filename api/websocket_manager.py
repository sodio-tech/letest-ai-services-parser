"""
WebSocket implementation for real-time CSV processing updates
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and sessions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and register session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.processing_sessions[session_id] = {
            "status": "connected",
            "created_at": datetime.utcnow().isoformat(),
            "logs": []
        }
        
        # Send session start message
        await self.send_message(session_id, {
            "type": "session_start",
            "session_id": session_id,
            "message": "WebSocket session established. Ready to process CSV.",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"WebSocket session {session_id} connected")
    
    async def disconnect(self, session_id: str):
        """Remove WebSocket connection and cleanup session"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        if session_id in self.processing_sessions:
            # Send session end message before cleanup
            try:
                await self.send_message(session_id, {
                    "type": "session_end",
                    "session_id": session_id,
                    "message": "WebSocket session ended.",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except:
                pass  # Connection might already be closed
            
            del self.processing_sessions[session_id]
        
        logger.info(f"WebSocket session {session_id} disconnected")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket session"""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                await websocket.send_text(json.dumps(message))
                
                # Store log message
                if session_id in self.processing_sessions:
                    self.processing_sessions[session_id]["logs"].append(message)
                    
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {e}")
                await self.disconnect(session_id)
    
    async def send_log(self, session_id: str, log_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Send log message to WebSocket session"""
        log_message = {
            "type": "log",
            "log_type": log_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if data:
            log_message["data"] = data
            
        await self.send_message(session_id, log_message)
    
    async def send_progress(self, session_id: str, step: str, progress: int, total: int, message: str):
        """Send progress update to WebSocket session"""
        await self.send_message(session_id, {
            "type": "progress",
            "step": step,
            "progress": progress,
            "total": total,
            "percentage": int((progress / total) * 100) if total > 0 else 0,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_final_result(self, session_id: str, result: Dict[str, Any]):
        """Send final processing result to WebSocket session"""
        await self.send_message(session_id, {
            "type": "final_result",
            "session_id": session_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_error(self, session_id: str, error: str, details: Optional[Dict[str, Any]] = None):
        """Send error message to WebSocket session"""
        error_message = {
            "type": "error",
            "session_id": session_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            error_message["details"] = details
            
        await self.send_message(session_id, error_message)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
