"""
Task Manager for AI Mashup Background Processing

Handles async mashup creation with progress tracking and result storage.
Uses in-memory storage for simplicity (can be upgraded to Redis/database later).

Author: MAI Team
Date: 2025-10-11
"""

import os
import time
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import soundfile as sf

# Import ONLY from new LLM system and utils
from app.llm.producer import AIMashupProducer
from app.llm.models import MashupResult

logger = logging.getLogger(__name__)


# =============================================================================
# TASK STATUS
# =============================================================================

class TaskStatus(str, Enum):
    """Task status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update for WebSocket."""
    message: str
    percentage: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MashupTask:
    """Represents a mashup creation task."""
    task_id: str
    status: TaskStatus
    progress: int
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Audio data
    track_a_audio: Optional[np.ndarray] = None
    track_b_audio: Optional[np.ndarray] = None
    mashup_audio: Optional[np.ndarray] = None
    
    # Metadata
    track_a_name: str = "Track A"
    track_b_name: str = "Track B"
    mashup_style: str = "dj_mix"
    custom_instructions: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    temperature: float = 0.7
    sr: int = 44100
    
    # Results
    result: Optional[MashupResult] = None
    mashup_file: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    # Progress tracking
    progress_history: List[ProgressUpdate] = field(default_factory=list)


# =============================================================================
# TASK MANAGER
# =============================================================================

class TaskManager:
    """
    Manages background mashup creation tasks.
    
    Uses in-memory storage with optional file persistence.
    """
    
    def __init__(self, output_dir: str = "./tmp/ai_mashups"):
        """
        Initialize task manager.
        
        Args:
            output_dir: Directory to store mashup files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory task storage
        self.tasks: Dict[str, MashupTask] = {}
        
        # WebSocket connections per task
        self.websocket_connections: Dict[str, List[asyncio.Queue]] = {}
        
        logger.info(f"Task Manager initialized (output: {self.output_dir})")
    
    def create_task(
        self,
        track_a_audio: np.ndarray,
        track_b_audio: np.ndarray,
        track_a_name: str = "Track A",
        track_b_name: str = "Track B",
        mashup_style: str = "dj_mix",
        custom_instructions: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
        sr: int = 44100
    ) -> str:
        """
        Create a new mashup task.
        
        Args:
            track_a_audio: Audio data for Track A
            track_b_audio: Audio data for Track B
            track_a_name: Name of Track A
            track_b_name: Name of Track B
            mashup_style: Mashup style
            custom_instructions: Custom instructions (overrides style)
            llm_provider: LLM provider
            llm_model: LLM model name
            temperature: LLM temperature
            sr: Sample rate
        
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = MashupTask(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            progress=0,
            message="Task created, waiting to start...",
            created_at=datetime.utcnow(),
            track_a_audio=track_a_audio,
            track_b_audio=track_b_audio,
            track_a_name=track_a_name,
            track_b_name=track_b_name,
            mashup_style=mashup_style,
            custom_instructions=custom_instructions,
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            sr=sr
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id} ({mashup_style}, {llm_provider})")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[MashupTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def update_progress(self, task_id: str, message: str, percentage: int):
        """Update task progress."""
        task = self.tasks.get(task_id)
        if task:
            task.progress = percentage
            task.message = message
            
            # Add to history
            update = ProgressUpdate(message=message, percentage=percentage)
            task.progress_history.append(update)
            
            # Notify WebSocket subscribers with proper status and stage
            asyncio.create_task(self._notify_websockets(task_id, 'progress', {
                'progress': percentage,
                'message': message,
                'status': task.status.value,
                'current_stage': message  # Add current stage for UI
            }))
    
    async def process_task(self, task_id: str):
        """
        Process a mashup task in the background.
        
        Args:
            task_id: Task ID to process
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        try:
            # Update status
            task.status = TaskStatus.PROCESSING
            self.update_progress(task_id, "Starting mashup creation...", 0)
            
            # Create progress callback
            def progress_callback(message: str, percentage: int):
                self.update_progress(task_id, message, percentage)
            
            # Initialize producer
            producer = AIMashupProducer(
                sr=task.sr,
                llm_provider=task.llm_provider,
                llm_model=task.llm_model,
                temperature=task.temperature
            )
            
            # Create mashup
            result = producer.create_mashup(
                track_a_audio=task.track_a_audio,
                track_b_audio=task.track_b_audio,
                track_a_name=task.track_a_name,
                track_b_name=task.track_b_name,
                mashup_style=task.mashup_style,
                custom_instructions=task.custom_instructions,
                progress_callback=progress_callback
            )
            
            # Check result
            if result.status == 'success':
                # Get mashup audio
                mashup_audio = producer.get_mashup_audio()
                
                if mashup_audio is not None:
                    # Save to file
                    mashup_file = self._save_mashup(task_id, mashup_audio, task.sr)
                    
                    # Update task
                    task.mashup_audio = mashup_audio
                    task.mashup_file = mashup_file
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    task.progress = 100
                    task.message = "Mashup completed successfully!"
                    
                    logger.info(f"Task {task_id} completed successfully")
                    
                    # Notify WebSockets
                    await self._notify_websockets(task_id, 'completed', {
                        'mashup_url': f"/api/v2/download/{task_id}",
                        'duration': result.total_duration
                    })
                else:
                    raise ValueError("No mashup audio generated")
            else:
                # Failed
                task.status = TaskStatus.FAILED
                task.errors = result.errors or ["Unknown error"]
                task.completed_at = datetime.utcnow()
                task.message = "Mashup creation failed"
                
                logger.error(f"Task {task_id} failed: {task.errors}")
                
                await self._notify_websockets(task_id, 'failed', {
                    'errors': task.errors
                })
        
        except Exception as e:
            logger.error(f"Task {task_id} error: {e}", exc_info=True)
            
            task.status = TaskStatus.FAILED
            task.errors = [str(e)]
            task.completed_at = datetime.utcnow()
            task.message = f"Error: {str(e)}"
            
            await self._notify_websockets(task_id, 'failed', {
                'errors': task.errors
            })
        
        finally:
            # Clean up audio data from memory (keep only file reference)
            task.track_a_audio = None
            task.track_b_audio = None
            # Keep mashup_audio for potential re-download
    
    def _save_mashup(self, task_id: str, audio: np.ndarray, sr: int) -> str:
        """Save mashup to file."""
        filename = f"mashup_{task_id}.wav"
        filepath = self.output_dir / filename
        
        sf.write(str(filepath), audio, sr)
        logger.info(f"Saved mashup to {filepath}")
        
        return str(filepath)
    
    def get_mashup_file(self, task_id: str) -> Optional[str]:
        """Get mashup file path."""
        task = self.tasks.get(task_id)
        if task and task.mashup_file and os.path.exists(task.mashup_file):
            return task.mashup_file
        return None
    
    async def _notify_websockets(self, task_id: str, msg_type: str, data: dict):
        """Notify all WebSocket connections for a task."""
        if task_id in self.websocket_connections:
            message = {
                'type': msg_type,
                'task_id': task_id,
                **data
            }
            
            for queue in self.websocket_connections[task_id]:
                try:
                    await queue.put(message)
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket message: {e}")
    
    def register_websocket(self, task_id: str) -> asyncio.Queue:
        """Register a WebSocket connection for task updates."""
        if task_id not in self.websocket_connections:
            self.websocket_connections[task_id] = []
        
        queue = asyncio.Queue()
        self.websocket_connections[task_id].append(queue)
        
        logger.info(f"WebSocket registered for task {task_id}")
        return queue
    
    def unregister_websocket(self, task_id: str, queue: asyncio.Queue):
        """Unregister a WebSocket connection."""
        if task_id in self.websocket_connections:
            try:
                self.websocket_connections[task_id].remove(queue)
                
                # Clean up if no more connections
                if not self.websocket_connections[task_id]:
                    del self.websocket_connections[task_id]
                
                logger.info(f"WebSocket unregistered for task {task_id}")
            except ValueError:
                pass
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Clean up old completed/failed tasks.
        
        Args:
            max_age_hours: Maximum age in hours to keep tasks
        """
        now = datetime.utcnow()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                if task.completed_at:
                    age = (now - task.completed_at).total_seconds() / 3600
                    if age > max_age_hours:
                        # Delete file if exists
                        if task.mashup_file and os.path.exists(task.mashup_file):
                            try:
                                os.remove(task.mashup_file)
                                logger.info(f"Deleted file {task.mashup_file}")
                            except Exception as e:
                                logger.warning(f"Failed to delete {task.mashup_file}: {e}")
                        
                        to_remove.append(task_id)
        
        # Remove from memory
        for task_id in to_remove:
            del self.tasks[task_id]
            logger.info(f"Cleaned up task {task_id}")
        
        return len(to_remove)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Singleton instance (will be initialized by router)
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        output_dir = os.getenv("AI_MASHUP_OUTPUT_DIR", "./tmp/ai_mashups")
        _task_manager = TaskManager(output_dir=output_dir)
    return _task_manager
