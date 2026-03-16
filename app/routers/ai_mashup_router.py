"""
AI Mashup Router - V2 API Endpoints

New V2 endpoints that use ONLY the new LLM system.
Zero dependencies on old mix_engine.py or audio_utils.py.

Dependencies:
- app/llm/* (new LLM system)
- app/utils/* (refactored utilities)
- app/services/task_manager.py (background processing)

Author: MAI Team
Date: 2025-10-11
"""

import logging
from typing import Optional
import io

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Form
)
from fastapi.responses import FileResponse, JSONResponse
import librosa
import soundfile as sf

# Import ONLY from new system
from app.schemas.ai_mashup_schemas import (
    CreateAIMashupRequest,
    CreateAIMashupResponse,
    TaskStatusResponse,
    PreviewPlanRequest,
    PreviewPlanResponse,
    HealthResponse,
    ErrorResponse,
    ExecutionLogEntry,
    convert_internal_log_to_api,
    convert_plan_to_dict
)
from app.services.task_manager import get_task_manager, TaskStatus
from app.llm.producer import AIMashupProducer
from app.llm.executor import get_available_operations
from app.llm.validators import PlanValidator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v2", tags=["AI Mashup V2"])


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for AI Mashup V2 system.
    
    Returns system status and available providers.
    """
    try:
        errors = []
        
        # Check LLM providers
        llm_providers = []
        try:
            import openai
            llm_providers.append("openai")
        except ImportError:
            errors.append("OpenAI client not installed")
        
        try:
            import anthropic
            llm_providers.append("anthropic")
        except ImportError:
            pass  # Optional
        
        # Always have local as option
        llm_providers.append("local")
        
        # Check operations
        operations = get_available_operations()
        
        # Check utils modules
        utils_modules = [
            "key_manipulations",
            "bpm_manipulations",
            "volume_manipulations",
            "mixing_manipulations",
            "audio_analysis",
            "effects_manipulations",
            "transition_manipulations"
        ]
        
        status = "healthy" if not errors else ("degraded" if llm_providers else "unhealthy")
        
        return HealthResponse(
            status=status,
            llm_providers_available=llm_providers,
            operations_count=len(operations),
            utils_modules=utils_modules,
            errors=errors if errors else None
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            llm_providers_available=[],
            operations_count=0,
            utils_modules=[],
            errors=[str(e)]
        )


# =============================================================================
# CREATE MASHUP
# =============================================================================

@router.post("/ai-mashup", response_model=CreateAIMashupResponse)
async def create_ai_mashup(
    background_tasks: BackgroundTasks,
    file_a: UploadFile = File(..., description="First audio track"),
    file_b: UploadFile = File(..., description="Second audio track"),
    mashup_style: Optional[str] = Form(default="dj_mix"),
    custom_instructions: Optional[str] = Form(default=None),
    llm_provider: Optional[str] = Form(default="openai"),
    llm_model: Optional[str] = Form(default=None),
    temperature: Optional[float] = Form(default=0.7),
    track_a_name: Optional[str] = Form(default="Track A"),
    track_b_name: Optional[str] = Form(default="Track B"),
):
    """
    Create an AI-powered mashup from two audio files.
    
    This endpoint:
    1. Accepts two audio files
    2. Creates a background task for mashup creation
    3. Returns task ID and WebSocket URL for progress tracking
    
    The mashup is created using LLM-powered decision making.
    """
    try:
        # Load audio files
        logger.info(f"Loading tracks: {file_a.filename}, {file_b.filename}")
        
        # Read file bytes
        track_a_bytes = await file_a.read()
        track_b_bytes = await file_b.read()
        
        # Load audio using librosa (supports multiple formats)
        track_a_audio, sr_a = librosa.load(io.BytesIO(track_a_bytes), sr=None, mono=True)
        track_b_audio, sr_b = librosa.load(io.BytesIO(track_b_bytes), sr=None, mono=True)
        
        # Resample if needed to common sample rate
        target_sr = 44100
        if sr_a != target_sr:
            track_a_audio = librosa.resample(track_a_audio, orig_sr=sr_a, target_sr=target_sr)
        if sr_b != target_sr:
            track_b_audio = librosa.resample(track_b_audio, orig_sr=sr_b, target_sr=target_sr)
        
        logger.info(f"Loaded Track A: {len(track_a_audio)/target_sr:.2f}s, Track B: {len(track_b_audio)/target_sr:.2f}s")
        
        # Create task
        task_manager = get_task_manager()
        task_id = task_manager.create_task(
            track_a_audio=track_a_audio,
            track_b_audio=track_b_audio,
            track_a_name=track_a_name or file_a.filename,
            track_b_name=track_b_name or file_b.filename,
            mashup_style=mashup_style or "dj_mix",
            custom_instructions=custom_instructions,
            llm_provider=llm_provider or "openai",
            llm_model=llm_model,
            temperature=float(temperature or 0.7),
            sr=target_sr
        )
        
        # Start background processing
        background_tasks.add_task(task_manager.process_task, task_id)
        
        logger.info(f"Created task {task_id}, started background processing")
        
        return CreateAIMashupResponse(
            task_id=task_id,
            status="queued",
            message="Mashup task created and queued for processing",
            websocket_url=f"/ws/ai-mashup/{task_id}",
            status_url=f"/api/v2/ai-mashup/{task_id}"
        )
    
    except Exception as e:
        logger.error(f"Failed to create mashup task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GET TASK STATUS
# =============================================================================

@router.get("/ai-mashup/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a mashup task.
    
    Returns current progress, status, and download URL when complete.
    """
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Build response
    response = TaskStatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        message=task.message,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None
    )
    
    # Add completion data
    if task.status == TaskStatus.COMPLETED:
        response.mashup_url = f"/api/v2/download/{task_id}"
        if task.result:
            response.duration_seconds = task.result.total_duration
            response.execution_log = [
                convert_internal_log_to_api(log)
                for log in task.result.execution_log
            ]
    
    # Add error data
    if task.status == TaskStatus.FAILED:
        response.errors = task.errors
        if task.result and task.result.execution_log:
            response.execution_log = [
                convert_internal_log_to_api(log)
                for log in task.result.execution_log
            ]
    
    return response


# =============================================================================
# DOWNLOAD MASHUP
# =============================================================================

@router.get("/download/{task_id}")
async def download_mashup(task_id: str):
    """
    Download the completed mashup file.
    
    Returns the audio file (WAV format).
    """
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed (status: {task.status.value})"
        )
    
    mashup_file = task_manager.get_mashup_file(task_id)
    if not mashup_file:
        raise HTTPException(status_code=404, detail="Mashup file not found")
    
    return FileResponse(
        mashup_file,
        media_type="audio/wav",
        filename=f"ai_mashup_{task_id}.wav"
    )


# =============================================================================
# PREVIEW PLAN (WITHOUT EXECUTION)
# =============================================================================

@router.post("/ai-mashup/preview", response_model=PreviewPlanResponse)
async def preview_mashup_plan(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    mashup_style: Optional[str] = Form(default="dj_mix"),
    custom_instructions: Optional[str] = Form(default=None),
    llm_provider: Optional[str] = Form(default="openai"),
    track_a_name: Optional[str] = Form(default="Track A"),
    track_b_name: Optional[str] = Form(default="Track B"),
):
    """
    Preview the mashup plan without executing it.
    
    Useful for:
    - Seeing what the LLM will do before committing
    - Iterating on custom instructions
    - Understanding the generated plan
    """
    try:
        # Load audio files
        track_a_bytes = await file_a.read()
        track_b_bytes = await file_b.read()
        
        track_a_audio, sr_a = librosa.load(io.BytesIO(track_a_bytes), sr=None, mono=True)
        track_b_audio, sr_b = librosa.load(io.BytesIO(track_b_bytes), sr=None, mono=True)
        
        target_sr = 44100
        if sr_a != target_sr:
            track_a_audio = librosa.resample(track_a_audio, orig_sr=sr_a, target_sr=target_sr)
        if sr_b != target_sr:
            track_b_audio = librosa.resample(track_b_audio, orig_sr=sr_b, target_sr=target_sr)
        
        # Create producer (for analysis and plan generation only)
        producer = AIMashupProducer(
            sr=target_sr,
            llm_provider=llm_provider or "openai",
            temperature=0.7
        )
        
        # Analyze tracks
        from app.utils.audio_analysis import analyze_track
        
        track_a_features = analyze_track(track_a_audio, target_sr)
        track_a_features['name'] = track_a_name or file_a.filename
        
        track_b_features = analyze_track(track_b_audio, target_sr)
        track_b_features['name'] = track_b_name or file_b.filename
        
        # Generate plan
        plan = producer._generate_plan(
            track_a_features,
            track_b_features,
            mashup_style or "dj_mix",
            custom_instructions
        )
        
        # Validate plan
        validator = PlanValidator(
            track_a_duration=track_a_features['duration'],
            track_b_duration=track_b_features['duration'],
            strict=False
        )
        
        is_valid = validator.validate_plan(plan)
        
        # Estimate duration (rough estimate)
        estimated_duration = max(
            track_a_features['duration'],
            track_b_features['duration']
        )
        
        return PreviewPlanResponse(
            status="success" if is_valid else "failed",
            plan=convert_plan_to_dict(plan) if is_valid else None,
            errors=validator.get_errors() if not is_valid else None,
            validation_warnings=validator.get_warnings() if validator.get_warnings() else None,
            estimated_duration=estimated_duration
        )
    
    except Exception as e:
        logger.error(f"Failed to preview plan: {e}", exc_info=True)
        return PreviewPlanResponse(
            status="failed",
            errors=[str(e)]
        )


# =============================================================================
# WEBSOCKET FOR REAL-TIME PROGRESS
# =============================================================================

@router.websocket("/ws/ai-mashup/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    
    Sends progress messages as mashup is being created.
    """
    await websocket.accept()
    
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)
    
    if not task:
        await websocket.send_json({
            "type": "error",
            "message": f"Task {task_id} not found"
        })
        await websocket.close()
        return
    
    # Register WebSocket
    queue = task_manager.register_websocket(task_id)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "status": task.status.value,
            "progress": task.progress,
            "message": task.message
        })
        
        # Listen for updates
        while True:
            try:
                # Get message from queue (with timeout)
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(message)
                
                # Close if task is done
                if message['type'] in ['completed', 'failed']:
                    break
            
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}", exc_info=True)
    
    finally:
        # Unregister WebSocket
        task_manager.unregister_websocket(task_id, queue)
        try:
            await websocket.close()
        except:
            pass


# =============================================================================
# UTILITIES
# =============================================================================

@router.get("/operations")
async def list_operations():
    """
    List all available audio operations.
    
    Useful for understanding what the LLM can do.
    """
    operations = get_available_operations()
    return {"operations": operations, "count": len(operations)}


import asyncio  # Add at top if missing
