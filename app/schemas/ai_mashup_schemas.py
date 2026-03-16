"""
API Schemas for AI Mashup V2 Endpoints

These schemas define the request/response models for the FastAPI endpoints.
They are separate from the internal LLM models to maintain clean API boundaries.

Author: MAI Team
Date: 2025-10-11
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field

# Import internal models for conversion
from app.llm.models import MashupPlan, ExecutionLog as InternalExecutionLog


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class CreateAIMashupRequest(BaseModel):
    """Request to create an AI-powered mashup."""
    
    mashup_style: Optional[Literal['dj_mix', 'creative_mashup', 'remix', 'quick_mix']] = Field(
        default='dj_mix',
        description="Predefined mashup style (ignored if custom_instructions provided)"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Free-form text describing the desired mashup. Overrides mashup_style if provided.",
        example="Create a high-energy festival mashup with heavy bass drops and dramatic buildups"
    )
    
    llm_provider: Optional[Literal['openai', 'anthropic', 'local']] = Field(
        default='openai',
        description="LLM provider to use for plan generation"
    )
    
    llm_model: Optional[str] = Field(
        default=None,
        description="Specific LLM model to use (uses provider default if not specified)"
    )
    
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM creativity (0=deterministic, 1=very creative)"
    )
    
    track_a_name: Optional[str] = Field(
        default="Track A",
        description="Name of the first track"
    )
    
    track_b_name: Optional[str] = Field(
        default="Track B",
        description="Name of the second track"
    )


class PreviewPlanRequest(BaseModel):
    """Request to preview a mashup plan without execution."""
    
    mashup_style: Optional[Literal['dj_mix', 'creative_mashup', 'remix', 'quick_mix']] = Field(
        default='dj_mix',
        description="Predefined mashup style"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Free-form text describing the desired mashup"
    )
    
    llm_provider: Optional[Literal['openai', 'anthropic', 'local']] = Field(
        default='openai',
        description="LLM provider"
    )
    
    track_a_name: Optional[str] = Field(default="Track A")
    track_b_name: Optional[str] = Field(default="Track B")


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class ExecutionLogEntry(BaseModel):
    """Single execution log entry."""
    step: int
    operation: str
    status: Literal['success', 'failed', 'skipped']
    message: str


class TaskStatusResponse(BaseModel):
    """Response for task status check."""
    
    task_id: str = Field(description="Unique task identifier")
    
    status: Literal['queued', 'processing', 'completed', 'failed'] = Field(
        description="Current task status"
    )
    
    progress: int = Field(
        ge=0,
        le=100,
        description="Progress percentage (0-100)"
    )
    
    message: str = Field(
        description="Current progress message"
    )
    
    mashup_url: Optional[str] = Field(
        default=None,
        description="Download URL for completed mashup (only when status=completed)"
    )
    
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of completed mashup in seconds"
    )
    
    execution_log: Optional[List[ExecutionLogEntry]] = Field(
        default=None,
        description="Detailed execution log (only when completed/failed)"
    )
    
    errors: Optional[List[str]] = Field(
        default=None,
        description="Error messages (only when status=failed)"
    )
    
    created_at: str = Field(
        description="ISO timestamp when task was created"
    )
    
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when task completed (if applicable)"
    )


class CreateAIMashupResponse(BaseModel):
    """Response when creating a new AI mashup task."""
    
    task_id: str = Field(description="Unique task identifier")
    
    status: Literal['queued', 'processing'] = Field(
        description="Initial task status"
    )
    
    message: str = Field(
        description="Status message"
    )
    
    websocket_url: str = Field(
        description="WebSocket URL for real-time progress updates"
    )
    
    status_url: str = Field(
        description="HTTP URL to check task status"
    )


class PreviewPlanResponse(BaseModel):
    """Response for plan preview (without execution)."""
    
    status: Literal['success', 'failed'] = Field(
        description="Plan generation status"
    )
    
    plan: Optional[dict] = Field(
        default=None,
        description="Generated mashup plan (if successful)"
    )
    
    errors: Optional[List[str]] = Field(
        default=None,
        description="Errors during plan generation (if failed)"
    )
    
    validation_warnings: Optional[List[str]] = Field(
        default=None,
        description="Validation warnings about the plan"
    )
    
    estimated_duration: Optional[float] = Field(
        default=None,
        description="Estimated duration of mashup if executed (seconds)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: Literal['healthy', 'degraded', 'unhealthy']
    
    llm_providers_available: List[str] = Field(
        description="Available LLM providers"
    )
    
    operations_count: int = Field(
        description="Number of available audio operations"
    )
    
    utils_modules: List[str] = Field(
        description="Loaded utility modules"
    )
    
    errors: Optional[List[str]] = Field(
        default=None,
        description="Any health check errors"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
    task_id: Optional[str] = Field(default=None, description="Task ID if applicable")


# =============================================================================
# WEBSOCKET MESSAGE SCHEMAS
# =============================================================================

class WebSocketProgressMessage(BaseModel):
    """Progress update message sent via WebSocket."""
    
    type: Literal['progress', 'log', 'completed', 'failed']
    
    task_id: str
    
    progress: Optional[int] = Field(
        default=None,
        description="Progress percentage (0-100)"
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Progress message"
    )
    
    log_entry: Optional[ExecutionLogEntry] = Field(
        default=None,
        description="Execution log entry (for type=log)"
    )
    
    mashup_url: Optional[str] = Field(
        default=None,
        description="Download URL (for type=completed)"
    )
    
    errors: Optional[List[str]] = Field(
        default=None,
        description="Error messages (for type=failed)"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_internal_log_to_api(log: InternalExecutionLog) -> ExecutionLogEntry:
    """Convert internal execution log to API schema."""
    return ExecutionLogEntry(
        step=log.step,
        operation=log.operation,
        status=log.status,
        message=log.message
    )


def convert_plan_to_dict(plan: MashupPlan) -> dict:
    """Convert internal MashupPlan to API-friendly dict."""
    return plan.model_dump(mode='json')


# =============================================================================
# EXAMPLES FOR API DOCUMENTATION
# =============================================================================

CREATE_MASHUP_EXAMPLE = {
    "mashup_style": "dj_mix",
    "track_a_name": "Shape of You",
    "track_b_name": "Closer",
    "llm_provider": "openai",
    "temperature": 0.7
}

CUSTOM_MASHUP_EXAMPLE = {
    "custom_instructions": "Create a high-energy festival mashup with heavy bass drops every 16 bars and dramatic buildups. Keep energy consistently high.",
    "track_a_name": "Track A",
    "track_b_name": "Track B",
    "llm_provider": "openai"
}

TASK_STATUS_EXAMPLE = {
    "task_id": "abc123def456",
    "status": "completed",
    "progress": 100,
    "message": "Mashup completed successfully",
    "mashup_url": "/api/v2/download/abc123def456",
    "duration_seconds": 245.3,
    "execution_log": [
        {"step": 1, "operation": "transpose_to_key", "status": "success", "message": "Transposed Track A to D minor"},
        {"step": 2, "operation": "time_stretch_to_bpm", "status": "success", "message": "Stretched Track B to 128 BPM"}
    ],
    "created_at": "2025-10-11T10:30:00Z",
    "completed_at": "2025-10-11T10:32:15Z"
}
