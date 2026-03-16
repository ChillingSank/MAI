"""
Pydantic Models for LLM-Generated Mashup Plans

These models define the structured output format that the LLM must follow
when generating mashup plans. This ensures reliable parsing and execution.

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# COMPATIBILITY ANALYSIS
# =============================================================================

class CompatibilityAnalysis(BaseModel):
    """Analysis of how compatible two tracks are for mashup."""
    
    key_compatibility: Literal['compatible', 'needs_adjustment'] = Field(
        description="Whether keys are harmonically compatible"
    )
    bpm_compatibility: Literal['compatible', 'needs_adjustment'] = Field(
        description="Whether BPMs are compatible or need adjustment"
    )
    energy_match: Literal['good', 'fair', 'poor'] = Field(
        description="How well energy levels match"
    )
    overall_score: int = Field(
        ge=0, le=100,
        description="Overall compatibility score (0-100)"
    )
    reasoning: str = Field(
        description="Detailed explanation of compatibility analysis"
    )


# =============================================================================
# PREPROCESSING STEPS
# =============================================================================

class PreprocessingStep(BaseModel):
    """A single preprocessing operation to apply to a track."""
    
    track: Literal['a', 'b'] = Field(
        description="Which track to apply operation to"
    )
    operation: str = Field(
        description="Operation name (e.g., 'transpose_to_key', 'time_stretch_to_bpm')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the operation"
    )
    reason: str = Field(
        description="Why this preprocessing step is necessary"
    )


# =============================================================================
# MASHUP STRUCTURE
# =============================================================================

class MashupStructure(BaseModel):
    """Overall structure/arrangement of the mashup."""
    
    type: Literal['sequential', 'parallel', 'sequential_with_overlap', 'sandwich'] = Field(
        description="How tracks are arranged"
    )
    description: str = Field(
        description="Detailed description of the mashup structure"
    )


# =============================================================================
# MIX PLAN STEPS
# =============================================================================

class MixStep(BaseModel):
    """A single step in the mixing/mashup creation process."""
    
    step: int = Field(
        description="Step number in sequence"
    )
    action: str = Field(
        description="Human-readable description of what happens"
    )
    operation: str = Field(
        description="Function name to call (e.g., 'crossfade_equal_power')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the operation"
    )
    timing: str = Field(
        description="When this happens (e.g., '0:45 - 0:53')"
    )
    reason: str = Field(
        description="Why this operation works musically"
    )


# =============================================================================
# CREATIVE ENHANCEMENTS
# =============================================================================

class CreativeEnhancement(BaseModel):
    """Optional creative effects to enhance the mashup."""
    
    effect: str = Field(
        description="Effect name (e.g., 'apply_reverb', 'apply_delay')"
    )
    target: Literal['track_a', 'track_b', 'mix'] = Field(
        description="What to apply the effect to"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Effect parameters"
    )
    placement: str = Field(
        description="Where/when to apply the effect"
    )


# =============================================================================
# COMPLETE MASHUP PLAN
# =============================================================================

class MashupPlan(BaseModel):
    """Complete LLM-generated mashup plan."""
    
    compatibility_analysis: CompatibilityAnalysis = Field(
        description="Analysis of track compatibility"
    )
    preprocessing: List[PreprocessingStep] = Field(
        default_factory=list,
        description="Preprocessing steps to apply before mixing"
    )
    mashup_structure: MashupStructure = Field(
        description="Overall mashup structure"
    )
    mix_plan: List[MixStep] = Field(
        description="Detailed mixing steps in order"
    )
    creative_enhancements: List[CreativeEnhancement] = Field(
        default_factory=list,
        description="Optional creative effects"
    )
    final_notes: str = Field(
        description="Additional tips and suggestions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "compatibility_analysis": {
                    "key_compatibility": "needs_adjustment",
                    "bpm_compatibility": "needs_adjustment",
                    "energy_match": "good",
                    "overall_score": 78,
                    "reasoning": "Both tracks are high-energy pop with strong beats..."
                },
                "preprocessing": [
                    {
                        "track": "a",
                        "operation": "transpose_to_key",
                        "parameters": {"target_key": "F# minor"},
                        "reason": "Harmonize with track B"
                    }
                ],
                "mashup_structure": {
                    "type": "sequential_with_overlap",
                    "description": "Intro from A, crossfade to B, layer vocals"
                },
                "mix_plan": [
                    {
                        "step": 1,
                        "action": "Start with Track A intro",
                        "operation": "extract_section",
                        "parameters": {"track": "a", "start": 0, "end": 45.2},
                        "timing": "0:00 - 0:45",
                        "reason": "Strong intro establishes energy"
                    }
                ],
                "creative_enhancements": [
                    {
                        "effect": "apply_reverb",
                        "target": "track_a",
                        "parameters": {"room_size": "large", "wet": 0.3},
                        "placement": "During intro section"
                    }
                ],
                "final_notes": "This mashup works because both tracks share high energy..."
            }
        }


# =============================================================================
# EXECUTION RESULTS
# =============================================================================

class ExecutionLog(BaseModel):
    """Log entry for an executed operation."""
    
    step: int = Field(description="Step number")
    operation: str = Field(description="Operation that was executed")
    status: Literal['success', 'failed', 'skipped'] = Field(description="Execution status")
    message: str = Field(description="Result message")
    duration_seconds: Optional[float] = Field(None, description="How long operation took")


class MashupResult(BaseModel):
    """Final result of mashup creation."""
    
    status: Literal['success', 'partial', 'failed'] = Field(
        description="Overall status"
    )
    mashup_file: Optional[str] = Field(
        None,
        description="Path to generated mashup file"
    )
    plan_used: MashupPlan = Field(
        description="The plan that was executed"
    )
    execution_log: List[ExecutionLog] = Field(
        default_factory=list,
        description="Log of all operations performed"
    )
    total_duration: Optional[float] = Field(
        None,
        description="Total mashup duration in seconds"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )


# =============================================================================
# TRACK FEATURES (for sending to LLM)
# =============================================================================

class TrackFeatures(BaseModel):
    """Comprehensive features extracted from a track."""
    
    name: str = Field(description="Track name")
    duration: float = Field(description="Duration in seconds")
    key: str = Field(description="Musical key (e.g., 'C major')")
    camelot_code: str = Field(description="Camelot wheel code (e.g., '8B')")
    bpm: float = Field(description="Tempo in BPM")
    energy: float = Field(ge=0, le=1, description="Energy level (0-1)")
    danceability: float = Field(ge=0, le=1, description="Danceability (0-1)")
    
    frequency_balance: Dict[str, float] = Field(
        description="Bass/mid/high frequency distribution"
    )
    harmonic_features: Dict[str, Any] = Field(
        description="Key, mode, harmonicity, etc."
    )
    rhythm_features: Dict[str, Any] = Field(
        description="Groove, beat strength, onset rate"
    )
    sections: Dict[str, List[float]] = Field(
        description="Detected sections (intro, verse, chorus, outro)"
    )
    transitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected transitions (build-ups, drops)"
    )
    stereo_width: float = Field(ge=0, le=1, description="Stereo width")
    loudness_lufs: float = Field(description="Loudness in LUFS")


class TwoTrackAnalysis(BaseModel):
    """Features from both tracks for LLM analysis."""
    
    track_a: TrackFeatures = Field(description="Track A features")
    track_b: TrackFeatures = Field(description="Track B features")
    mashup_style: str = Field(
        default="dj_mix",
        description="Desired mashup style (dj_mix, creative_mashup, remix)"
    )


if __name__ == '__main__':
    # Test model serialization
    print("Pydantic Models for AI Mashup Producer")
    print(f"Models: {len([CompatibilityAnalysis, PreprocessingStep, MashupStructure, MixStep, CreativeEnhancement, MashupPlan, ExecutionLog, MashupResult, TrackFeatures, TwoTrackAnalysis])}")
    
    # Test creating a sample plan
    sample_plan = MashupPlan(
        compatibility_analysis=CompatibilityAnalysis(
            key_compatibility='compatible',
            bpm_compatibility='needs_adjustment',
            energy_match='good',
            overall_score=85,
            reasoning='Test reasoning'
        ),
        preprocessing=[],
        mashup_structure=MashupStructure(
            type='sequential',
            description='Test structure'
        ),
        mix_plan=[],
        creative_enhancements=[],
        final_notes='Test notes'
    )
    
    print(f"✓ Sample plan created successfully")
    print(f"✓ JSON serialization: {len(sample_plan.model_dump_json())} chars")
