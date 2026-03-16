"""
Validators for LLM Output

Validates and sanitizes LLM-generated mashup plans before execution.
Ensures safety and correctness of operations and parameters.

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from app.llm.models import (
    MashupPlan,
    PreprocessingStep,
    MixStep,
    CreativeEnhancement
)
from app.llm.executor import validate_operation

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER LIMITS
# =============================================================================

PARAMETER_LIMITS = {
    # Key manipulation
    'semitones': (-12, 12),
    
    # BPM manipulation
    'target_bpm': (60, 200),
    'bpm': (60, 200),
    
    # Volume manipulation
    'gain_db': (-20, 20),
    'gain': (0.0, 2.0),
    'target_lufs': (-20, -5),
    'target_db': (-20, 0),
    'threshold_db': (-60, 0),
    'ratio': (1.0, 20.0),
    'attack': (0.001, 1.0),
    'release': (0.01, 5.0),
    
    # Mixing
    'duration': (0.1, 30.0),
    'wet': (0.0, 1.0),
    'dry': (0.0, 1.0),
    'crossfade_duration': (0.1, 10.0),
    'gain_a': (0.0, 2.0),
    'gain_b': (0.0, 2.0),
    
    # Effects
    'feedback': (0.0, 0.95),
    'decay': (0.0, 1.0),
    'depth': (0.0, 1.0),
    'rate_hz': (0.1, 20.0),
    'delay_time': (0.01, 2.0),
    'echo_duration': (0.5, 10.0),
    
    # Transitions
    'buildup_duration': (1.0, 16.0),
    'drop_duration': (0.0, 2.0),
    'silence_duration': (0.0, 2.0),
    'repeats': (1, 16),
    
    # Time ranges
    'start_time': (0.0, 10000.0),  # Will be validated against track duration
    'end_time': (0.0, 10000.0),
    'start': (0.0, 10000.0),
    'end': (0.0, 10000.0),
}


# =============================================================================
# VALIDATORS
# =============================================================================

class PlanValidator:
    """Validates LLM-generated mashup plans."""
    
    def __init__(
        self,
        track_a_duration: float,
        track_b_duration: float,
        strict: bool = False
    ):
        """
        Initialize validator.
        
        Args:
            track_a_duration: Duration of Track A in seconds
            track_b_duration: Duration of Track B in seconds
            strict: If True, raise errors; if False, log warnings and fix
        """
        self.track_a_duration = track_a_duration
        self.track_b_duration = track_b_duration
        self.strict = strict
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def validate_plan(self, plan: MashupPlan) -> bool:
        """
        Validate entire mashup plan with auto-correction.
        
        Args:
            plan: Mashup plan to validate
        
        Returns:
            True if valid, False otherwise
        """
        self.warnings = []
        self.errors = []
        
        # Validate preprocessing steps
        for step in plan.preprocessing:
            self._validate_preprocessing_step(step)
        
        # Check for aggressive preprocessing combinations
        self._check_preprocessing_combinations(plan.preprocessing)
        
        # Validate mix plan steps
        for step in plan.mix_plan:
            self._validate_mix_step(step)
        
        # Validate creative enhancements
        for enhancement in plan.creative_enhancements:
            self._validate_enhancement(enhancement)
        
        if self.errors:
            logger.error(f"Plan validation failed with {len(self.errors)} errors")
            return False
        
        if self.warnings:
            logger.warning(f"Plan has {len(self.warnings)} warnings")
        
        return True
    
    def _validate_preprocessing_step(self, step: PreprocessingStep):
        """Validate a preprocessing step."""
        # Check operation exists
        if not validate_operation(step.operation):
            msg = f"Unknown operation: {step.operation}"
            if self.strict:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)
        
        # Validate parameters
        self._validate_parameters(step.operation, step.parameters, step.track)
    
    def _check_preprocessing_combinations(self, steps: List[PreprocessingStep]):
        """
        Check for dangerous preprocessing combinations (e.g., transpose + time_stretch).
        The 'chipmunk effect' happens when both pitch shift AND tempo change are applied.
        """
        # Group steps by track
        track_a_steps = [s for s in steps if s.track in ['a', 'track_a']]
        track_b_steps = [s for s in steps if s.track in ['b', 'track_b']]
        
        for track_name, track_steps in [('Track A', track_a_steps), ('Track B', track_b_steps)]:
            has_transpose = False
            has_time_stretch = False
            transpose_amount = 0
            stretch_amount = 0
            
            for step in track_steps:
                if step.operation in ['transpose', 'pitch_shift']:
                    has_transpose = True
                    transpose_amount = step.parameters.get('semitones', 0)
                
                if step.operation in ['time_stretch', 'change_tempo']:
                    has_time_stretch = True
                    # time_stretch might use 'rate' (1.2 = 20% faster) or 'factor'
                    rate = step.parameters.get('rate', step.parameters.get('factor', 1.0))
                    stretch_amount = abs(1.0 - rate) * 100  # Convert to percentage
            
            # CRITICAL WARNING: Both transpose and time_stretch
            if has_transpose and has_time_stretch:
                if abs(transpose_amount) > 3 and stretch_amount > 10:
                    msg = (f"{track_name}: DANGEROUS combination - transpose ({transpose_amount:+d} semitones) "
                           f"AND time_stretch ({stretch_amount:.0f}%) will cause chipmunk/distortion effects. "
                           f"Consider using ONLY key matching (transpose) without tempo changes.")
                    self.warnings.append(msg)
                    logger.warning(msg)
                else:
                    msg = (f"{track_name}: Combining transpose ({transpose_amount:+d}) and time_stretch "
                           f"({stretch_amount:.0f}%), may affect quality")
                    self.warnings.append(msg)
    
    def _validate_mix_step(self, step: MixStep):
        """Validate a mix plan step with auto-correction for section lengths."""
        # Check operation exists
        if not validate_operation(step.operation):
            msg = f"Unknown operation in step {step.step}: {step.operation}"
            if self.strict:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)
        
        # Validate parameters
        track_id = step.parameters.get('track', step.parameters.get('track_id', 'a'))
        self._validate_parameters(step.operation, step.parameters, track_id)
        
        # AUTO-CORRECTION: Check section lengths for extract operations
        if step.operation in ['extract', 'extract_section']:
            start = step.parameters.get('start_time', step.parameters.get('start', 0))
            end = step.parameters.get('end_time', step.parameters.get('end', 0))
            section_length = end - start
            
            # Maximum section length: 40 seconds
            MAX_SECTION_LENGTH = 40.0
            if section_length > MAX_SECTION_LENGTH:
                # Auto-trim to 40 seconds
                new_end = start + MAX_SECTION_LENGTH
                msg = f"Step {step.step}: Section too long ({section_length:.1f}s), auto-trimmed to {MAX_SECTION_LENGTH}s"
                self.warnings.append(msg)
                logger.warning(msg)
                
                # Update the parameter
                if 'end_time' in step.parameters:
                    step.parameters['end_time'] = new_end
                else:
                    step.parameters['end'] = new_end
            
            # Minimum section length: 8 seconds (flag warning)
            MIN_SECTION_LENGTH = 8.0
            if section_length < MIN_SECTION_LENGTH:
                msg = f"Step {step.step}: Section very short ({section_length:.1f}s), may cause abrupt transitions"
                self.warnings.append(msg)
                logger.warning(msg)
        
        # AUTO-CORRECTION: Check for aggressive preprocessing combinations
        if step.operation in ['transpose', 'pitch_shift']:
            semitones = step.parameters.get('semitones', 0)
            if abs(semitones) > 4:
                msg = f"Step {step.step}: Large pitch shift ({semitones:+d} semitones) may cause quality issues"
                self.warnings.append(msg)
    
    def _validate_enhancement(self, enhancement: CreativeEnhancement):
        """Validate a creative enhancement."""
        # Check operation exists
        if not validate_operation(enhancement.effect):
            msg = f"Unknown effect: {enhancement.effect}"
            if self.strict:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)
        
        # Validate parameters
        track_id = 'a' if enhancement.target == 'track_a' else 'b'
        self._validate_parameters(enhancement.effect, enhancement.parameters, track_id)
    
    def _validate_parameters(self, operation: str, parameters: Dict[str, Any], track_id: str):
        """Validate and sanitize operation parameters."""
        # Get track duration
        track_duration = self.track_a_duration if track_id in ['a', 'track_a'] else self.track_b_duration
        
        for param_name, param_value in parameters.items():
            # Skip non-numeric parameters
            if not isinstance(param_value, (int, float)):
                continue
            
            # Check if parameter has limits
            if param_name in PARAMETER_LIMITS:
                min_val, max_val = PARAMETER_LIMITS[param_name]
                
                # For time parameters, use track duration as max
                if param_name in ['start_time', 'end_time', 'start', 'end']:
                    max_val = track_duration
                
                # Check bounds
                if param_value < min_val or param_value > max_val:
                    # Clamp value
                    clamped = max(min_val, min(max_val, param_value))
                    msg = f"Parameter {param_name} out of range ({param_value}) in {operation}, clamped to {clamped}"
                    self.warnings.append(msg)
                    parameters[param_name] = clamped
            
            # Special validation for time ranges
            if param_name == 'end_time' or param_name == 'end':
                start_param = 'start_time' if param_name == 'end_time' else 'start'
                start_val = parameters.get(start_param, 0)
                if param_value <= start_val:
                    msg = f"End time ({param_value}) must be greater than start time ({start_val}) in {operation}"
                    self.errors.append(msg)
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings."""
        return self.warnings
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors


# =============================================================================
# SANITIZATION FUNCTIONS
# =============================================================================

def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize parameters to safe ranges.
    
    Args:
        parameters: Parameter dictionary
    
    Returns:
        Sanitized parameters
    """
    sanitized = parameters.copy()
    
    for param_name, param_value in parameters.items():
        if not isinstance(param_value, (int, float)):
            continue
        
        if param_name in PARAMETER_LIMITS:
            min_val, max_val = PARAMETER_LIMITS[param_name]
            sanitized[param_name] = max(min_val, min(max_val, param_value))
    
    return sanitized


def validate_track_id(track_id: str) -> str:
    """
    Validate and normalize track ID.
    
    Args:
        track_id: Track identifier
    
    Returns:
        Normalized track ID ('a' or 'b')
    """
    track_id = str(track_id).lower().strip()
    
    if track_id in ['a', 'track_a', 'track a']:
        return 'a'
    elif track_id in ['b', 'track_b', 'track b']:
        return 'b'
    else:
        logger.warning(f"Unknown track ID: {track_id}, defaulting to 'a'")
        return 'a'


def validate_time_range(
    start: float,
    end: float,
    duration: float
) -> tuple[float, float]:
    """
    Validate time range against track duration.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        duration: Track duration in seconds
    
    Returns:
        Validated (start, end) tuple
    """
    # Clamp to valid range
    start = max(0.0, min(start, duration))
    end = max(start + 0.1, min(end, duration))
    
    # Ensure end > start
    if end <= start:
        end = min(start + 1.0, duration)
    
    return start, end


def validate_key(key: str) -> Optional[str]:
    """
    Validate musical key string.
    
    Args:
        key: Key string (e.g., 'C major', 'A minor')
    
    Returns:
        Validated key or None if invalid
    """
    key = key.strip()
    
    # Valid notes
    valid_notes = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
    
    # Valid modes
    valid_modes = ['major', 'minor', 'maj', 'min', 'm', 'M']
    
    # Parse key
    parts = key.split()
    if len(parts) != 2:
        logger.warning(f"Invalid key format: {key}")
        return None
    
    note, mode = parts
    
    if note not in valid_notes:
        logger.warning(f"Invalid note: {note}")
        return None
    
    if mode not in valid_modes:
        logger.warning(f"Invalid mode: {mode}")
        return None
    
    # Normalize mode
    if mode in ['maj', 'M']:
        mode = 'major'
    elif mode in ['min', 'm']:
        mode = 'minor'
    
    return f"{note} {mode}"


# =============================================================================
# JSON SCHEMA VALIDATION
# =============================================================================

def validate_json_structure(data: Dict[str, Any]) -> bool:
    """
    Validate JSON structure matches expected schema.
    
    Args:
        data: JSON data to validate
    
    Returns:
        True if valid structure
    """
    required_keys = [
        'compatibility_analysis',
        'preprocessing',
        'mashup_structure',
        'mix_plan',
        'creative_enhancements',
        'final_notes'
    ]
    
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Validate compatibility_analysis
    if not isinstance(data['compatibility_analysis'], dict):
        logger.error("compatibility_analysis must be a dict")
        return False
    
    # Validate lists
    for key in ['preprocessing', 'mix_plan', 'creative_enhancements']:
        if not isinstance(data[key], list):
            logger.error(f"{key} must be a list")
            return False
    
    return True


if __name__ == '__main__':
    print("Validators Module")
    print(f"Parameter limits defined: {len(PARAMETER_LIMITS)}")
    
    # Test validator
    validator = PlanValidator(track_a_duration=200.0, track_b_duration=180.0, strict=False)
    print(f"\nValidator initialized for tracks: 200s and 180s")
    
    # Test parameter sanitization
    test_params = {
        'target_bpm': 250,  # Out of range
        'gain': 5.0,  # Out of range
        'wet': 0.5,  # Valid
    }
    sanitized = sanitize_parameters(test_params)
    print(f"\nSanitized parameters: {sanitized}")
