"""
Operation Executor

Maps LLM operation names to actual utility functions and executes them.
Handles parameter passing, error handling, and logging.

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Callable, Optional
import numpy as np

# Import all utility functions
from app.utils import key_manipulations as key_utils
from app.utils import bpm_manipulations as bpm_utils
from app.utils import volume_manipulations as vol_utils
from app.utils import mixing_manipulations as mix_utils
from app.utils import audio_analysis as analysis_utils
from app.utils import effects_manipulations as fx_utils
from app.utils import transition_manipulations as trans_utils

logger = logging.getLogger(__name__)


# =============================================================================
# OPERATION REGISTRY
# =============================================================================

# This maps LLM operation names to actual functions
OPERATION_REGISTRY: Dict[str, Callable] = {
    # Key manipulation
    'detect_key': key_utils.detect_key,
    'transpose_to_key': key_utils.transpose_to_key,
    'pitch_shift': key_utils.pitch_shift,
    'pitch_shift_semitones': key_utils.pitch_shift_semitones,
    'get_compatible_keys': key_utils.get_compatible_keys,
    
    # BPM manipulation
    'detect_bpm': bpm_utils.detect_bpm,
    'time_stretch_to_bpm': bpm_utils.time_stretch_to_bpm,
    'detect_beats': bpm_utils.detect_beats,
    'align_beats': bpm_utils.align_beats,
    'detect_downbeats': bpm_utils.detect_downbeats,
    'quantize_to_beat_grid': bpm_utils.quantize_to_beat_grid,
    
    # Volume manipulation
    'normalize_lufs': vol_utils.normalize_lufs,
    'normalize_peak': vol_utils.normalize_peak,
    'normalize_rms': vol_utils.normalize_rms,
    'apply_gain': vol_utils.apply_gain,
    'compress': vol_utils.compress,
    'limit': vol_utils.limit,
    'gate': vol_utils.gate,
    'auto_gain_match': vol_utils.auto_gain_match,
    'sidechain_compress': vol_utils.sidechain_compress,
    'parallel_compress': vol_utils.parallel_compress,
    
    # Mixing
    'crossfade_equal_power': mix_utils.crossfade_equal_power,
    'crossfade_frequency': mix_utils.crossfade_frequency,
    'crossfade_exponential': mix_utils.crossfade_exponential,
    'blend_tracks': mix_utils.blend_tracks,
    'eq_highpass': mix_utils.eq_highpass,
    'eq_lowpass': mix_utils.eq_lowpass,
    'eq_bandpass': mix_utils.eq_bandpass,
    'pan': mix_utils.pan,
    'stereo_width': mix_utils.stereo_width,
    
    # Analysis
    'analyze_track': analysis_utils.analyze_track,
    'compare_tracks': analysis_utils.compare_tracks,
    'find_mix_points': analysis_utils.find_mix_points,
    'get_spectral_features': analysis_utils.get_spectral_features,
    'get_harmonic_features': analysis_utils.get_harmonic_features,
    'suggest_mashup_strategy': analysis_utils.suggest_mashup_strategy,
    
    # Effects
    'apply_reverb': fx_utils.apply_reverb,
    'apply_delay': fx_utils.apply_delay,
    'apply_echo': fx_utils.apply_echo,
    'apply_chorus': fx_utils.apply_chorus,
    'apply_flanger': fx_utils.apply_flanger,
    'apply_phaser': fx_utils.apply_phaser,
    'apply_tremolo': fx_utils.apply_tremolo,
    'apply_vibrato': fx_utils.apply_vibrato,
    'apply_distortion': fx_utils.apply_distortion,
    'apply_saturation': fx_utils.apply_saturation,
    'apply_bitcrush': fx_utils.apply_bitcrush,
    'apply_ring_mod': fx_utils.apply_ring_mod,
    
    # Transitions
    'create_riser': trans_utils.create_riser,
    'create_white_noise_riser': trans_utils.create_white_noise_riser,
    'create_filter_riser': trans_utils.create_filter_riser,
    'apply_buildup': trans_utils.apply_buildup,
    'create_impact': trans_utils.create_impact,
    'apply_drop': trans_utils.apply_drop,
    'create_reverse_cymbal': trans_utils.create_reverse_cymbal,
    'apply_echo_out': trans_utils.apply_echo_out,
    'apply_delay_throw': trans_utils.apply_delay_throw,
    'apply_stutter_buildup': trans_utils.apply_stutter_buildup,
    'apply_beat_repeat': trans_utils.apply_beat_repeat,
    'apply_half_time': trans_utils.apply_half_time,
    'apply_double_time': trans_utils.apply_double_time,
    'apply_reverse_buildup': trans_utils.apply_reverse_buildup,
    'apply_spinback': trans_utils.apply_spinback,
    'apply_filter_sweep_transition': trans_utils.apply_filter_sweep_transition,
    'apply_highpass_sweep': trans_utils.apply_highpass_sweep,
    'apply_lowpass_sweep': trans_utils.apply_lowpass_sweep,
    'apply_vinyl_stop': trans_utils.apply_vinyl_stop,
    'apply_scratch': trans_utils.apply_scratch,
    'create_silence_gap': trans_utils.create_silence_gap,
    'apply_gate_stutter': trans_utils.apply_gate_stutter,
}


# =============================================================================
# EXECUTOR
# =============================================================================

class OperationExecutor:
    """Executes operations from LLM plans."""
    
    def __init__(self, sr: int = 44100):
        """
        Initialize executor.
        
        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr
        self.execution_log = []
    
    def execute_operation(
        self,
        operation_name: str,
        parameters: Dict[str, Any],
        audio_data: Optional[np.ndarray] = None
    ) -> Any:
        """
        Execute a single operation.
        
        Args:
            operation_name: Name of operation to execute
            parameters: Parameters for the operation
            audio_data: Audio data to process (if needed)
        
        Returns:
            Result of operation (varies by operation)
        
        Raises:
            ValueError: If operation is unknown or invalid
        """
        if operation_name not in OPERATION_REGISTRY:
            logger.error(f"Unknown operation: {operation_name}")
            raise ValueError(f"Unknown operation: {operation_name}")
        
        # Get the function
        func = OPERATION_REGISTRY[operation_name]
        
        try:
            # Add sample rate if needed
            if 'sr' in func.__code__.co_varnames:
                parameters['sr'] = self.sr
            
            # Special handling for time_stretch_to_bpm - auto-detect source_bpm if missing
            if operation_name == 'time_stretch_to_bpm' and 'source_bpm' not in parameters and audio_data is not None:
                logger.info("Auto-detecting source BPM for time_stretch_to_bpm")
                from app.utils.bpm_manipulations import detect_bpm
                source_bpm, _ = detect_bpm(audio_data, self.sr)
                parameters['source_bpm'] = source_bpm
                logger.info(f"Detected source BPM: {source_bpm:.1f}")
            
            # Parameter name mapping - handle common LLM parameter name variations
            parameter_mappings = {
                'apply_delay': {'time': 'delay_time'},
                'apply_reverb': {'time': 'reverb_time', 'size': 'room_size'},
                'apply_echo': {'time': 'delay_time'},
            }
            
            if operation_name in parameter_mappings:
                mappings = parameter_mappings[operation_name]
                for old_name, new_name in mappings.items():
                    if old_name in parameters and new_name not in parameters:
                        parameters[new_name] = parameters.pop(old_name)
                        logger.debug(f"Mapped parameter '{old_name}' -> '{new_name}' for {operation_name}")
            
            # Add audio data if needed (check for both 'y' and 'audio' parameter names)
            if audio_data is not None:
                if 'y' in func.__code__.co_varnames:
                    result = func(audio_data, **parameters)
                elif 'audio' in func.__code__.co_varnames:
                    result = func(audio_data, **parameters)
                else:
                    result = func(**parameters)
            else:
                result = func(**parameters)
            
            # Handle functions that return tuples (various formats)
            if isinstance(result, tuple):
                # Check if first element is audio data (numpy array)
                if len(result) >= 1 and isinstance(result[0], np.ndarray):
                    if len(result) == 2:
                        # Most common: (audio, metadata)
                        audio_result, metadata = result
                        logger.debug(f"{operation_name} returned (audio, metadata)")
                        result = audio_result
                    elif len(result) > 2:
                        # Multi-output: (audio1, audio2, ...) or (audio, meta1, meta2, ...)
                        logger.warning(f"{operation_name} returned tuple of length {len(result)}, using first element only")
                        result = result[0]
                    else:
                        # Single element tuple
                        result = result[0]
                else:
                    # Tuple doesn't start with audio (e.g., detect_bpm returns (float, float))
                    # These are analysis functions that don't modify audio
                    logger.warning(f"{operation_name} returned non-audio data {type(result[0]).__name__} - analysis operation, not modifying audio")
                    # Return original audio unchanged if this was called on audio
                    if audio_data is not None:
                        result = audio_data
                    # Otherwise keep the analysis result (might be used for something else)
            
            # Log success
            self.execution_log.append({
                'operation': operation_name,
                'status': 'success',
                'message': f'Successfully executed {operation_name}'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {operation_name}: {e}")
            self.execution_log.append({
                'operation': operation_name,
                'status': 'failed',
                'message': str(e)
            })
            raise
    
    def get_log(self) -> list:
        """Get execution log."""
        return self.execution_log
    
    def clear_log(self):
        """Clear execution log."""
        self.execution_log = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_operation(operation_name: str) -> bool:
    """
    Check if an operation is valid.
    
    Args:
        operation_name: Operation name to validate
    
    Returns:
        True if operation exists, False otherwise
    """
    return operation_name in OPERATION_REGISTRY


def get_available_operations() -> list:
    """
    Get list of all available operations.
    
    Returns:
        List of operation names
    """
    return list(OPERATION_REGISTRY.keys())


def get_operation_categories() -> Dict[str, list]:
    """
    Get operations grouped by category.
    
    Returns:
        Dictionary mapping categories to operation lists
    """
    categories = {
        'key': [],
        'bpm': [],
        'volume': [],
        'mixing': [],
        'analysis': [],
        'effects': [],
        'transitions': []
    }
    
    for op in OPERATION_REGISTRY.keys():
        if 'key' in op or 'pitch' in op or 'transpose' in op:
            categories['key'].append(op)
        elif 'bpm' in op or 'beat' in op or 'tempo' in op or 'stretch' in op or 'time' in op:
            categories['bpm'].append(op)
        elif 'volume' in op or 'gain' in op or 'normalize' in op or 'compress' in op or 'limit' in op:
            categories['volume'].append(op)
        elif 'crossfade' in op or 'blend' in op or 'eq_' in op or 'stereo' in op or 'pan' in op:
            categories['mixing'].append(op)
        elif 'analyze' in op or 'compare' in op or 'find' in op or 'get_' in op or 'suggest' in op:
            categories['analysis'].append(op)
        elif 'reverb' in op or 'delay' in op or 'echo' in op or 'chorus' in op or 'flanger' in op or \
             'phaser' in op or 'tremolo' in op or 'vibrato' in op or 'distortion' in op or 'saturation' in op or \
             'bitcrush' in op or 'ring' in op:
            categories['effects'].append(op)
        elif 'riser' in op or 'buildup' in op or 'drop' in op or 'cymbal' in op or 'stutter' in op or \
             'half_time' in op or 'double_time' in op or 'reverse' in op or 'spinback' in op or \
             'sweep' in op or 'vinyl' in op or 'scratch' in op or 'silence' in op or 'gate' in op:
            categories['transitions'].append(op)
    
    return categories


def extract_section(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float
) -> np.ndarray:
    """
    Extract a time range from audio.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        Extracted audio section
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample].copy()


# Add extract_section to registry
OPERATION_REGISTRY['extract_section'] = extract_section


if __name__ == '__main__':
    print("Operation Executor Module")
    print(f"Total operations available: {len(OPERATION_REGISTRY)}")
    
    categories = get_operation_categories()
    for category, ops in categories.items():
        print(f"\n{category.upper()}: {len(ops)} operations")
        for op in sorted(ops)[:5]:  # Show first 5
            print(f"  - {op}")
        if len(ops) > 5:
            print(f"  ... and {len(ops) - 5} more")
