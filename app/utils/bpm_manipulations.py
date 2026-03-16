"""
BPM & Tempo Manipulation Utilities for Mashup Creation

This module provides fundamental BPM/tempo operations for music mashups.
All functions are independent and designed for easy programmatic use.

Core Functions:
- detect_bpm: Detect BPM with confidence scoring
- time_stretch_to_bpm: Time-stretch audio to target BPM
- detect_beats: Get beat positions (frame indices)
- get_beat_positions: Get beat positions in seconds
- detect_downbeats: Find bar-start positions (4/4 time)
- beats_to_seconds / seconds_to_beats: Time conversions
- bpm_are_compatible: Check if two BPMs can mix
- get_canonical_bpms: Get 0.5x, 1x, 2x variants

Advanced Functions:
- extract_beat_window: Extract audio between beat ranges
- calculate_stretch_ratio: Compute stretch factor between BPMs
- find_intro_by_beats: Detect clean intro sections
- find_outro_by_beats: Detect smooth outro sections
- estimate_optimal_crossfade_beats: Calculate crossfade length based on energy
- quantize_to_beat_grid: Snap audio to beat grid
- align_beats: Align two tracks by their beats
- detect_tempo_changes: Find tempo variations in track
- get_beat_strength: Get energy at beat positions
- create_beat_mask: Create mask for beat-synchronous processing
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import librosa

logger = logging.getLogger(__name__)

__all__ = [
    'detect_bpm',
    'time_stretch_to_bpm',
    'detect_beats',
    'get_beat_positions',
    'detect_downbeats',
    'beats_to_seconds',
    'seconds_to_beats',
    'bpm_are_compatible',
    'get_canonical_bpms',
    'extract_beat_window',
    'calculate_stretch_ratio',
    'find_intro_by_beats',
    'find_outro_by_beats',
    'estimate_optimal_crossfade_beats',
    'quantize_to_beat_grid',
    'align_beats',
    'detect_tempo_changes',
    'get_beat_strength',
    'create_beat_mask',
]


# =============================================================================
# CORE BPM DETECTION & MANIPULATION
# =============================================================================

def detect_bpm(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    return_beats: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    Detect BPM (tempo) of audio using onset strength analysis.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        hop_length: Frame hop length for analysis
        start_bpm: Initial BPM estimate to guide detection
        return_beats: If True, also return beat frame positions
    
    Returns:
        (bpm, confidence) or (bpm, confidence, beat_frames) if return_beats=True
        - bpm: Detected tempo in beats per minute
        - confidence: Detection confidence [0-1]
        - beat_frames: Frame indices of detected beats (if requested)
    
    Example:
        >>> bpm, conf = detect_bpm(audio, 44100)
        >>> print(f"Detected {bpm:.1f} BPM (confidence: {conf:.2f})")
    """
    try:
        # Use librosa's beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=y, 
            sr=sr, 
            hop_length=hop_length,
            start_bpm=start_bpm,
            units='frames'
        )
        
        bpm = float(tempo)
        
        # Calculate confidence based on beat consistency
        if len(beats) > 2:
            # Check inter-beat interval consistency
            intervals = np.diff(beats)
            if len(intervals) > 0:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                # Low variance = high confidence
                confidence = float(np.clip(1.0 - (std_interval / (mean_interval + 1)), 0.0, 1.0))
            else:
                confidence = 0.5
        else:
            confidence = 0.3
        
        if return_beats:
            return bpm, confidence, beats
        return bpm, confidence
        
    except Exception as e:
        logger.warning(f"BPM detection failed: {e}")
        if return_beats:
            return 120.0, 0.0, np.array([])
        return 120.0, 0.0


def time_stretch_to_bpm(
    y: np.ndarray,
    sr: int,
    source_bpm: float,
    target_bpm: float,
    preserve_pitch: bool = True
) -> np.ndarray:
    """
    Time-stretch audio to match a target BPM without changing pitch.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        source_bpm: Original BPM of the audio
        target_bpm: Desired target BPM
        preserve_pitch: If True, preserve pitch (default); if False, change pitch too
    
    Returns:
        Time-stretched audio array
    
    Example:
        >>> # Slow down 140 BPM track to 120 BPM
        >>> stretched = time_stretch_to_bpm(audio, 44100, 140.0, 120.0)
    
    Note:
        Stretch ratio is clamped to [0.5, 2.0] to maintain audio quality.
        For larger changes, use multiple passes or adjust BPM detection.
    """
    if source_bpm <= 0 or target_bpm <= 0:
        logger.warning("Invalid BPM values, returning original audio")
        return y
    
    # Calculate stretch ratio (target/source means speed up if target > source)
    ratio = target_bpm / source_bpm
    
    # Clamp to reasonable range for quality
    ratio = float(np.clip(ratio, 0.5, 2.0))
    
    if abs(ratio - 1.0) < 0.01:
        # No meaningful stretching needed
        return y
    
    try:
        if preserve_pitch:
            # Time stretch without pitch change
            y_stretched = librosa.effects.time_stretch(y, rate=ratio)
        else:
            # Simple resampling (changes both tempo and pitch)
            y_stretched = librosa.resample(y, orig_sr=sr, target_sr=int(sr * ratio))
        
        return y_stretched
    
    except Exception as e:
        logger.error(f"Time stretching failed: {e}")
        return y


def detect_beats(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    start_bpm: Optional[float] = None
) -> Tuple[float, np.ndarray]:
    """
    Detect beat positions in audio.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        hop_length: Frame hop length for analysis
        start_bpm: Optional BPM hint to improve detection
    
    Returns:
        (tempo, beat_frames)
        - tempo: Estimated BPM
        - beat_frames: Frame indices where beats occur
    
    Example:
        >>> tempo, beats = detect_beats(audio, 44100)
        >>> beat_times = librosa.frames_to_time(beats, sr=44100, hop_length=512)
    """
    try:
        kwargs = {'y': y, 'sr': sr, 'hop_length': hop_length, 'units': 'frames'}
        if start_bpm is not None:
            kwargs['start_bpm'] = start_bpm
        
        tempo, beats = librosa.beat.beat_track(**kwargs)
        return float(tempo), beats
    
    except Exception as e:
        logger.error(f"Beat detection failed: {e}")
        return 120.0, np.array([])


def get_beat_positions(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    start_bpm: Optional[float] = None
) -> Tuple[float, np.ndarray]:
    """
    Get beat positions in seconds (more intuitive than frames).
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        hop_length: Frame hop length for analysis
        start_bpm: Optional BPM hint
    
    Returns:
        (tempo, beat_times)
        - tempo: Estimated BPM
        - beat_times: Times in seconds where beats occur
    
    Example:
        >>> tempo, beat_times = get_beat_positions(audio, 44100)
        >>> print(f"First beat at {beat_times[0]:.2f}s")
    """
    tempo, beat_frames = detect_beats(y, sr, hop_length, start_bpm)
    
    if len(beat_frames) == 0:
        return tempo, np.array([])
    
    # Convert frames to seconds
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return tempo, beat_times


def detect_downbeats(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    time_signature: int = 4
) -> np.ndarray:
    """
    Detect downbeats (bar-start positions) assuming constant time signature.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        hop_length: Frame hop length
        time_signature: Beats per bar (default 4 for 4/4 time)
    
    Returns:
        Sample indices where downbeats (bar starts) occur
    
    Example:
        >>> downbeats = detect_downbeats(audio, 44100, time_signature=4)
        >>> downbeat_times = downbeats / 44100  # Convert to seconds
    
    Note:
        Assumes constant time signature throughout. For variable signatures,
        use detect_tempo_changes() first to segment the audio.
    """
    try:
        _, beat_frames = detect_beats(y, sr, hop_length)
        
        if len(beat_frames) == 0:
            return np.array([], dtype=int)
        
        # Take every Nth beat as downbeat (assumes first beat is downbeat)
        downbeat_indices = np.arange(0, len(beat_frames), time_signature, dtype=int)
        downbeat_frames = beat_frames[downbeat_indices]
        
        # Convert to sample indices
        downbeat_samples = librosa.frames_to_samples(downbeat_frames, hop_length=hop_length)
        
        return downbeat_samples.astype(int)
    
    except Exception as e:
        logger.error(f"Downbeat detection failed: {e}")
        return np.array([], dtype=int)


# =============================================================================
# TIME & BPM CONVERSIONS
# =============================================================================

def beats_to_seconds(n_beats: float, bpm: float) -> float:
    """
    Convert number of beats to duration in seconds.
    
    Args:
        n_beats: Number of beats
        bpm: Tempo in beats per minute
    
    Returns:
        Duration in seconds
    
    Example:
        >>> duration = beats_to_seconds(16, 120)  # 16 beats at 120 BPM
        >>> print(f"{duration:.2f} seconds")  # 8.00 seconds
    """
    if bpm <= 0:
        return 0.0
    return float(n_beats * 60.0 / bpm)


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """
    Convert duration in seconds to number of beats.
    
    Args:
        seconds: Duration in seconds
        bpm: Tempo in beats per minute
    
    Returns:
        Number of beats
    
    Example:
        >>> beats = seconds_to_beats(8.0, 120)  # 8 seconds at 120 BPM
        >>> print(f"{beats:.1f} beats")  # 16.0 beats
    """
    if bpm <= 0:
        return 0.0
    return float(seconds * bpm / 60.0)


def bpm_are_compatible(
    bpm1: float,
    bpm2: float,
    tolerance: float = 3.0,
    check_multiples: bool = True
) -> Tuple[bool, str, float]:
    """
    Check if two BPMs are compatible for mixing/mashup.
    
    Args:
        bpm1: First BPM
        bpm2: Second BPM
        tolerance: Acceptable BPM difference (default ±3 BPM)
        check_multiples: If True, also check half/double time compatibility
    
    Returns:
        (compatible, relationship, ratio)
        - compatible: True if BPMs can be mixed
        - relationship: 'exact', 'close', 'half-time', 'double-time', or 'incompatible'
        - ratio: Stretch ratio needed (bpm2/bpm1)
    
    Example:
        >>> compatible, rel, ratio = bpm_are_compatible(120, 122)
        >>> print(f"{rel}: stretch by {ratio:.3f}x")  # close: stretch by 1.017x
        
        >>> compatible, rel, ratio = bpm_are_compatible(140, 70, check_multiples=True)
        >>> print(rel)  # half-time
    """
    if bpm1 <= 0 or bpm2 <= 0:
        return False, 'incompatible', 0.0
    
    ratio = bpm2 / bpm1
    diff = abs(bpm2 - bpm1)
    
    # Check exact/close match
    if diff < 0.5:
        return True, 'exact', ratio
    elif diff <= tolerance:
        return True, 'close', ratio
    
    # Check half/double time relationships
    if check_multiples:
        # Half-time: bpm2 ≈ bpm1/2
        if abs(bpm2 - bpm1/2) <= tolerance:
            return True, 'half-time', ratio
        
        # Double-time: bpm2 ≈ bpm1*2
        if abs(bpm2 - bpm1*2) <= tolerance:
            return True, 'double-time', ratio
        
        # Reverse checks
        if abs(bpm1 - bpm2/2) <= tolerance:
            return True, 'double-time', ratio
        if abs(bpm1 - bpm2*2) <= tolerance:
            return True, 'half-time', ratio
    
    return False, 'incompatible', ratio


def get_canonical_bpms(bpm: float) -> Tuple[float, float, float]:
    """
    Get canonical BPM variants (half-time, normal, double-time).
    
    Args:
        bpm: Original BPM
    
    Returns:
        (half_bpm, normal_bpm, double_bpm) rounded to 2 decimals
    
    Example:
        >>> half, normal, double = get_canonical_bpms(120)
        >>> print(half, normal, double)  # 60.0, 120.0, 240.0
    
    Note:
        Useful for matching tracks at different tempos (e.g., 140 BPM techno
        can match with 70 BPM hip-hop or 280 BPM drum & bass).
    """
    if bpm <= 0:
        return 0.0, 0.0, 0.0
    
    return (
        round(bpm / 2.0, 2),
        round(bpm, 2),
        round(bpm * 2.0, 2)
    )


def calculate_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
    """
    Calculate time-stretch ratio needed to convert source BPM to target BPM.
    
    Args:
        source_bpm: Original tempo
        target_bpm: Desired tempo
    
    Returns:
        Stretch ratio (target/source). >1 speeds up, <1 slows down
    
    Example:
        >>> ratio = calculate_stretch_ratio(140, 120)
        >>> print(f"Stretch by {ratio:.3f}x")  # Stretch by 0.857x (slow down)
    """
    if source_bpm <= 0 or target_bpm <= 0:
        return 1.0
    return float(target_bpm / source_bpm)


# =============================================================================
# BEAT-SYNCHRONOUS AUDIO EXTRACTION
# =============================================================================

def extract_beat_window(
    y: np.ndarray,
    sr: int,
    beat_frames: np.ndarray,
    start_beat: int,
    end_beat: int,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract audio segment between specified beat positions.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        beat_frames: Frame indices of beats (from detect_beats)
        start_beat: Starting beat index (0-based)
        end_beat: Ending beat index (exclusive)
        hop_length: Hop length used for beat detection
    
    Returns:
        Audio segment from start_beat to end_beat
    
    Example:
        >>> _, beats = detect_beats(audio, 44100)
        >>> # Extract beats 16-32 (one bar in 4/4)
        >>> segment = extract_beat_window(audio, 44100, beats, 16, 32)
    """
    if len(beat_frames) == 0:
        return np.array([])
    
    # Clamp indices
    start_beat = max(0, min(start_beat, len(beat_frames) - 1))
    end_beat = max(start_beat + 1, min(end_beat, len(beat_frames)))
    
    # Convert to sample positions
    start_sample = librosa.frames_to_samples(beat_frames[start_beat], hop_length=hop_length)
    
    if end_beat < len(beat_frames):
        end_sample = librosa.frames_to_samples(beat_frames[end_beat], hop_length=hop_length)
    else:
        # If end_beat is beyond last beat, go to end of audio
        end_sample = len(y)
    
    start_sample = int(start_sample)
    end_sample = int(min(end_sample, len(y)))
    
    return y[start_sample:end_sample]


def create_beat_mask(
    y: np.ndarray,
    sr: int,
    beat_frames: np.ndarray,
    hop_length: int = 512,
    beat_duration: float = 0.1
) -> np.ndarray:
    """
    Create a binary mask highlighting beat positions (useful for beat-sync processing).
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        beat_frames: Frame indices of beats
        hop_length: Hop length used for beat detection
        beat_duration: Duration of each beat marker in seconds
    
    Returns:
        Binary mask array (same length as y) with 1s at beat positions
    
    Example:
        >>> _, beats = detect_beats(audio, 44100)
        >>> mask = create_beat_mask(audio, 44100, beats)
        >>> beat_enhanced = audio * (1 + 0.5 * mask)  # Boost beats by 50%
    """
    mask = np.zeros(len(y), dtype=np.float32)
    
    beat_samples = librosa.frames_to_samples(beat_frames, hop_length=hop_length)
    beat_width = int(beat_duration * sr)
    
    for beat_pos in beat_samples:
        beat_pos = int(beat_pos)
        start = max(0, beat_pos - beat_width // 2)
        end = min(len(mask), beat_pos + beat_width // 2)
        mask[start:end] = 1.0
    
    return mask


# =============================================================================
# SECTION DETECTION (INTRO/OUTRO)
# =============================================================================

def find_intro_by_beats(
    y: np.ndarray,
    sr: int,
    max_duration: float = 30.0,
    hop_length: int = 512
) -> Tuple[int, int]:
    """
    Find intro section with low energy suitable for mixing in.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        max_duration: Maximum intro length in seconds
        hop_length: Hop length for analysis
    
    Returns:
        (start_sample, end_sample) of intro section
    
    Example:
        >>> intro_start, intro_end = find_intro_by_beats(audio, 44100)
        >>> intro = audio[intro_start:intro_end]
    
    Note:
        Looks for sections with gradually increasing energy.
        Returns (0, 0) if no suitable intro found.
    """
    try:
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        if len(rms) == 0:
            return 0, 0
        
        # Smooth RMS
        rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')
        
        # Find where energy starts increasing significantly
        rms_diff = np.diff(rms_smooth)
        
        # Look for sustained increase in first max_duration seconds
        max_frames = int(max_duration * sr / hop_length)
        search_region = rms_diff[:min(max_frames, len(rms_diff))]
        
        if len(search_region) == 0:
            return 0, 0
        
        # Find point where energy ramps up
        threshold = np.percentile(search_region, 75)
        rising_points = np.where(search_region > threshold)[0]
        
        if len(rising_points) > 0:
            intro_end_frame = rising_points[0]
            intro_end_sample = librosa.frames_to_samples(intro_end_frame, hop_length=hop_length)
            return 0, int(intro_end_sample)
        
        # Default: use first 8 seconds or 10% of track
        default_intro = min(int(8.0 * sr), len(y) // 10)
        return 0, default_intro
    
    except Exception as e:
        logger.error(f"Intro detection failed: {e}")
        return 0, 0


def find_outro_by_beats(
    y: np.ndarray,
    sr: int,
    max_duration: float = 30.0,
    hop_length: int = 512
) -> Tuple[int, int]:
    """
    Find outro section with decreasing energy suitable for mixing out.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        max_duration: Maximum outro length in seconds
        hop_length: Hop length for analysis
    
    Returns:
        (start_sample, end_sample) of outro section
    
    Example:
        >>> outro_start, outro_end = find_outro_by_beats(audio, 44100)
        >>> outro = audio[outro_start:outro_end]
    
    Note:
        Looks for sections with gradually decreasing energy.
        Returns (len(y), len(y)) if no suitable outro found.
    """
    try:
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        if len(rms) == 0:
            return len(y), len(y)
        
        # Smooth RMS
        rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')
        
        # Find where energy starts decreasing significantly
        rms_diff = np.diff(rms_smooth)
        
        # Look for sustained decrease in last max_duration seconds
        max_frames = int(max_duration * sr / hop_length)
        search_start = max(0, len(rms_diff) - max_frames)
        search_region = rms_diff[search_start:]
        
        if len(search_region) == 0:
            return len(y), len(y)
        
        # Find point where energy drops
        threshold = np.percentile(search_region, 25)
        falling_points = np.where(search_region < threshold)[0]
        
        if len(falling_points) > 0:
            outro_start_frame = search_start + falling_points[0]
            outro_start_sample = librosa.frames_to_samples(outro_start_frame, hop_length=hop_length)
            return int(outro_start_sample), len(y)
        
        # Default: use last 8 seconds or 10% of track
        default_outro_start = max(0, len(y) - min(int(8.0 * sr), len(y) // 10))
        return default_outro_start, len(y)
    
    except Exception as e:
        logger.error(f"Outro detection failed: {e}")
        return len(y), len(y)


def get_beat_strength(
    y: np.ndarray,
    sr: int,
    beat_frames: np.ndarray,
    hop_length: int = 512
) -> np.ndarray:
    """
    Calculate energy/strength at each beat position.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        beat_frames: Frame indices of beats
        hop_length: Hop length for analysis
    
    Returns:
        Array of strength values (RMS energy) at each beat
    
    Example:
        >>> _, beats = detect_beats(audio, 44100)
        >>> strengths = get_beat_strength(audio, 44100, beats)
        >>> strong_beats = beats[strengths > np.median(strengths)]
    """
    if len(beat_frames) == 0:
        return np.array([])
    
    # Calculate onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Sample at beat positions
    beat_frames_clamped = np.clip(beat_frames, 0, len(onset_env) - 1).astype(int)
    strengths = onset_env[beat_frames_clamped]
    
    return strengths


def estimate_optimal_crossfade_beats(
    y: np.ndarray,
    sr: int,
    bpm: float,
    base_beats: int = 16,
    min_beats: int = 8,
    max_beats: int = 32,
    hop_length: int = 512
) -> int:
    """
    Calculate optimal crossfade length in beats based on audio energy variance.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        bpm: Tempo in BPM
        base_beats: Default crossfade length
        min_beats: Minimum crossfade length
        max_beats: Maximum crossfade length
        hop_length: Hop length for analysis
    
    Returns:
        Optimal crossfade length in beats
    
    Example:
        >>> fade_beats = estimate_optimal_crossfade_beats(audio, 44100, 128)
        >>> fade_duration = beats_to_seconds(fade_beats, 128)
    
    Note:
        High energy/variance = shorter fades (punchy transition)
        Low energy/variance = longer fades (smooth transition)
    """
    if bpm <= 0 or len(y) < sr:
        return base_beats
    
    try:
        # Calculate RMS energy curve
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        if len(rms) < 8:
            return base_beats
        
        # Focus on last portion (outro region)
        tail_length = min(len(rms), 64)
        tail = rms[-tail_length:]
        
        # Calculate variance
        variance = float(np.var(tail))
        
        # Normalize variance using robust scaling
        p90 = float(np.percentile(rms, 90)) + 1e-9
        normalized_variance = np.clip(variance / (p90**2 + 1e-9), 0.0, 1.0)
        
        # High variance = shorter fade, low variance = longer fade
        fade_adjustment = int(normalized_variance * (base_beats - min_beats))
        optimal_beats = base_beats - fade_adjustment
        
        return int(np.clip(optimal_beats, min_beats, max_beats))
    
    except Exception as e:
        logger.warning(f"Crossfade calculation failed: {e}")
        return base_beats


# =============================================================================
# ADVANCED TEMPO OPERATIONS
# =============================================================================

def quantize_to_beat_grid(
    y: np.ndarray,
    sr: int,
    target_bpm: float,
    bars: int = 4,
    time_signature: int = 4
) -> np.ndarray:
    """
    Quantize audio to exact beat grid (useful for loop-based mashups).
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        target_bpm: Target tempo for grid
        bars: Number of bars to fit audio into
        time_signature: Beats per bar
    
    Returns:
        Audio quantized to exact bar length
    
    Example:
        >>> # Force 4-bar loop at 128 BPM
        >>> loop = quantize_to_beat_grid(audio, 44100, 128, bars=4)
    
    Note:
        This stretches audio to fit exactly N bars, useful for creating
        perfect loops or aligning samples to a grid.
    """
    if target_bpm <= 0 or bars <= 0:
        return y
    
    # Calculate target duration
    total_beats = bars * time_signature
    target_duration = beats_to_seconds(total_beats, target_bpm)
    target_samples = int(target_duration * sr)
    
    if target_samples <= 0 or len(y) == 0:
        return y
    
    # Calculate stretch ratio
    ratio = len(y) / target_samples
    
    try:
        # Time stretch to exact length
        y_quantized = librosa.effects.time_stretch(y, rate=ratio)
        
        # Ensure exact length (trim or pad)
        if len(y_quantized) > target_samples:
            y_quantized = y_quantized[:target_samples]
        elif len(y_quantized) < target_samples:
            padding = target_samples - len(y_quantized)
            y_quantized = np.pad(y_quantized, (0, padding), mode='constant')
        
        return y_quantized
    
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return y


def align_beats(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align two audio tracks by their beats (synchronize downbeats).
    
    Args:
        y1: First audio track (reference)
        y2: Second audio track (to be aligned)
        sr: Sample rate
        hop_length: Hop length for beat detection
    
    Returns:
        (y1_aligned, y2_aligned, offset_seconds)
        - y1_aligned: First track (possibly trimmed)
        - y2_aligned: Second track with offset applied
        - offset_seconds: Time shift applied to y2
    
    Example:
        >>> track1_aligned, track2_aligned, offset = align_beats(track1, track2, 44100)
        >>> # Now both tracks have synchronized beats
    
    Note:
        Useful for starting mashups with beats aligned.
        Pads with silence if needed.
    """
    try:
        # Detect beats in both tracks
        _, beats1 = detect_beats(y1, sr, hop_length)
        _, beats2 = detect_beats(y2, sr, hop_length)
        
        if len(beats1) == 0 or len(beats2) == 0:
            logger.warning("Could not detect beats for alignment")
            return y1, y2, 0.0
        
        # Get first beat positions
        first_beat1 = librosa.frames_to_samples(beats1[0], hop_length=hop_length)
        first_beat2 = librosa.frames_to_samples(beats2[0], hop_length=hop_length)
        
        # Calculate offset needed
        offset_samples = int(first_beat1 - first_beat2)
        offset_seconds = offset_samples / sr
        
        if offset_samples > 0:
            # y2 needs to be shifted right (padded at start)
            y2_aligned = np.pad(y2, (offset_samples, 0), mode='constant')
        elif offset_samples < 0:
            # y2 needs to be shifted left (trimmed at start)
            y2_aligned = y2[-offset_samples:]
        else:
            y2_aligned = y2
        
        # Make both tracks same length
        min_length = min(len(y1), len(y2_aligned))
        y1_aligned = y1[:min_length]
        y2_aligned = y2_aligned[:min_length]
        
        return y1_aligned, y2_aligned, offset_seconds
    
    except Exception as e:
        logger.error(f"Beat alignment failed: {e}")
        return y1, y2, 0.0


def detect_tempo_changes(
    y: np.ndarray,
    sr: int,
    window_duration: float = 10.0,
    hop_duration: float = 5.0
) -> List[Dict[str, float]]:
    """
    Detect tempo changes throughout a track (for variable-tempo music).
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        window_duration: Analysis window size in seconds
        hop_duration: Hop between windows in seconds
    
    Returns:
        List of tempo segments: [{'start': seconds, 'end': seconds, 'bpm': float}, ...]
    
    Example:
        >>> tempo_map = detect_tempo_changes(audio, 44100)
        >>> for segment in tempo_map:
        >>>     print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['bpm']:.1f} BPM")
    
    Note:
        Useful for classical music, live recordings, or tracks with tempo automation.
        Most electronic music has constant tempo and will return single segment.
    """
    segments = []
    
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    if window_samples <= 0 or len(y) < window_samples:
        # Track too short, analyze whole thing
        bpm, _ = detect_bpm(y, sr)
        return [{'start': 0.0, 'end': len(y) / sr, 'bpm': bpm}]
    
    pos = 0
    while pos < len(y):
        end_pos = min(pos + window_samples, len(y))
        window = y[pos:end_pos]
        
        if len(window) < sr:  # Skip very short windows
            break
        
        bpm, _ = detect_bpm(window, sr)
        
        segment = {
            'start': pos / sr,
            'end': end_pos / sr,
            'bpm': bpm
        }
        segments.append(segment)
        
        pos += hop_samples
    
    # Merge adjacent segments with similar BPM
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if abs(seg['bpm'] - prev['bpm']) < 3.0:
                # Similar tempo, merge
                prev['end'] = seg['end']
                prev['bpm'] = (prev['bpm'] + seg['bpm']) / 2  # Average
            else:
                merged.append(seg)
        return merged
    
    return segments


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _rms_curve(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Internal helper: Calculate RMS energy curve.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length for framing
    
    Returns:
        RMS energy per frame
    """
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True)[0]
    return rms.astype(np.float32)


if __name__ == '__main__':
    # Quick test
    print("BPM Manipulations Module")
    print(f"Available functions: {len(__all__)}")
    print(f"Core functions: detect_bpm, time_stretch_to_bpm, detect_beats, etc.")
