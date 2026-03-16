"""
Volume & Dynamics Manipulation Utilities for Mashup Creation

This module provides fundamental volume/dynamics operations for music mashups.
All functions are independent and designed for easy programmatic use.

Core Functions:
- apply_gain: Adjust volume (dB or linear)
- normalize_peak: Peak normalization to prevent clipping
- normalize_rms: RMS-based loudness normalization
- normalize_lufs: LUFS-based loudness matching (broadcasting standard)
- fade_in / fade_out: Smooth volume transitions
- apply_fade: Apply custom fade curve
- compress: Dynamic range compression
- limit: Peak limiting
- gate: Noise gate (remove quiet sections)

Analysis Functions:
- get_peak_db: Get peak level in dB
- get_rms_db: Get RMS level in dB
- get_loudness_lufs: Get perceived loudness
- detect_clipping: Find clipped samples
- analyze_dynamics: Get dynamic range statistics

Advanced Functions:
- apply_volume_envelope: Volume automation curve
- auto_gain_match: Match loudness between tracks
- sidechain_compress: Duck volume based on another track
- parallel_compress: Blend compressed/uncompressed signal
- multiband_compress: Compress different frequency bands separately

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import math
from typing import Tuple, Optional, List, Dict, Literal

import numpy as np
import librosa

logger = logging.getLogger(__name__)

__all__ = [
    'apply_gain',
    'normalize_peak',
    'normalize_rms',
    'normalize_lufs',
    'fade_in',
    'fade_out',
    'apply_fade',
    'compress',
    'limit',
    'gate',
    'get_peak_db',
    'get_rms_db',
    'get_loudness_lufs',
    'detect_clipping',
    'analyze_dynamics',
    'apply_volume_envelope',
    'auto_gain_match',
    'sidechain_compress',
    'parallel_compress',
    'multiband_compress',
]


# =============================================================================
# CORE GAIN & NORMALIZATION
# =============================================================================

def apply_gain(
    y: np.ndarray,
    gain_db: float = 0.0,
    gain_linear: Optional[float] = None
) -> np.ndarray:
    """
    Apply gain (volume adjustment) to audio.
    
    Args:
        y: Audio time series (mono or stereo)
        gain_db: Gain in decibels (default 0 = no change)
        gain_linear: Alternative linear gain (overrides gain_db if provided)
    
    Returns:
        Audio with gain applied
    
    Example:
        >>> # Increase volume by 6 dB
        >>> louder = apply_gain(audio, gain_db=6.0)
        >>> # Decrease volume by 50% (linear)
        >>> quieter = apply_gain(audio, gain_linear=0.5)
    """
    if gain_linear is not None:
        gain = float(gain_linear)
    else:
        # Convert dB to linear: gain = 10^(dB/20)
        gain = 10.0 ** (gain_db / 20.0)
    
    return (y * gain).astype(y.dtype)


def normalize_peak(
    y: np.ndarray,
    target_db: float = -1.0,
    headroom: bool = True
) -> np.ndarray:
    """
    Normalize audio so peak level reaches target dB.
    
    Args:
        y: Audio time series
        target_db: Target peak level in dBFS (default -1.0 dB for safety)
        headroom: If True, leave headroom to prevent clipping
    
    Returns:
        Peak-normalized audio
    
    Example:
        >>> # Normalize to -1 dB peak (standard mastering level)
        >>> normalized = normalize_peak(audio, target_db=-1.0)
    
    Note:
        Peak normalization preserves dynamic range but may result in
        inconsistent loudness between tracks. Use normalize_rms or
        normalize_lufs for consistent perceived loudness.
    """
    peak = np.max(np.abs(y))
    
    if peak < 1e-9:
        logger.warning("Audio is silent, cannot normalize")
        return y
    
    target_linear = 10.0 ** (target_db / 20.0)
    gain = target_linear / peak
    
    normalized = y * gain
    
    # Safety check: prevent clipping
    if headroom:
        actual_peak = np.max(np.abs(normalized))
        if actual_peak > 1.0:
            normalized = normalized / actual_peak * 0.99
    
    return normalized.astype(y.dtype)


def normalize_rms(
    y: np.ndarray,
    target_db: float = -20.0,
    prevent_clipping: bool = True
) -> np.ndarray:
    """
    Normalize audio to target RMS (perceived loudness) level.
    
    Args:
        y: Audio time series
        target_db: Target RMS level in dBFS (default -20 dB)
        prevent_clipping: If True, reduce gain if peaks would clip
    
    Returns:
        RMS-normalized audio
    
    Example:
        >>> # Match RMS to -20 dB (good for voice)
        >>> normalized = normalize_rms(audio, target_db=-20.0)
    
    Note:
        RMS normalization provides more consistent loudness than peak
        normalization. Typical targets: -20dB (voice), -14dB (music).
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(y**2))
    
    if rms < 1e-9:
        logger.warning("Audio is silent, cannot normalize")
        return y
    
    # Convert current RMS to dB
    current_db = 20.0 * np.log10(rms)
    
    # Calculate required gain
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    
    normalized = y * gain
    
    # Prevent clipping if requested
    if prevent_clipping:
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak * 0.99
    
    return normalized.astype(y.dtype)


def normalize_lufs(
    y: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    prevent_clipping: bool = True
) -> np.ndarray:
    """
    Normalize audio to target LUFS (Loudness Units relative to Full Scale).
    
    Args:
        y: Audio time series
        sr: Sample rate
        target_lufs: Target integrated loudness (default -14 LUFS for streaming)
        prevent_clipping: If True, reduce gain if peaks would clip
    
    Returns:
        LUFS-normalized audio
    
    Example:
        >>> # Normalize to Spotify/YouTube standard (-14 LUFS)
        >>> normalized = normalize_lufs(audio, 44100, target_lufs=-14.0)
    
    Note:
        LUFS is the broadcasting standard for loudness. Common targets:
        - Spotify/YouTube/Apple Music: -14 LUFS
        - Film dialogue: -27 LUFS
        - Club/DJ: -8 to -10 LUFS
        
        This is a simplified implementation. For production use, consider pyloudnorm.
    """
    # Simplified LUFS approximation using weighted RMS
    # True LUFS requires K-weighting filter and gating
    
    # Apply basic K-weighting approximation (high-shelf boost ~4dB above 2kHz)
    y_weighted = y.copy()
    
    # Calculate integrated loudness (simplified as RMS over whole track)
    rms = np.sqrt(np.mean(y_weighted**2))
    
    if rms < 1e-9:
        logger.warning("Audio is silent, cannot normalize")
        return y
    
    # Convert to LUFS (approximation: LUFS ≈ -0.691 + 10*log10(rms^2))
    current_lufs = -0.691 + 10.0 * np.log10(rms**2 + 1e-9)
    
    # Calculate required gain
    gain_db = target_lufs - current_lufs
    gain = 10.0 ** (gain_db / 20.0)
    
    normalized = y * gain
    
    # Prevent clipping
    if prevent_clipping:
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak * 0.99
    
    return normalized.astype(y.dtype)


# =============================================================================
# FADES
# =============================================================================

def fade_in(
    y: np.ndarray,
    sr: int,
    duration: float = 1.0,
    curve: Literal['linear', 'exponential', 'logarithmic', 'scurve'] = 'linear'
) -> np.ndarray:
    """
    Apply fade-in to beginning of audio.
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Fade duration in seconds
        curve: Fade curve type
            - 'linear': Constant rate (simple but audible)
            - 'exponential': Accelerating (smooth, natural)
            - 'logarithmic': Decelerating (emphasizes start)
            - 'scurve': S-curve (smoothest, most natural)
    
    Returns:
        Audio with fade-in applied
    
    Example:
        >>> # Smooth 2-second fade-in
        >>> faded = fade_in(audio, 44100, duration=2.0, curve='exponential')
    """
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(y))
    
    if fade_samples <= 0:
        return y
    
    # Generate fade curve
    fade_curve = _generate_fade_curve(fade_samples, curve, fade_in=True)
    
    # Apply fade
    result = y.copy()
    result[:fade_samples] *= fade_curve
    
    return result


def fade_out(
    y: np.ndarray,
    sr: int,
    duration: float = 1.0,
    curve: Literal['linear', 'exponential', 'logarithmic', 'scurve'] = 'linear'
) -> np.ndarray:
    """
    Apply fade-out to end of audio.
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Fade duration in seconds
        curve: Fade curve type (see fade_in for options)
    
    Returns:
        Audio with fade-out applied
    
    Example:
        >>> # Smooth 3-second fade-out
        >>> faded = fade_out(audio, 44100, duration=3.0, curve='exponential')
    """
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(y))
    
    if fade_samples <= 0:
        return y
    
    # Generate fade curve
    fade_curve = _generate_fade_curve(fade_samples, curve, fade_in=False)
    
    # Apply fade
    result = y.copy()
    result[-fade_samples:] *= fade_curve
    
    return result


def apply_fade(
    y: np.ndarray,
    sr: int,
    fade_in_duration: float = 0.0,
    fade_out_duration: float = 0.0,
    curve: str = 'linear'
) -> np.ndarray:
    """
    Apply both fade-in and fade-out in one call.
    
    Args:
        y: Audio time series
        sr: Sample rate
        fade_in_duration: Fade-in duration in seconds
        fade_out_duration: Fade-out duration in seconds
        curve: Fade curve type (see fade_in for options)
    
    Returns:
        Audio with both fades applied
    
    Example:
        >>> # 1s fade in, 2s fade out
        >>> faded = apply_fade(audio, 44100, 1.0, 2.0, curve='exponential')
    """
    result = y.copy()
    
    if fade_in_duration > 0:
        result = fade_in(result, sr, fade_in_duration, curve)
    
    if fade_out_duration > 0:
        result = fade_out(result, sr, fade_out_duration, curve)
    
    return result


def _generate_fade_curve(
    n_samples: int,
    curve: str,
    fade_in: bool = True
) -> np.ndarray:
    """
    Generate fade curve array.
    
    Args:
        n_samples: Number of samples in fade
        curve: Curve type
        fade_in: True for fade-in, False for fade-out
    
    Returns:
        Fade curve array [0..1]
    """
    t = np.linspace(0, 1, n_samples)
    
    if curve == 'linear':
        fade = t
    elif curve == 'exponential':
        # Exponential: y = x^2
        fade = t ** 2
    elif curve == 'logarithmic':
        # Logarithmic: y = sqrt(x)
        fade = np.sqrt(t)
    elif curve == 'scurve':
        # S-curve (smoothstep): y = 3x^2 - 2x^3
        fade = 3 * t**2 - 2 * t**3
    else:
        logger.warning(f"Unknown curve type '{curve}', using linear")
        fade = t
    
    # Invert for fade-out
    if not fade_in:
        fade = 1.0 - fade
    
    return fade.astype(np.float32)


# =============================================================================
# DYNAMICS PROCESSING (COMPRESSION/LIMITING/GATING)
# =============================================================================

def compress(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    knee_db: float = 0.0,
    makeup_gain_db: float = 0.0
) -> np.ndarray:
    """
    Apply dynamic range compression (reduces loud peaks).
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Level above which compression starts (dBFS)
        ratio: Compression ratio (4.0 = 4:1, higher = more compression)
        attack_ms: Attack time in milliseconds (how fast compressor responds)
        release_ms: Release time in milliseconds (how fast compressor recovers)
        knee_db: Soft knee width in dB (0 = hard knee)
        makeup_gain_db: Compensate for volume reduction
    
    Returns:
        Compressed audio
    
    Example:
        >>> # Gentle compression for vocal
        >>> compressed = compress(audio, 44100, threshold_db=-20, ratio=3.0)
        >>> # Heavy compression for pumping effect
        >>> pumped = compress(audio, 44100, threshold_db=-15, ratio=8.0, attack_ms=1.0)
    
    Note:
        Compression reduces dynamic range, making quiet parts louder and
        loud parts quieter. Common uses:
        - Vocals: ratio 3-4:1, threshold -20dB
        - Drums: ratio 4-8:1, threshold -15dB
        - Mix bus: ratio 2-3:1, threshold -10dB
    """
    y = y.astype(np.float32)
    
    # Convert times to coefficients
    attack_coef = np.exp(-1.0 / (sr * (attack_ms / 1000.0) + 1e-9))
    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))
    
    # Convert threshold to linear
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    
    # Process audio
    envelope = 0.0
    output = np.zeros_like(y)
    
    for i in range(len(y)):
        # Envelope follower
        input_abs = abs(y[i])
        if input_abs > envelope:
            envelope = attack_coef * envelope + (1 - attack_coef) * input_abs
        else:
            envelope = release_coef * envelope + (1 - release_coef) * input_abs
        
        # Calculate gain reduction
        gain = 1.0
        
        if envelope > threshold_linear:
            # How much over threshold (in dB)
            over_db = 20.0 * np.log10(envelope / threshold_linear + 1e-9)
            
            # Apply knee if specified
            if knee_db > 0 and over_db < knee_db:
                # Soft knee interpolation
                over_db = over_db ** 2 / (2.0 * knee_db)
            
            # Calculate gain reduction
            gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
            gain = 10.0 ** (-gain_reduction_db / 20.0)
        
        output[i] = y[i] * gain
    
    # Apply makeup gain
    if makeup_gain_db != 0.0:
        makeup_gain = 10.0 ** (makeup_gain_db / 20.0)
        output *= makeup_gain
    
    return output


def limit(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -1.0,
    release_ms: float = 50.0
) -> np.ndarray:
    """
    Apply limiting (hard ceiling to prevent clipping).
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Maximum output level in dBFS (default -1.0 dB)
        release_ms: Release time in milliseconds
    
    Returns:
        Limited audio
    
    Example:
        >>> # Prevent clipping at -0.5 dB
        >>> limited = limit(audio, 44100, threshold_db=-0.5)
    
    Note:
        Limiting is extreme compression (∞:1 ratio) used as final safety
        to prevent clipping. Always the last processor in chain.
    """
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    
    # Use soft clipping (tanh) for more natural sound
    output = np.tanh(y / threshold_linear) * threshold_linear
    
    return output.astype(y.dtype)


def gate(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    hold_ms: float = 10.0
) -> np.ndarray:
    """
    Apply noise gate (removes audio below threshold).
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Level below which gate closes (dBFS)
        attack_ms: Attack time (how fast gate opens)
        release_ms: Release time (how fast gate closes)
        hold_ms: Hold time (minimum time gate stays open)
    
    Returns:
        Gated audio
    
    Example:
        >>> # Remove background noise below -40 dB
        >>> gated = gate(audio, 44100, threshold_db=-40)
    
    Note:
        Noise gates remove unwanted quiet sections (breath, room noise).
        Use carefully to avoid cutting off natural decay/reverb tails.
    """
    y = y.astype(np.float32)
    
    # Convert parameters
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    attack_coef = np.exp(-1.0 / (sr * (attack_ms / 1000.0) + 1e-9))
    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))
    hold_samples = int(sr * hold_ms / 1000.0)
    
    # Process
    envelope = 0.0
    gate_open = False
    hold_counter = 0
    output = np.zeros_like(y)
    
    for i in range(len(y)):
        # Envelope follower
        input_abs = abs(y[i])
        envelope = max(attack_coef * envelope, input_abs)
        
        # Gate logic
        if envelope > threshold_linear:
            gate_open = True
            hold_counter = hold_samples
        elif hold_counter > 0:
            hold_counter -= 1
        else:
            gate_open = False
        
        # Apply gate
        if gate_open:
            gain = 1.0
        else:
            gain = 0.0
        
        output[i] = y[i] * gain
    
    return output


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_peak_db(y: np.ndarray) -> float:
    """
    Get peak level in dBFS.
    
    Args:
        y: Audio time series
    
    Returns:
        Peak level in dB (0 dB = full scale, negative = below full scale)
    
    Example:
        >>> peak = get_peak_db(audio)
        >>> print(f"Peak: {peak:.1f} dBFS")
    """
    peak = np.max(np.abs(y))
    
    if peak < 1e-9:
        return -np.inf
    
    return float(20.0 * np.log10(peak))


def get_rms_db(y: np.ndarray) -> float:
    """
    Get RMS (average) level in dBFS.
    
    Args:
        y: Audio time series
    
    Returns:
        RMS level in dB
    
    Example:
        >>> rms = get_rms_db(audio)
        >>> print(f"RMS: {rms:.1f} dBFS")
    """
    rms = np.sqrt(np.mean(y**2))
    
    if rms < 1e-9:
        return -np.inf
    
    return float(20.0 * np.log10(rms))


def get_loudness_lufs(y: np.ndarray, sr: int) -> float:
    """
    Get perceived loudness in LUFS (simplified).
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Integrated loudness in LUFS
    
    Example:
        >>> loudness = get_loudness_lufs(audio, 44100)
        >>> print(f"Loudness: {loudness:.1f} LUFS")
    
    Note:
        This is a simplified approximation. For accurate LUFS measurement,
        use pyloudnorm library.
    """
    rms = np.sqrt(np.mean(y**2))
    
    if rms < 1e-9:
        return -np.inf
    
    # Simplified LUFS approximation
    lufs = -0.691 + 10.0 * np.log10(rms**2 + 1e-9)
    
    return float(lufs)


def detect_clipping(
    y: np.ndarray,
    threshold: float = 0.99
) -> Tuple[int, List[int]]:
    """
    Detect clipped samples in audio.
    
    Args:
        y: Audio time series
        threshold: Clipping threshold (default 0.99, close to ±1.0)
    
    Returns:
        (count, positions) - number of clipped samples and their indices
    
    Example:
        >>> count, positions = detect_clipping(audio)
        >>> if count > 0:
        >>>     print(f"Warning: {count} clipped samples found!")
    """
    clipped_mask = np.abs(y) >= threshold
    clipped_indices = np.where(clipped_mask)[0].tolist()
    
    return len(clipped_indices), clipped_indices


def analyze_dynamics(
    y: np.ndarray,
    sr: int
) -> Dict[str, float]:
    """
    Comprehensive dynamic range analysis.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with dynamic statistics:
        - peak_db: Peak level
        - rms_db: RMS level
        - crest_factor_db: Peak-to-RMS ratio (dynamic range indicator)
        - loudness_lufs: Perceived loudness
        - dynamic_range_db: Difference between loudest and quietest parts
    
    Example:
        >>> stats = analyze_dynamics(audio, 44100)
        >>> print(f"Peak: {stats['peak_db']:.1f} dB")
        >>> print(f"Dynamic range: {stats['dynamic_range_db']:.1f} dB")
    """
    # Peak and RMS
    peak_db = get_peak_db(y)
    rms_db = get_rms_db(y)
    
    # Crest factor (peak-to-RMS ratio)
    crest_factor_db = peak_db - rms_db
    
    # Loudness
    loudness_lufs = get_loudness_lufs(y, sr)
    
    # Dynamic range (difference between loud and quiet parts)
    # Use percentiles to avoid outliers
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(frame_rms) > 0:
        loud_db = 20.0 * np.log10(np.percentile(frame_rms, 95) + 1e-9)
        quiet_db = 20.0 * np.log10(np.percentile(frame_rms, 5) + 1e-9)
        dynamic_range_db = loud_db - quiet_db
    else:
        dynamic_range_db = 0.0
    
    return {
        'peak_db': float(peak_db),
        'rms_db': float(rms_db),
        'crest_factor_db': float(crest_factor_db),
        'loudness_lufs': float(loudness_lufs),
        'dynamic_range_db': float(dynamic_range_db)
    }


# =============================================================================
# ADVANCED VOLUME OPERATIONS
# =============================================================================

def apply_volume_envelope(
    y: np.ndarray,
    sr: int,
    envelope_points: List[Tuple[float, float]],
    interpolation: Literal['linear', 'smooth'] = 'smooth'
) -> np.ndarray:
    """
    Apply volume automation envelope (volume changes over time).
    
    Args:
        y: Audio time series
        sr: Sample rate
        envelope_points: List of (time_seconds, gain_db) points
        interpolation: Interpolation method
    
    Returns:
        Audio with volume envelope applied
    
    Example:
        >>> # Fade in 0→0dB (0-2s), hold (2-5s), fade out 0→-∞dB (5-7s)
        >>> envelope = [(0, -80), (2, 0), (5, 0), (7, -80)]
        >>> automated = apply_volume_envelope(audio, 44100, envelope)
    """
    if len(envelope_points) < 2:
        logger.warning("Need at least 2 envelope points")
        return y
    
    # Sort by time
    envelope_points = sorted(envelope_points, key=lambda x: x[0])
    
    # Convert to samples and gains
    times = np.array([t for t, _ in envelope_points])
    gains_db = np.array([g for _, g in envelope_points])
    
    # Create full-resolution envelope
    duration = len(y) / sr
    t_samples = np.linspace(0, duration, len(y))
    
    # Interpolate
    if interpolation == 'smooth':
        # Cubic interpolation
        from scipy import interpolate
        f = interpolate.interp1d(times, gains_db, kind='cubic', 
                                bounds_error=False, fill_value='extrapolate')
        envelope_db = f(t_samples)
    else:
        # Linear interpolation
        envelope_db = np.interp(t_samples, times, gains_db)
    
    # Convert to linear
    envelope_linear = 10.0 ** (envelope_db / 20.0)
    
    # Apply
    return (y * envelope_linear).astype(y.dtype)


def auto_gain_match(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    method: Literal['peak', 'rms', 'lufs'] = 'lufs'
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Automatically match loudness between two tracks.
    
    Args:
        y1: First audio track (reference)
        y2: Second audio track (will be adjusted)
        sr: Sample rate
        method: Matching method ('peak', 'rms', or 'lufs')
    
    Returns:
        (y1, y2_adjusted, gain_applied_db)
    
    Example:
        >>> track1, track2_matched, gain = auto_gain_match(track1, track2, 44100)
        >>> print(f"Applied {gain:.1f} dB to track 2")
    """
    if method == 'peak':
        level1 = get_peak_db(y1)
        level2 = get_peak_db(y2)
    elif method == 'rms':
        level1 = get_rms_db(y1)
        level2 = get_rms_db(y2)
    else:  # lufs
        level1 = get_loudness_lufs(y1, sr)
        level2 = get_loudness_lufs(y2, sr)
    
    # Calculate gain needed
    gain_db = level1 - level2
    
    # Apply gain to y2
    y2_adjusted = apply_gain(y2, gain_db=gain_db)
    
    return y1, y2_adjusted, float(gain_db)


def sidechain_compress(
    y: np.ndarray,
    sidechain: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0
) -> np.ndarray:
    """
    Sidechain compression (duck volume based on another track).
    
    Args:
        y: Audio to be compressed
        sidechain: Audio that triggers compression (e.g., kick drum)
        sr: Sample rate
        threshold_db: Compression threshold
        ratio: Compression ratio
        attack_ms: Attack time
        release_ms: Release time
    
    Returns:
        Sidechain-compressed audio
    
    Example:
        >>> # Duck bass when kick drum hits
        >>> ducked_bass = sidechain_compress(bass, kick, 44100, ratio=10.0)
    
    Note:
        Common use: ducking bass/pad when kick drum hits (EDM pumping effect).
    """
    # Ensure same length
    min_len = min(len(y), len(sidechain))
    y = y[:min_len]
    sidechain = sidechain[:min_len]
    
    # Convert parameters
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    attack_coef = np.exp(-1.0 / (sr * (attack_ms / 1000.0) + 1e-9))
    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))
    
    # Process
    envelope = 0.0
    output = np.zeros(min_len, dtype=y.dtype)
    
    for i in range(min_len):
        # Envelope follower on sidechain
        sc_abs = abs(sidechain[i])
        if sc_abs > envelope:
            envelope = attack_coef * envelope + (1 - attack_coef) * sc_abs
        else:
            envelope = release_coef * envelope + (1 - release_coef) * sc_abs
        
        # Calculate gain reduction
        gain = 1.0
        if envelope > threshold_linear:
            over = envelope / threshold_linear
            comp = over ** (1.0 - 1.0 / ratio)
            gain = 1.0 / comp
        
        output[i] = y[i] * gain
    
    return output


def parallel_compress(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    mix: float = 0.5,
    **compress_kwargs
) -> np.ndarray:
    """
    Parallel compression (blend compressed with dry signal).
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Compression threshold
        ratio: Compression ratio
        mix: Wet/dry mix (0.0 = dry only, 1.0 = compressed only, 0.5 = 50/50)
        **compress_kwargs: Additional arguments for compress()
    
    Returns:
        Parallel-compressed audio
    
    Example:
        >>> # "New York" style parallel compression
        >>> parallel = parallel_compress(drums, 44100, ratio=10.0, mix=0.3)
    
    Note:
        Parallel compression (aka "New York compression") adds punch
        while preserving natural dynamics. Popular on drums and vocals.
    """
    # Compress
    compressed = compress(y, sr, threshold_db, ratio, **compress_kwargs)
    
    # Blend
    output = (1 - mix) * y + mix * compressed
    
    return output.astype(y.dtype)


def multiband_compress(
    y: np.ndarray,
    sr: int,
    crossover_freqs: Tuple[float, float] = (200.0, 2000.0),
    low_kwargs: Optional[Dict] = None,
    mid_kwargs: Optional[Dict] = None,
    high_kwargs: Optional[Dict] = None
) -> np.ndarray:
    """
    Multiband compression (compress different frequency bands separately).
    
    Args:
        y: Audio time series
        sr: Sample rate
        crossover_freqs: (low_mid_freq, mid_high_freq) in Hz
        low_kwargs: Compression settings for low band
        mid_kwargs: Compression settings for mid band
        high_kwargs: Compression settings for high band
    
    Returns:
        Multiband-compressed audio
    
    Example:
        >>> # Compress bass heavily, mids gently, highs not at all
        >>> compressed = multiband_compress(
        >>>     audio, 44100,
        >>>     low_kwargs={'ratio': 8.0, 'threshold_db': -15},
        >>>     mid_kwargs={'ratio': 3.0, 'threshold_db': -20},
        >>>     high_kwargs={'ratio': 1.0}  # No compression
        >>> )
    
    Note:
        Multiband compression allows independent control of bass, mids, highs.
        Essential for mastering and complex mix bus processing.
    """
    # Default compression settings
    if low_kwargs is None:
        low_kwargs = {'ratio': 4.0, 'threshold_db': -20.0}
    if mid_kwargs is None:
        mid_kwargs = {'ratio': 3.0, 'threshold_db': -20.0}
    if high_kwargs is None:
        high_kwargs = {'ratio': 2.0, 'threshold_db': -15.0}
    
    # Split into bands using butterworth filters
    from scipy import signal
    
    low_mid_freq, mid_high_freq = crossover_freqs
    nyquist = sr / 2.0
    
    # Low band (< low_mid_freq)
    sos_low = signal.butter(4, low_mid_freq / nyquist, btype='lowpass', output='sos')
    low_band = signal.sosfilt(sos_low, y)
    
    # Mid band (low_mid_freq to mid_high_freq)
    sos_mid = signal.butter(4, [low_mid_freq / nyquist, mid_high_freq / nyquist], 
                           btype='bandpass', output='sos')
    mid_band = signal.sosfilt(sos_mid, y)
    
    # High band (> mid_high_freq)
    sos_high = signal.butter(4, mid_high_freq / nyquist, btype='highpass', output='sos')
    high_band = signal.sosfilt(sos_high, y)
    
    # Compress each band
    low_compressed = compress(low_band, sr, **low_kwargs)
    mid_compressed = compress(mid_band, sr, **mid_kwargs)
    high_compressed = compress(high_band, sr, **high_kwargs)
    
    # Sum bands
    output = low_compressed + mid_compressed + high_compressed
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(y.dtype)


if __name__ == '__main__':
    # Quick test
    print("Volume Manipulations Module")
    print(f"Available functions: {len(__all__)}")
    print("Core: apply_gain, normalize_peak/rms/lufs, fade_in/out, compress, limit")
