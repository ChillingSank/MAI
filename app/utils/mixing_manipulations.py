"""
Mixing & Blending Utilities for Mashup Creation

This module provides fundamental mixing operations to combine tracks.
All functions are independent and designed for easy programmatic use.

Core Functions:
- crossfade_linear: Simple linear crossfade
- crossfade_equal_power: Perceptually smooth crossfade (standard)
- crossfade_exponential: Exponential curve crossfade
- crossfade_scurve: S-curve crossfade (smoothest)
- crossfade_frequency: Frequency-split crossfade (swap bass/keep highs)

EQ/Filter Functions:
- eq_highpass: Remove frequencies below cutoff
- eq_lowpass: Remove frequencies above cutoff
- eq_bandpass: Keep only frequencies in range
- eq_bandstop: Remove frequencies in range (notch)
- eq_parametric: Boost/cut at specific frequency
- eq_shelf_high: Boost/cut high frequencies
- eq_shelf_low: Boost/cut low frequencies

Stereo Functions:
- pan: Position audio in stereo field (left/right)
- stereo_width: Adjust stereo width (mono→wide)
- mid_side_decode: Convert mid/side to left/right
- mid_side_encode: Convert left/right to mid/side
- mid_side_process: Process mid and side separately

Advanced Mixing:
- blend_tracks: Mix multiple tracks with gain control
- harmonic_crossfade: Duck mid frequencies during transition
- split_frequency_bands: Split into low/mid/high bands
- filter_sweep: Animated filter sweep effect

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import math
from typing import Tuple, Optional, List, Dict, Literal, Callable

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

__all__ = [
    'crossfade_linear',
    'crossfade_equal_power',
    'crossfade_exponential',
    'crossfade_scurve',
    'crossfade_frequency',
    'eq_highpass',
    'eq_lowpass',
    'eq_bandpass',
    'eq_bandstop',
    'eq_parametric',
    'eq_shelf_high',
    'eq_shelf_low',
    'pan',
    'stereo_width',
    'mid_side_encode',
    'mid_side_decode',
    'mid_side_process',
    'blend_tracks',
    'harmonic_crossfade',
    'split_frequency_bands',
    'filter_sweep',
]


# =============================================================================
# CROSSFADE FUNCTIONS
# =============================================================================

def crossfade_linear(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0
) -> np.ndarray:
    """
    Linear crossfade between two tracks (simple but audible dip in middle).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> mixed = crossfade_linear(track1, track2, 44100, duration=4.0)
    
    Note:
        Linear crossfades have a 3dB dip in the middle. For smooth transitions,
        use crossfade_equal_power() instead.
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks at crossfade point
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # Linear fade curves
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    gain_a = 1.0 - t  # Fade out
    gain_b = t        # Fade in
    
    # Apply crossfade
    overlap = a_tail * gain_a + b_head * gain_b
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def crossfade_equal_power(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0
) -> np.ndarray:
    """
    Equal-power crossfade (constant perceived loudness - RECOMMENDED).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> mixed = crossfade_equal_power(track1, track2, 44100, duration=4.0)
    
    Note:
        This is the standard crossfade used in professional audio.
        Uses sine/cosine curves to maintain constant power (loudness).
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # Equal-power curves (sine/cosine)
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    gain_a = np.cos(t * math.pi / 2.0) ** 2  # Cosine squared
    gain_b = np.sin(t * math.pi / 2.0) ** 2  # Sine squared
    
    # Apply crossfade
    overlap = a_tail * gain_a + b_head * gain_b
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def crossfade_exponential(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0
) -> np.ndarray:
    """
    Exponential crossfade (fast start, slow finish).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> mixed = crossfade_exponential(track1, track2, 44100, duration=4.0)
    
    Note:
        Exponential curves emphasize the beginning of the transition.
        Good for dramatic effect or when you want track B to come in quickly.
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # Exponential curves
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    gain_a = np.sqrt(1.0 - t)  # Square root (exponential decay)
    gain_b = np.sqrt(t)        # Square root (exponential rise)
    
    # Apply crossfade
    overlap = a_tail * gain_a + b_head * gain_b
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def crossfade_scurve(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0
) -> np.ndarray:
    """
    S-curve crossfade (smoothest transition).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> mixed = crossfade_scurve(track1, track2, 44100, duration=4.0)
    
    Note:
        S-curves (smoothstep) provide the smoothest, most natural transition.
        Slow at beginning and end, fast in middle.
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # S-curve (smoothstep): 3t² - 2t³
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    s = 3 * t**2 - 2 * t**3
    gain_a = 1.0 - s
    gain_b = s
    
    # Apply crossfade
    overlap = a_tail * gain_a + b_head * gain_b
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def crossfade_frequency(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0,
    crossover_freqs: Tuple[float, float] = (200.0, 2000.0),
    bass_swap: bool = True,
    curve: Literal['linear', 'equal_power', 'exponential', 'scurve'] = 'equal_power'
) -> np.ndarray:
    """
    Frequency-split crossfade (DJ-style bass swap).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
        crossover_freqs: (low_mid_freq, mid_high_freq) in Hz
        bass_swap: If True, aggressively swap bass (duck outgoing bass)
        curve: Crossfade curve type
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> # Swap bass aggressively, fade mids/highs smoothly
        >>> mixed = crossfade_frequency(track1, track2, 44100, 
        >>>                             duration=4.0, bass_swap=True)
    
    Note:
        This is the classic DJ crossfade technique:
        - Low frequencies (bass) swap quickly to avoid muddy overlap
        - Mid/high frequencies fade smoothly for musical transition
        Essential for electronic music mashups!
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # Get fade curves
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    gain_a, gain_b = _get_crossfade_curves(t, curve)
    
    # Split into frequency bands
    low_mid_freq, mid_high_freq = crossover_freqs
    
    # Split track A tail
    a_low = eq_lowpass(a_tail, sr, low_mid_freq)
    a_mid = eq_bandpass(a_tail, sr, low_mid_freq, mid_high_freq)
    a_high = eq_highpass(a_tail, sr, mid_high_freq)
    
    # Split track B head
    b_low = eq_lowpass(b_head, sr, low_mid_freq)
    b_mid = eq_bandpass(b_head, sr, low_mid_freq, mid_high_freq)
    b_high = eq_highpass(b_head, sr, mid_high_freq)
    
    # Crossfade each band
    if bass_swap:
        # Aggressive bass swap (reduce outgoing bass by 50%)
        overlap_low = a_low * (gain_a * 0.5) + b_low * gain_b
    else:
        overlap_low = a_low * gain_a + b_low * gain_b
    
    overlap_mid = a_mid * gain_a + b_mid * gain_b
    overlap_high = a_high * gain_a + b_high * gain_b
    
    # Sum bands
    overlap = overlap_low + overlap_mid + overlap_high
    overlap = np.clip(overlap, -1.0, 1.0)
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def _get_crossfade_curves(
    t: np.ndarray,
    curve: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to generate crossfade curves.
    
    Args:
        t: Time array [0..1]
        curve: Curve type
    
    Returns:
        (gain_a, gain_b) arrays
    """
    if curve == 'linear':
        return 1.0 - t, t
    elif curve == 'equal_power':
        return np.cos(t * math.pi / 2.0) ** 2, np.sin(t * math.pi / 2.0) ** 2
    elif curve == 'exponential':
        return np.sqrt(1.0 - t), np.sqrt(t)
    elif curve == 'scurve':
        s = 3 * t**2 - 2 * t**3
        return 1.0 - s, s
    else:
        logger.warning(f"Unknown curve '{curve}', using equal_power")
        return np.cos(t * math.pi / 2.0) ** 2, np.sin(t * math.pi / 2.0) ** 2


# =============================================================================
# EQ & FILTERING
# =============================================================================

def eq_highpass(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    High-pass filter (remove frequencies below cutoff).
    
    Args:
        y: Audio time series
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz
        order: Filter order (higher = steeper slope)
    
    Returns:
        Filtered audio
    
    Example:
        >>> # Remove bass below 100 Hz
        >>> filtered = eq_highpass(audio, 44100, cutoff_hz=100)
    
    Note:
        Common uses:
        - Remove rumble/sub-bass: 20-40 Hz
        - Clean up bass: 80-100 Hz
        - Thin out track for layering: 200-300 Hz
    """
    nyquist = sr / 2.0
    cutoff_norm = cutoff_hz / nyquist
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
    
    sos = signal.butter(order, cutoff_norm, btype='highpass', output='sos')
    filtered = signal.sosfilt(sos, y)
    
    return filtered.astype(np.float32)


def eq_lowpass(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 5000.0,
    order: int = 4
) -> np.ndarray:
    """
    Low-pass filter (remove frequencies above cutoff).
    
    Args:
        y: Audio time series
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz
        order: Filter order (higher = steeper slope)
    
    Returns:
        Filtered audio
    
    Example:
        >>> # Keep only bass and mids
        >>> filtered = eq_lowpass(audio, 44100, cutoff_hz=2000)
    
    Note:
        Common uses:
        - Isolate bass: 200-300 Hz
        - Remove harshness: 8000-10000 Hz
        - Create telephone effect: 3000-4000 Hz
        - Lo-fi effect: 5000-8000 Hz
    """
    nyquist = sr / 2.0
    cutoff_norm = cutoff_hz / nyquist
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
    
    sos = signal.butter(order, cutoff_norm, btype='lowpass', output='sos')
    filtered = signal.sosfilt(sos, y)
    
    return filtered.astype(np.float32)


def eq_bandpass(
    y: np.ndarray,
    sr: int,
    low_hz: float = 200.0,
    high_hz: float = 2000.0,
    order: int = 4
) -> np.ndarray:
    """
    Band-pass filter (keep only frequencies in range).
    
    Args:
        y: Audio time series
        sr: Sample rate
        low_hz: Low cutoff frequency in Hz
        high_hz: High cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio
    
    Example:
        >>> # Keep only vocal range
        >>> filtered = eq_bandpass(audio, 44100, low_hz=300, high_hz=3000)
    
    Note:
        Common uses:
        - Isolate vocals: 300-3000 Hz
        - Isolate kick drum: 50-100 Hz
        - Telephone effect: 300-3400 Hz
    """
    nyquist = sr / 2.0
    low_norm = np.clip(low_hz / nyquist, 0.01, 0.99)
    high_norm = np.clip(high_hz / nyquist, 0.01, 0.99)
    
    if low_norm >= high_norm:
        logger.warning("Invalid bandpass range, returning original audio")
        return y
    
    sos = signal.butter(order, [low_norm, high_norm], btype='bandpass', output='sos')
    filtered = signal.sosfilt(sos, y)
    
    return filtered.astype(np.float32)


def eq_bandstop(
    y: np.ndarray,
    sr: int,
    low_hz: float = 1000.0,
    high_hz: float = 2000.0,
    order: int = 4
) -> np.ndarray:
    """
    Band-stop filter (remove frequencies in range - notch filter).
    
    Args:
        y: Audio time series
        sr: Sample rate
        low_hz: Low cutoff frequency in Hz
        high_hz: High cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio
    
    Example:
        >>> # Remove 1-2 kHz range
        >>> filtered = eq_bandstop(audio, 44100, low_hz=1000, high_hz=2000)
    
    Note:
        Common uses:
        - Remove harsh resonances
        - Notch out specific instruments
        - Create space for vocals in mix
    """
    nyquist = sr / 2.0
    low_norm = np.clip(low_hz / nyquist, 0.01, 0.99)
    high_norm = np.clip(high_hz / nyquist, 0.01, 0.99)
    
    if low_norm >= high_norm:
        logger.warning("Invalid bandstop range, returning original audio")
        return y
    
    sos = signal.butter(order, [low_norm, high_norm], btype='bandstop', output='sos')
    filtered = signal.sosfilt(sos, y)
    
    return filtered.astype(np.float32)


def eq_parametric(
    y: np.ndarray,
    sr: int,
    center_hz: float = 1000.0,
    gain_db: float = 0.0,
    q: float = 1.0
) -> np.ndarray:
    """
    Parametric EQ (boost or cut at specific frequency).
    
    Args:
        y: Audio time series
        sr: Sample rate
        center_hz: Center frequency in Hz
        gain_db: Boost (+) or cut (-) in dB
        q: Q factor (bandwidth control, higher = narrower)
    
    Returns:
        EQ'd audio
    
    Example:
        >>> # Boost 2kHz by 6dB (presence boost)
        >>> boosted = eq_parametric(audio, 44100, center_hz=2000, gain_db=6.0, q=2.0)
        >>> # Cut 500Hz by 3dB (reduce muddiness)
        >>> cut = eq_parametric(audio, 44100, center_hz=500, gain_db=-3.0, q=1.5)
    
    Note:
        Parametric EQ is the most flexible EQ type. Common uses:
        - Presence boost: +3-6dB at 2-5kHz
        - Remove muddiness: -3dB at 200-500Hz
        - Add warmth: +2-3dB at 100-200Hz
    """
    if abs(gain_db) < 0.1:
        return y  # No change needed
    
    nyquist = sr / 2.0
    center_norm = np.clip(center_hz / nyquist, 0.01, 0.99)
    
    # Convert gain to linear
    A = 10.0 ** (gain_db / 40.0)
    
    # Calculate filter coefficients (peaking EQ)
    w0 = 2 * math.pi * center_norm
    alpha = math.sin(w0) / (2 * q)
    
    # Peaking filter coefficients
    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    
    # Normalize
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    
    # Apply filter
    filtered = signal.lfilter(b, a, y)
    
    return filtered.astype(np.float32)


def eq_shelf_high(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 5000.0,
    gain_db: float = 0.0,
    slope: float = 1.0
) -> np.ndarray:
    """
    High shelf EQ (boost/cut all frequencies above cutoff).
    
    Args:
        y: Audio time series
        sr: Sample rate
        cutoff_hz: Shelf frequency in Hz
        gain_db: Boost (+) or cut (-) in dB
        slope: Shelf slope (0.5-2.0, default 1.0)
    
    Returns:
        EQ'd audio
    
    Example:
        >>> # Add air/brightness (+3dB above 8kHz)
        >>> bright = eq_shelf_high(audio, 44100, cutoff_hz=8000, gain_db=3.0)
        >>> # Reduce harshness (-2dB above 6kHz)
        >>> smooth = eq_shelf_high(audio, 44100, cutoff_hz=6000, gain_db=-2.0)
    """
    if abs(gain_db) < 0.1:
        return y
    
    nyquist = sr / 2.0
    cutoff_norm = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
    
    # Use IIR filter for shelf
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * math.pi * cutoff_norm
    alpha = math.sin(w0) / 2 * math.sqrt((A + 1/A) * (1/slope - 1) + 2)
    
    cos_w0 = math.cos(w0)
    
    # High shelf coefficients
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    
    filtered = signal.lfilter(b, a, y)
    
    return filtered.astype(np.float32)


def eq_shelf_low(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 200.0,
    gain_db: float = 0.0,
    slope: float = 1.0
) -> np.ndarray:
    """
    Low shelf EQ (boost/cut all frequencies below cutoff).
    
    Args:
        y: Audio time series
        sr: Sample rate
        cutoff_hz: Shelf frequency in Hz
        gain_db: Boost (+) or cut (-) in dB
        slope: Shelf slope (0.5-2.0, default 1.0)
    
    Returns:
        EQ'd audio
    
    Example:
        >>> # Add bass weight (+4dB below 150Hz)
        >>> bassy = eq_shelf_low(audio, 44100, cutoff_hz=150, gain_db=4.0)
        >>> # Reduce bass (-3dB below 200Hz)
        >>> thin = eq_shelf_low(audio, 44100, cutoff_hz=200, gain_db=-3.0)
    """
    if abs(gain_db) < 0.1:
        return y
    
    nyquist = sr / 2.0
    cutoff_norm = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
    
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * math.pi * cutoff_norm
    alpha = math.sin(w0) / 2 * math.sqrt((A + 1/A) * (1/slope - 1) + 2)
    
    cos_w0 = math.cos(w0)
    
    # Low shelf coefficients
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    
    filtered = signal.lfilter(b, a, y)
    
    return filtered.astype(np.float32)


# =============================================================================
# STEREO OPERATIONS
# =============================================================================

def pan(
    y: np.ndarray,
    position: float = 0.0,
    law: Literal['linear', 'constant_power'] = 'constant_power'
) -> np.ndarray:
    """
    Pan mono audio in stereo field.
    
    Args:
        y: Mono audio time series
        position: Pan position (-1.0 = full left, 0.0 = center, 1.0 = full right)
        law: Panning law ('linear' or 'constant_power')
    
    Returns:
        Stereo audio [2, n_samples]
    
    Example:
        >>> # Pan to left
        >>> stereo = pan(mono, position=-0.7)
        >>> # Pan to right
        >>> stereo = pan(mono, position=0.7)
    
    Note:
        Constant power panning maintains perceived loudness as sound moves.
        Use this for smooth panning without volume dips.
    """
    position = np.clip(position, -1.0, 1.0)
    
    # Normalize to [0, 1]
    t = (position + 1.0) / 2.0
    
    if law == 'constant_power':
        # Constant power (equal energy)
        left_gain = np.cos(t * math.pi / 2.0)
        right_gain = np.sin(t * math.pi / 2.0)
    else:
        # Linear
        left_gain = 1.0 - t
        right_gain = t
    
    # Create stereo
    left = y * left_gain
    right = y * right_gain
    
    stereo = np.vstack([left, right])
    
    return stereo.astype(np.float32)


def stereo_width(
    y: np.ndarray,
    width: float = 1.0
) -> np.ndarray:
    """
    Adjust stereo width (mono ← → wide).
    
    Args:
        y: Stereo audio [2, n_samples] or [n_samples] (will convert mono to stereo)
        width: Width amount
            - 0.0 = mono (no stereo)
            - 1.0 = original width
            - >1.0 = wider stereo (e.g., 1.5 = 150% wider)
            - <0.0 = inverted stereo (left↔right swapped)
    
    Returns:
        Width-adjusted stereo audio [2, n_samples]
    
    Example:
        >>> # Narrow stereo field (80%)
        >>> narrow = stereo_width(stereo, width=0.8)
        >>> # Widen stereo field (150%)
        >>> wide = stereo_width(stereo, width=1.5)
        >>> # Collapse to mono
        >>> mono_out = stereo_width(stereo, width=0.0)
    """
    # Handle mono input
    if y.ndim == 1:
        y = np.vstack([y, y])
    
    # Extract left and right
    left = y[0]
    right = y[1]
    
    # Convert to mid-side
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    
    # Adjust side signal
    side = side * width
    
    # Convert back to left-right
    left_out = mid + side
    right_out = mid - side
    
    output = np.vstack([left_out, right_out])
    
    return output.astype(np.float32)


def mid_side_encode(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert stereo (L/R) to mid-side (M/S) encoding.
    
    Args:
        y: Stereo audio [2, n_samples] or [n_samples, 2]
    
    Returns:
        (mid, side) channels
        - mid: Sum of L+R (center/mono information)
        - side: Difference L-R (stereo information)
    
    Example:
        >>> mid, side = mid_side_encode(stereo)
        >>> # Process mid and side separately
        >>> mid_compressed = compress(mid, sr, ...)
        >>> side_widened = side * 1.5
        >>> stereo_out = mid_side_decode(mid_compressed, side_widened)
    """
    # Handle different input shapes
    if y.ndim == 1:
        return y, np.zeros_like(y)
    
    if y.shape[0] == 2:
        left, right = y[0], y[1]
    else:
        left, right = y[:, 0], y[:, 1]
    
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    
    return mid.astype(np.float32), side.astype(np.float32)


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Convert mid-side (M/S) back to stereo (L/R).
    
    Args:
        mid: Mid channel (center/mono)
        side: Side channel (stereo)
    
    Returns:
        Stereo audio [2, n_samples]
    
    Example:
        >>> mid, side = mid_side_encode(stereo)
        >>> # Process...
        >>> stereo_out = mid_side_decode(mid, side)
    """
    left = mid + side
    right = mid - side
    
    stereo = np.vstack([left, right])
    
    return stereo.astype(np.float32)


def mid_side_process(
    y: np.ndarray,
    mid_processor: Optional[Callable] = None,
    side_processor: Optional[Callable] = None
) -> np.ndarray:
    """
    Process mid and side channels separately.
    
    Args:
        y: Stereo audio [2, n_samples]
        mid_processor: Function to process mid channel (e.g., lambda m: compress(m, ...))
        side_processor: Function to process side channel (e.g., lambda s: s * 1.5)
    
    Returns:
        Processed stereo audio
    
    Example:
        >>> # Compress center, widen sides
        >>> processed = mid_side_process(
        >>>     stereo,
        >>>     mid_processor=lambda m: compress(m, 44100, ratio=4.0),
        >>>     side_processor=lambda s: s * 1.3
        >>> )
    
    Note:
        Mid-side processing is powerful for:
        - Compressing vocals (mid) without affecting stereo width
        - Widening background (side) without affecting center
        - De-essing only the center channel
    """
    # Encode to M/S
    mid, side = mid_side_encode(y)
    
    # Process
    if mid_processor is not None:
        mid = mid_processor(mid)
    
    if side_processor is not None:
        side = side_processor(side)
    
    # Decode back to L/R
    return mid_side_decode(mid, side)


# =============================================================================
# ADVANCED MIXING
# =============================================================================

def blend_tracks(
    tracks: List[np.ndarray],
    gains_db: Optional[List[float]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Mix multiple tracks with individual gain control.
    
    Args:
        tracks: List of audio tracks (will be aligned to shortest)
        gains_db: List of gains in dB (one per track, default 0dB each)
        normalize: If True, normalize output to prevent clipping
    
    Returns:
        Mixed audio
    
    Example:
        >>> # Mix 3 tracks: drums at 0dB, bass at -3dB, melody at -6dB
        >>> mixed = blend_tracks([drums, bass, melody], 
        >>>                      gains_db=[0, -3, -6])
    """
    if not tracks:
        return np.array([])
    
    # Default gains
    if gains_db is None:
        gains_db = [0.0] * len(tracks)
    
    # Find shortest track
    min_length = min(len(t) for t in tracks)
    
    # Initialize output
    output = np.zeros(min_length, dtype=np.float32)
    
    # Sum tracks with gains
    for track, gain_db in zip(tracks, gains_db):
        gain = 10.0 ** (gain_db / 20.0)
        output += track[:min_length] * gain
    
    # Normalize if requested
    if normalize:
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output / peak * 0.99
    
    return output


def harmonic_crossfade(
    track_a: np.ndarray,
    track_b: np.ndarray,
    sr: int,
    duration: float = 2.0,
    mid_band: Tuple[float, float] = (300.0, 3000.0),
    duck_db: float = 4.0,
    curve: str = 'equal_power'
) -> np.ndarray:
    """
    Harmonic crossfade (duck mid frequencies to reduce clashing).
    
    Args:
        track_a: First audio track (fading out)
        track_b: Second audio track (fading in)
        sr: Sample rate
        duration: Crossfade duration in seconds
        mid_band: (low_hz, high_hz) frequency range to duck
        duck_db: How much to duck mid band in dB
        curve: Crossfade curve type
    
    Returns:
        Crossfaded audio
    
    Example:
        >>> # Reduce vocal clashing during transition
        >>> mixed = harmonic_crossfade(track1, track2, 44100,
        >>>                           mid_band=(500, 4000), duck_db=6.0)
    
    Note:
        This technique reduces harmonic clashing (especially vocals/melodies)
        during transitions by temporarily ducking the mid frequencies of the
        outgoing track. Common in professional DJ mixing.
    """
    track_a = track_a.flatten()
    track_b = track_b.flatten()
    
    n_fade = max(1, int(duration * sr))
    
    # Split tracks
    a_head = track_a[:-n_fade] if len(track_a) > n_fade else np.zeros(0, dtype=np.float32)
    a_tail = track_a[-n_fade:]
    b_head = track_b[:n_fade]
    b_tail = track_b[n_fade:] if len(track_b) > n_fade else np.zeros(0, dtype=np.float32)
    
    # Get fade curves
    t = np.linspace(0, 1, n_fade, dtype=np.float32)
    gain_a, gain_b = _get_crossfade_curves(t, curve)
    
    # Split track A into mid and sides (low+high)
    a_mid = eq_bandpass(a_tail, sr, mid_band[0], mid_band[1])
    a_low = eq_lowpass(a_tail, sr, mid_band[0])
    a_high = eq_highpass(a_tail, sr, mid_band[1])
    a_sides = a_low + a_high
    
    # Duck mid band
    duck_gain = 10.0 ** (-duck_db / 20.0)
    
    # Apply crossfade
    overlap = (a_mid * gain_a * duck_gain + a_sides * gain_a) + (b_head * gain_b)
    overlap = np.clip(overlap, -1.0, 1.0)
    
    # Concatenate
    output = np.concatenate([a_head, overlap, b_tail])
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)


def split_frequency_bands(
    y: np.ndarray,
    sr: int,
    crossover_freqs: Tuple[float, float] = (200.0, 2000.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split audio into low, mid, and high frequency bands.
    
    Args:
        y: Audio time series
        sr: Sample rate
        crossover_freqs: (low_mid_freq, mid_high_freq) in Hz
    
    Returns:
        (low_band, mid_band, high_band)
    
    Example:
        >>> low, mid, high = split_frequency_bands(audio, 44100, (200, 2000))
        >>> # Process each band separately
        >>> low_compressed = compress(low, 44100, ratio=4.0)
        >>> # Recombine
        >>> processed = low_compressed + mid + high
    """
    low_mid_freq, mid_high_freq = crossover_freqs
    
    low = eq_lowpass(y, sr, low_mid_freq)
    mid = eq_bandpass(y, sr, low_mid_freq, mid_high_freq)
    high = eq_highpass(y, sr, mid_high_freq)
    
    return low, mid, high


def filter_sweep(
    y: np.ndarray,
    sr: int,
    filter_type: Literal['lowpass', 'highpass'] = 'lowpass',
    start_hz: float = 20000.0,
    end_hz: float = 200.0
) -> np.ndarray:
    """
    Animated filter sweep effect (DJ-style).
    
    Args:
        y: Audio time series
        sr: Sample rate
        filter_type: 'lowpass' (typical) or 'highpass'
        start_hz: Starting cutoff frequency
        end_hz: Ending cutoff frequency
    
    Returns:
        Audio with filter sweep applied
    
    Example:
        >>> # Classic DJ filter sweep (high to low)
        >>> swept = filter_sweep(audio, 44100, 'lowpass', 
        >>>                      start_hz=20000, end_hz=500)
    
    Note:
        This creates a time-varying filter effect popular in DJ transitions.
        Lowpass sweep (high→low): Classic "closing" effect
        Highpass sweep (low→high): "Opening up" effect
    """
    y = y.astype(np.float32).copy()
    n = len(y)
    
    # Create frequency sweep
    freqs = np.linspace(start_hz, end_hz, n).astype(np.float32)
    
    # Simple 1-pole filter with time-varying cutoff
    x = 0.0
    output = np.zeros_like(y)
    
    for i in range(n):
        fc = float(np.clip(freqs[i], 40.0, sr/2 - 100.0))
        
        # Calculate filter coefficient
        alpha = float(np.exp(-2.0 * math.pi * fc / sr))
        
        # Apply filter
        x = alpha * x + (1 - alpha) * y[i]
        
        if filter_type == 'lowpass':
            output[i] = x
        else:  # highpass
            output[i] = y[i] - x
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output = output / peak
    
    return output.astype(np.float32)


if __name__ == '__main__':
    # Quick test
    print("Mixing Manipulations Module")
    print(f"Available functions: {len(__all__)}")
    print("Core: crossfades, EQ/filters, stereo operations, blending")
