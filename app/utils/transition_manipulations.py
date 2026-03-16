"""
Transition Effects for Professional DJ-Style Mashups

This module provides creative transition effects that make mashups sound
professional and polished. These are the "tricks" DJs use to smoothly
move between tracks or create exciting moments.

Build-Up Effects:
- create_riser: Upward pitch sweep (builds tension)
- create_white_noise_riser: White noise build-up
- create_filter_riser: Filter sweep build-up
- apply_buildup: Automated build-up with multiple effects

Drop Effects:
- create_impact: Heavy impact/hit for drops
- apply_drop: Automated drop with silence/filter
- create_reverse_cymbal: Reverse cymbal crash

Echo/Delay Transitions:
- apply_echo_out: Echoing fade-out (classic DJ exit)
- apply_delay_throw: Delay throw effect
- apply_ping_pong_delay: Stereo ping-pong delay

Rhythmic Effects:
- apply_stutter_buildup: Stuttering build-up
- apply_beat_repeat: Repeat beats with decay
- apply_half_time: Half-time effect (trap-style)
- apply_double_time: Double-time effect

Reverse Effects:
- apply_reverse_buildup: Reverse audio build-up
- apply_reverse_echo: Reverse echo/delay
- apply_spinback: DJ spinback/rewind effect

Filter Sweeps:
- apply_filter_sweep_transition: Animated filter sweep
- apply_highpass_sweep: High-pass sweep (remove bass gradually)
- apply_lowpass_sweep: Low-pass sweep (darken sound gradually)

Special Transitions:
- apply_vinyl_stop: Vinyl stop/slowdown
- apply_scratch: DJ scratch effect
- create_silence_gap: Create dramatic silence
- apply_gate_stutter: Gated stutter pattern

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import math
from typing import Tuple, Optional, Literal, List

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

__all__ = [
    'create_riser',
    'create_white_noise_riser',
    'create_filter_riser',
    'apply_buildup',
    'create_impact',
    'apply_drop',
    'create_reverse_cymbal',
    'apply_echo_out',
    'apply_delay_throw',
    'apply_ping_pong_delay',
    'apply_stutter_buildup',
    'apply_beat_repeat',
    'apply_half_time',
    'apply_double_time',
    'apply_reverse_buildup',
    'apply_reverse_echo',
    'apply_spinback',
    'apply_filter_sweep_transition',
    'apply_highpass_sweep',
    'apply_lowpass_sweep',
    'apply_vinyl_stop',
    'apply_scratch',
    'create_silence_gap',
    'apply_gate_stutter',
]


# =============================================================================
# BUILD-UP EFFECTS (RISERS, TENSION BUILDERS)
# =============================================================================

def create_riser(
    duration: float,
    sr: int,
    start_freq: float = 100.0,
    end_freq: float = 2000.0,
    riser_type: Literal['sine', 'saw', 'noise'] = 'sine'
) -> np.ndarray:
    """
    Create upward pitch sweep riser (builds tension before drops).
    
    Args:
        duration: Riser duration in seconds
        sr: Sample rate
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        riser_type: Riser sound type
    
    Returns:
        Riser audio signal
    
    Example:
        >>> # 4-bar riser before drop
        >>> riser = create_riser(duration=8.0, sr=44100, 
        >>>                      start_freq=100, end_freq=4000, riser_type='saw')
    
    Note:
        Layer this over the track during build-ups. Works great
        with white noise risers for extra energy.
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Exponential frequency sweep
    freq_curve = start_freq * (end_freq / start_freq) ** (t / duration)
    
    # Generate oscillator
    if riser_type == 'sine':
        # Sine wave sweep
        phase = 2 * np.pi * np.cumsum(freq_curve) / sr
        riser = np.sin(phase)
    elif riser_type == 'saw':
        # Sawtooth sweep (brighter, more harmonics)
        phase = np.cumsum(freq_curve) / sr
        riser = 2.0 * (phase % 1.0) - 1.0
    elif riser_type == 'noise':
        # Filtered noise sweep
        riser = np.random.randn(n_samples)
        
        # Apply time-varying highpass filter
        for i in range(0, n_samples, 512):
            block_end = min(i + 512, n_samples)
            nyquist = sr / 2.0
            cutoff = min(freq_curve[i] / nyquist, 0.99)
            
            if cutoff > 0.01:
                sos = signal.butter(2, cutoff, btype='highpass', output='sos')
                riser[i:block_end] = signal.sosfilt(sos, riser[i:block_end])
    else:
        phase = 2 * np.pi * np.cumsum(freq_curve) / sr
        riser = np.sin(phase)
    
    # Apply amplitude envelope (fade in)
    envelope = np.linspace(0, 1, n_samples) ** 2
    riser = riser * envelope
    
    # Normalize
    peak = np.max(np.abs(riser))
    if peak > 1e-9:
        riser = riser / peak * 0.7  # Leave headroom
    
    return riser.astype(np.float32)


def create_white_noise_riser(
    duration: float,
    sr: int,
    highpass_start: float = 200.0,
    highpass_end: float = 4000.0
) -> np.ndarray:
    """
    Create white noise riser with highpass sweep.
    
    Args:
        duration: Riser duration in seconds
        sr: Sample rate
        highpass_start: Starting highpass frequency in Hz
        highpass_end: Ending highpass frequency in Hz
    
    Returns:
        White noise riser
    
    Example:
        >>> # Classic EDM white noise riser
        >>> noise_riser = create_white_noise_riser(8.0, 44100,
        >>>                                        highpass_start=500, 
        >>>                                        highpass_end=8000)
    """
    n_samples = int(duration * sr)
    
    # Generate white noise
    noise = np.random.randn(n_samples).astype(np.float32)
    
    # Exponential frequency sweep
    t = np.arange(n_samples) / sr
    freq_curve = highpass_start * (highpass_end / highpass_start) ** (t / duration)
    
    # Apply time-varying highpass filter
    output = np.zeros_like(noise)
    nyquist = sr / 2.0
    
    for i in range(0, n_samples, 512):
        block_end = min(i + 512, n_samples)
        cutoff = min(freq_curve[i] / nyquist, 0.99)
        
        if cutoff > 0.01:
            sos = signal.butter(4, cutoff, btype='highpass', output='sos')
            output[i:block_end] = signal.sosfilt(sos, noise[i:block_end])
        else:
            output[i:block_end] = noise[i:block_end]
    
    # Apply amplitude envelope
    envelope = np.linspace(0, 1, n_samples) ** 1.5
    output = output * envelope
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output = output / peak * 0.6
    
    return output.astype(np.float32)


def create_filter_riser(
    y: np.ndarray,
    sr: int,
    duration: float,
    start_cutoff: float = 200.0,
    end_cutoff: float = 8000.0
) -> np.ndarray:
    """
    Create filter sweep riser from existing audio.
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Riser duration in seconds
        start_cutoff: Starting highpass cutoff in Hz
        end_cutoff: Ending highpass cutoff in Hz
    
    Returns:
        Filtered riser from input audio
    
    Example:
        >>> # Use last 8 seconds as filter riser
        >>> riser = create_filter_riser(audio[-8*44100:], 44100, duration=8.0)
    """
    n_samples = int(duration * sr)
    n_samples = min(n_samples, len(y))
    
    # Extract section
    riser = y[-n_samples:].copy()
    
    # Frequency sweep curve
    t = np.arange(n_samples) / sr
    freq_curve = start_cutoff * (end_cutoff / start_cutoff) ** (t / duration)
    
    # Apply time-varying highpass filter
    output = np.zeros_like(riser)
    nyquist = sr / 2.0
    
    for i in range(0, n_samples, 512):
        block_end = min(i + 512, n_samples)
        cutoff = min(freq_curve[i] / nyquist, 0.99)
        
        if cutoff > 0.01:
            sos = signal.butter(4, cutoff, btype='highpass', output='sos')
            output[i:block_end] = signal.sosfilt(sos, riser[i:block_end])
        else:
            output[i:block_end] = riser[i:block_end]
    
    # Apply amplitude envelope
    envelope = np.linspace(0, 1, n_samples) ** 1.5
    output = output * envelope
    
    return output.astype(np.float32)


def apply_buildup(
    y: np.ndarray,
    sr: int,
    buildup_duration: float = 4.0,
    add_riser: bool = True,
    add_filter_sweep: bool = True,
    add_volume_automation: bool = True
) -> np.ndarray:
    """
    Apply automated build-up effect with multiple techniques.
    
    Args:
        y: Audio time series
        sr: Sample rate
        buildup_duration: Build-up duration in seconds
        add_riser: Add pitch riser
        add_filter_sweep: Add highpass filter sweep
        add_volume_automation: Add volume automation
    
    Returns:
        Audio with build-up applied to the end
    
    Example:
        >>> # Automatic 8-bar build-up
        >>> buildup = apply_buildup(audio, 44100, buildup_duration=8.0)
    
    Note:
        This combines multiple build-up techniques for maximum impact.
        Apply before drops for professional transitions.
    """
    buildup_samples = int(buildup_duration * sr)
    buildup_samples = min(buildup_samples, len(y))
    
    output = y.copy()
    buildup_section = output[-buildup_samples:]
    
    # Apply filter sweep
    if add_filter_sweep:
        t = np.arange(buildup_samples) / sr
        freq_curve = 200 * (8000 / 200) ** (t / buildup_duration)
        
        nyquist = sr / 2.0
        for i in range(0, buildup_samples, 512):
            block_end = min(i + 512, buildup_samples)
            cutoff = min(freq_curve[i] / nyquist, 0.99)
            
            if cutoff > 0.01:
                sos = signal.butter(2, cutoff, btype='highpass', output='sos')
                buildup_section[i:block_end] = signal.sosfilt(sos, buildup_section[i:block_end])
    
    # Apply volume automation
    if add_volume_automation:
        envelope = np.linspace(0.6, 1.0, buildup_samples) ** 2
        buildup_section = buildup_section * envelope
    
    # Add riser
    if add_riser:
        riser = create_riser(buildup_duration, sr, start_freq=100, end_freq=4000, riser_type='saw')
        noise_riser = create_white_noise_riser(buildup_duration, sr, 
                                               highpass_start=500, highpass_end=8000)
        
        # Mix risers with buildup
        buildup_section = buildup_section + riser * 0.3 + noise_riser * 0.25
    
    # Replace end section
    output[-buildup_samples:] = buildup_section
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


# =============================================================================
# DROP EFFECTS (IMPACTS, SILENCE, FILTERS)
# =============================================================================

def create_impact(
    sr: int,
    duration: float = 0.2,
    frequency: float = 50.0
) -> np.ndarray:
    """
    Create heavy impact/hit sound for drops.
    
    Args:
        sr: Sample rate
        duration: Impact duration in seconds
        frequency: Impact frequency in Hz (typically 40-80 Hz)
    
    Returns:
        Impact sound
    
    Example:
        >>> # Create sub-bass impact for drop
        >>> impact = create_impact(44100, duration=0.3, frequency=50)
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Sub-bass sine wave
    impact = np.sin(2 * np.pi * frequency * t)
    
    # Exponential decay envelope
    envelope = np.exp(-t * 10)
    impact = impact * envelope
    
    # Add click/transient at start
    click_samples = min(50, n_samples)
    click = np.exp(-np.arange(click_samples) * 0.5)
    impact[:click_samples] += click * 0.5
    
    # Normalize
    peak = np.max(np.abs(impact))
    if peak > 1e-9:
        impact = impact / peak * 0.8
    
    return impact.astype(np.float32)


def apply_drop(
    y: np.ndarray,
    sr: int,
    drop_position: Optional[int] = None,
    silence_duration: float = 0.0,
    add_impact: bool = True,
    filter_type: Optional[Literal['highpass', 'lowpass']] = 'highpass'
) -> np.ndarray:
    """
    Apply drop effect with optional silence and filtering.
    
    Args:
        y: Audio time series
        sr: Sample rate
        drop_position: Drop position in samples (None = auto-detect end)
        silence_duration: Silence gap before drop in seconds
        add_impact: Add sub-bass impact at drop
        filter_type: Optional filter to remove before drop
    
    Returns:
        Audio with drop effect
    
    Example:
        >>> # Classic EDM drop with silence gap
        >>> dropped = apply_drop(audio, 44100, silence_duration=0.5, 
        >>>                      add_impact=True, filter_type='highpass')
    
    Note:
        The silence gap creates anticipation. The impact adds weight.
    """
    if drop_position is None:
        drop_position = len(y)
    
    drop_position = min(drop_position, len(y))
    
    # Calculate silence samples
    silence_samples = int(silence_duration * sr)
    
    # Split audio
    before_drop = y[:drop_position - silence_samples]
    after_drop = y[drop_position:]
    
    # Apply filter to section before drop (last 2 seconds)
    if filter_type is not None:
        filter_section_duration = min(2.0, len(before_drop) / sr)
        filter_samples = int(filter_section_duration * sr)
        
        if filter_samples > 0 and len(before_drop) >= filter_samples:
            filter_section = before_drop[-filter_samples:]
            
            # Apply filter with increasing cutoff
            nyquist = sr / 2.0
            
            if filter_type == 'highpass':
                cutoff = 100.0  # Remove bass
            else:  # lowpass
                cutoff = 2000.0  # Remove highs
            
            cutoff_norm = min(cutoff / nyquist, 0.99)
            sos = signal.butter(4, cutoff_norm, btype=filter_type, output='sos')
            filter_section = signal.sosfilt(sos, filter_section)
            
            before_drop[-filter_samples:] = filter_section
    
    # Create silence gap
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    # Create impact if requested
    impact_audio = np.array([], dtype=np.float32)
    if add_impact:
        impact_audio = create_impact(sr, duration=0.3, frequency=50)
    
    # Combine sections
    output = np.concatenate([before_drop, silence, impact_audio, after_drop])
    
    return output.astype(np.float32)


def create_reverse_cymbal(
    duration: float,
    sr: int
) -> np.ndarray:
    """
    Create reverse cymbal crash (classic drop anticipation sound).
    
    Args:
        duration: Cymbal duration in seconds
        sr: Sample rate
    
    Returns:
        Reverse cymbal sound
    
    Example:
        >>> # 2-second reverse cymbal before drop
        >>> rev_cymbal = create_reverse_cymbal(2.0, 44100)
    
    Note:
        This simulates a reverse cymbal crash using filtered noise.
        Layer over build-ups for extra anticipation.
    """
    n_samples = int(duration * sr)
    
    # Generate noise
    noise = np.random.randn(n_samples).astype(np.float32)
    
    # Bandpass filter (cymbal frequency range)
    nyquist = sr / 2.0
    low_freq = 4000.0 / nyquist
    high_freq = 12000.0 / nyquist
    
    sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', output='sos')
    cymbal = signal.sosfilt(sos, noise)
    
    # Reverse envelope (crescendo)
    t = np.arange(n_samples) / sr
    envelope = (t / duration) ** 2
    cymbal = cymbal * envelope
    
    # Normalize
    peak = np.max(np.abs(cymbal))
    if peak > 1e-9:
        cymbal = cymbal / peak * 0.5
    
    return cymbal.astype(np.float32)


# =============================================================================
# ECHO/DELAY TRANSITIONS
# =============================================================================

def apply_echo_out(
    y: np.ndarray,
    sr: int,
    echo_duration: float = 4.0,
    delay_time: float = 0.5,
    feedback: float = 0.6
) -> np.ndarray:
    """
    Apply echoing fade-out (classic DJ exit technique).
    
    Args:
        y: Audio time series
        sr: Sample rate
        echo_duration: Duration to apply echo to (from end) in seconds
        delay_time: Delay time in seconds
        feedback: Feedback amount (0-1)
    
    Returns:
        Audio with echo-out applied
    
    Example:
        >>> # Echo out last 4 bars
        >>> echo_out = apply_echo_out(audio, 44100, echo_duration=8.0,
        >>>                           delay_time=0.375, feedback=0.7)
    
    Note:
        This is the classic "echo into oblivion" DJ technique.
        Use for smooth exits or transitions.
    """
    echo_samples = int(echo_duration * sr)
    echo_samples = min(echo_samples, len(y))
    
    delay_samples = int(delay_time * sr)
    
    # Extract echo section
    echo_section = y[-echo_samples:].copy()
    
    # Create extended buffer for echoes
    total_length = echo_samples + delay_samples * 8
    output_section = np.zeros(total_length, dtype=np.float32)
    output_section[:echo_samples] = echo_section
    
    # Apply delay with feedback and fadeout
    for i in range(len(echo_section)):
        delayed_pos = i + delay_samples
        
        while delayed_pos < total_length:
            # Calculate feedback decay
            delay_num = (delayed_pos - i) // delay_samples
            decay = feedback ** delay_num
            
            # Add fadeout
            fadeout = 1.0 - (i / len(echo_section))
            
            output_section[delayed_pos] += echo_section[i] * decay * fadeout
            delayed_pos += delay_samples
    
    # Replace end section and extend
    output = np.concatenate([y[:-echo_samples], output_section])
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_delay_throw(
    y: np.ndarray,
    sr: int,
    throw_position: int,
    throw_duration: float = 2.0,
    delay_time: float = 0.25,
    repeats: int = 4
) -> np.ndarray:
    """
    Apply delay throw effect (sudden delay on a section).
    
    Args:
        y: Audio time series
        sr: Sample rate
        throw_position: Position to apply throw (in samples)
        throw_duration: Duration of thrown section in seconds
        delay_time: Delay time in seconds
        repeats: Number of delay repeats
    
    Returns:
        Audio with delay throw
    
    Example:
        >>> # Throw delay on vocal phrase
        >>> thrown = apply_delay_throw(audio, 44100, throw_position=44100*32,
        >>>                            throw_duration=1.0, delay_time=0.5)
    """
    throw_samples = int(throw_duration * sr)
    delay_samples = int(delay_time * sr)
    
    # Extract throw section
    start = max(0, throw_position)
    end = min(len(y), throw_position + throw_samples)
    throw_section = y[start:end].copy()
    
    # Create delay repeats
    total_length = len(throw_section) + delay_samples * repeats
    delayed = np.zeros(total_length, dtype=np.float32)
    delayed[:len(throw_section)] = throw_section
    
    for i in range(1, repeats + 1):
        offset = delay_samples * i
        decay = 0.6 ** i
        delayed[offset:offset + len(throw_section)] += throw_section * decay
    
    # Insert back into audio
    output = y.copy()
    insert_end = min(len(output), start + len(delayed))
    output[start:insert_end] = delayed[:insert_end - start]
    
    return output.astype(np.float32)


def apply_ping_pong_delay(
    y: np.ndarray,
    sr: int,
    delay_time: float = 0.5,
    feedback: float = 0.5,
    wet: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply stereo ping-pong delay.
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        delay_time: Delay time in seconds
        feedback: Feedback amount
        wet: Wet/dry mix
    
    Returns:
        Tuple of (left_channel, right_channel) with ping-pong delay
    
    Example:
        >>> # Create stereo ping-pong effect
        >>> left, right = apply_ping_pong_delay(mono, 44100, 
        >>>                                     delay_time=0.375, feedback=0.6)
    
    Note:
        Delay alternates between left and right channels.
        Creates wide stereo movement.
    """
    delay_samples = int(delay_time * sr)
    
    # Create stereo outputs
    left = np.zeros(len(y) + delay_samples * 8, dtype=np.float32)
    right = np.zeros(len(y) + delay_samples * 8, dtype=np.float32)
    
    # Initialize with dry signal in center
    left[:len(y)] = y
    right[:len(y)] = y
    
    # Ping-pong between channels
    for i in range(len(y)):
        delay_pos = i + delay_samples
        
        for tap in range(8):
            if delay_pos >= len(left):
                break
            
            decay = feedback ** (tap + 1)
            
            if tap % 2 == 0:
                # Ping to left
                left[delay_pos] += y[i] * decay * wet
            else:
                # Pong to right
                right[delay_pos] += y[i] * decay * wet
            
            delay_pos += delay_samples
    
    # Normalize
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 1.0:
        left = left / peak * 0.99
        right = right / peak * 0.99
    
    return left.astype(np.float32), right.astype(np.float32)


# =============================================================================
# RHYTHMIC EFFECTS
# =============================================================================

def apply_stutter_buildup(
    y: np.ndarray,
    sr: int,
    bpm: float,
    buildup_duration: float = 4.0,
    stutter_beats: List[float] = [0.5, 0.25, 0.125]
) -> np.ndarray:
    """
    Apply stuttering build-up (stutter gets faster).
    
    Args:
        y: Audio time series
        sr: Sample rate
        bpm: Tempo in BPM
        buildup_duration: Build-up duration in seconds
        stutter_beats: Stutter subdivisions in beats (gets faster)
    
    Returns:
        Audio with stutter build-up
    
    Example:
        >>> # Stutter build-up: half → quarter → eighth notes
        >>> stutter = apply_stutter_buildup(audio, 44100, bpm=128, 
        >>>                                 buildup_duration=4.0,
        >>>                                 stutter_beats=[0.5, 0.25, 0.125])
    """
    buildup_samples = int(buildup_duration * sr)
    buildup_samples = min(buildup_samples, len(y))
    
    beat_duration = 60.0 / bpm
    output = y.copy()
    
    # Extract buildup section
    buildup_section = output[-buildup_samples:].copy()
    
    # Split into segments for each stutter rate
    n_segments = len(stutter_beats)
    segment_samples = buildup_samples // n_segments
    
    for seg_idx, stutter_beat in enumerate(stutter_beats):
        seg_start = seg_idx * segment_samples
        seg_end = min((seg_idx + 1) * segment_samples, buildup_samples)
        
        stutter_duration = beat_duration * stutter_beat
        stutter_samples = int(stutter_duration * sr)
        
        if stutter_samples > 0:
            # Extract stutter slice
            slice_audio = buildup_section[seg_start:seg_start + stutter_samples]
            
            # Repeat to fill segment
            repeats = (seg_end - seg_start) // len(slice_audio) + 1
            stuttered = np.tile(slice_audio, repeats)[:seg_end - seg_start]
            
            buildup_section[seg_start:seg_end] = stuttered
    
    # Replace end section
    output[-buildup_samples:] = buildup_section
    
    return output.astype(np.float32)


def apply_beat_repeat(
    y: np.ndarray,
    sr: int,
    bpm: float,
    repeat_position: int,
    beat_length: float = 1.0,
    repeats: int = 4,
    decay: float = 0.7
) -> np.ndarray:
    """
    Apply beat repeat effect (Ableton-style).
    
    Args:
        y: Audio time series
        sr: Sample rate
        bpm: Tempo in BPM
        repeat_position: Position to start repeat (in samples)
        beat_length: Length of beat to repeat (in beats)
        repeats: Number of repeats
        decay: Volume decay per repeat
    
    Returns:
        Audio with beat repeat
    
    Example:
        >>> # Repeat 1 beat, 4 times
        >>> repeated = apply_beat_repeat(audio, 44100, bpm=128,
        >>>                              repeat_position=44100*16,
        >>>                              beat_length=1.0, repeats=4)
    """
    beat_duration = 60.0 / bpm
    repeat_samples = int(beat_length * beat_duration * sr)
    
    # Extract beat to repeat
    start = max(0, repeat_position)
    end = min(len(y), start + repeat_samples)
    beat_slice = y[start:end].copy()
    
    # Create repeated section
    repeated_section = []
    for i in range(repeats):
        gain = decay ** i
        repeated_section.append(beat_slice * gain)
    
    repeated = np.concatenate(repeated_section)
    
    # Insert into audio
    output = y.copy()
    insert_len = min(len(repeated), len(output) - start)
    output[start:start + insert_len] = repeated[:insert_len]
    
    return output.astype(np.float32)


def apply_half_time(
    y: np.ndarray,
    sr: int,
    duration: float
) -> np.ndarray:
    """
    Apply half-time effect (trap-style slow-down).
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Duration to apply effect (from end) in seconds
    
    Returns:
        Audio with half-time effect
    
    Example:
        >>> # Half-time last 4 bars
        >>> half_time = apply_half_time(audio, 44100, duration=8.0)
    
    Note:
        Popular in trap and hip-hop. Makes rhythm sound slower
        without changing pitch.
    """
    effect_samples = int(duration * sr)
    effect_samples = min(effect_samples, len(y))
    
    # Extract section
    section = y[-effect_samples:].copy()
    
    # Time-stretch to 2x length (half speed)
    stretched_length = effect_samples * 2
    stretched = np.interp(
        np.linspace(0, len(section) - 1, stretched_length),
        np.arange(len(section)),
        section
    )
    
    # Replace end section (will make track longer)
    output = np.concatenate([y[:-effect_samples], stretched])
    
    return output.astype(np.float32)


def apply_double_time(
    y: np.ndarray,
    sr: int,
    duration: float
) -> np.ndarray:
    """
    Apply double-time effect (speed up rhythm).
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Duration to apply effect (from end) in seconds
    
    Returns:
        Audio with double-time effect
    
    Example:
        >>> # Double-time last 2 bars
        >>> double_time = apply_double_time(audio, 44100, duration=4.0)
    """
    effect_samples = int(duration * sr)
    effect_samples = min(effect_samples, len(y))
    
    # Extract section
    section = y[-effect_samples:].copy()
    
    # Time-compress to 0.5x length (double speed)
    compressed_length = effect_samples // 2
    compressed = np.interp(
        np.linspace(0, len(section) - 1, compressed_length),
        np.arange(len(section)),
        section
    )
    
    # Replace end section (will make track shorter)
    output = np.concatenate([y[:-effect_samples], compressed])
    
    return output.astype(np.float32)


# =============================================================================
# REVERSE EFFECTS
# =============================================================================

def apply_reverse_buildup(
    y: np.ndarray,
    sr: int,
    duration: float = 2.0
) -> np.ndarray:
    """
    Apply reverse audio build-up.
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Duration to reverse (from end) in seconds
    
    Returns:
        Audio with reverse build-up at end
    
    Example:
        >>> # Reverse last 2 seconds before drop
        >>> reversed = apply_reverse_buildup(audio, 44100, duration=2.0)
    """
    reverse_samples = int(duration * sr)
    reverse_samples = min(reverse_samples, len(y))
    
    output = y.copy()
    
    # Reverse end section
    output[-reverse_samples:] = output[-reverse_samples:][::-1]
    
    return output.astype(np.float32)


def apply_reverse_echo(
    y: np.ndarray,
    sr: int,
    delay_time: float = 0.5,
    feedback: float = 0.5
) -> np.ndarray:
    """
    Apply reverse echo/delay (echoes come before the sound).
    
    Args:
        y: Audio time series
        sr: Sample rate
        delay_time: Delay time in seconds
        feedback: Feedback amount
    
    Returns:
        Audio with reverse echo
    
    Example:
        >>> # Psychedelic reverse delay
        >>> rev_echo = apply_reverse_echo(vocal, 44100, delay_time=0.5)
    
    Note:
        Creates a backwards echo effect where echoes precede the sound.
        Psychedelic and disorienting effect.
    """
    delay_samples = int(delay_time * sr)
    
    # Extend audio for pre-echoes
    output = np.zeros(len(y) + delay_samples * 4, dtype=np.float32)
    output[delay_samples * 4:] = y
    
    # Add reverse echoes (before the sound)
    for i in range(len(y)):
        for tap in range(1, 5):
            pos = i + delay_samples * (4 - tap)
            if pos >= 0 and pos < len(output):
                decay = feedback ** tap
                output[pos] += y[i] * decay
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_spinback(
    y: np.ndarray,
    sr: int,
    spinback_duration: float = 1.0,
    direction: Literal['forward', 'backward'] = 'backward'
) -> np.ndarray:
    """
    Apply DJ spinback/rewind effect.
    
    Args:
        y: Audio time series
        sr: Sample rate
        spinback_duration: Duration of spinback in seconds
        direction: Spin direction ('forward' = slow, 'backward' = rewind)
    
    Returns:
        Audio with spinback applied to end
    
    Example:
        >>> # DJ rewind effect
        >>> spinback = apply_spinback(audio, 44100, spinback_duration=1.5,
        >>>                           direction='backward')
    
    Note:
        Classic DJ trick. Backward creates rewind effect.
    """
    spinback_samples = int(spinback_duration * sr)
    spinback_samples = min(spinback_samples, len(y))
    
    # Extract end section
    section = y[-spinback_samples:].copy()
    
    # Create speed curve
    t = np.linspace(0, 1, spinback_samples)
    
    if direction == 'backward':
        # Exponential slowdown and reverse
        speed_curve = np.exp(-5 * t)
        section = section[::-1]
    else:
        # Exponential slowdown forward
        speed_curve = np.exp(-3 * t)
    
    # Resample with varying speed
    output_section = []
    pos = 0.0
    
    for i in range(spinback_samples):
        idx = int(pos)
        if idx < len(section):
            output_section.append(section[idx])
        pos += speed_curve[i]
    
    output_section = np.array(output_section, dtype=np.float32)
    
    # Replace end
    output = y.copy()
    output[-len(output_section):] = output_section
    
    return output.astype(np.float32)


# =============================================================================
# FILTER SWEEPS
# =============================================================================

def apply_filter_sweep_transition(
    y: np.ndarray,
    sr: int,
    duration: float,
    start_freq: float = 200.0,
    end_freq: float = 8000.0,
    filter_type: Literal['highpass', 'lowpass'] = 'highpass',
    resonance: float = 2.0
) -> np.ndarray:
    """
    Apply animated filter sweep transition.
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Sweep duration in seconds
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        filter_type: Filter type
        resonance: Filter resonance (1-10)
    
    Returns:
        Audio with filter sweep
    
    Example:
        >>> # Highpass sweep for build-up
        >>> swept = apply_filter_sweep_transition(audio, 44100, duration=8.0,
        >>>                                       start_freq=200, end_freq=8000)
    """
    sweep_samples = int(duration * sr)
    sweep_samples = min(sweep_samples, len(y))
    
    # Extract section
    section = y[-sweep_samples:].copy()
    
    # Frequency sweep curve
    t = np.arange(sweep_samples) / sr
    if start_freq < end_freq:
        freq_curve = start_freq * (end_freq / start_freq) ** (t / duration)
    else:
        freq_curve = start_freq * (end_freq / start_freq) ** (t / duration)
    
    # Apply time-varying filter
    output_section = np.zeros_like(section)
    nyquist = sr / 2.0
    
    for i in range(0, sweep_samples, 512):
        block_end = min(i + 512, sweep_samples)
        cutoff = min(freq_curve[i] / nyquist, 0.99)
        
        if cutoff > 0.01:
            order = int(resonance * 2)
            sos = signal.butter(order, cutoff, btype=filter_type, output='sos')
            output_section[i:block_end] = signal.sosfilt(sos, section[i:block_end])
        else:
            output_section[i:block_end] = section[i:block_end]
    
    # Replace end section
    output = y.copy()
    output[-sweep_samples:] = output_section
    
    return output.astype(np.float32)


def apply_highpass_sweep(
    y: np.ndarray,
    sr: int,
    duration: float,
    start_freq: float = 20.0,
    end_freq: float = 2000.0
) -> np.ndarray:
    """
    Apply highpass sweep (gradually remove bass).
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Sweep duration in seconds
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
    
    Returns:
        Audio with highpass sweep
    
    Example:
        >>> # Remove bass gradually
        >>> no_bass = apply_highpass_sweep(audio, 44100, duration=4.0,
        >>>                                start_freq=20, end_freq=500)
    """
    return apply_filter_sweep_transition(y, sr, duration, start_freq, end_freq,
                                        filter_type='highpass', resonance=1.0)


def apply_lowpass_sweep(
    y: np.ndarray,
    sr: int,
    duration: float,
    start_freq: float = 20000.0,
    end_freq: float = 500.0
) -> np.ndarray:
    """
    Apply lowpass sweep (gradually darken sound).
    
    Args:
        y: Audio time series
        sr: Sample rate
        duration: Sweep duration in seconds
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
    
    Returns:
        Audio with lowpass sweep
    
    Example:
        >>> # Darken sound gradually
        >>> dark = apply_lowpass_sweep(audio, 44100, duration=4.0,
        >>>                            start_freq=20000, end_freq=1000)
    """
    return apply_filter_sweep_transition(y, sr, duration, start_freq, end_freq,
                                        filter_type='lowpass', resonance=1.0)


# =============================================================================
# SPECIAL TRANSITIONS
# =============================================================================

def apply_vinyl_stop(
    y: np.ndarray,
    sr: int,
    stop_duration: float = 2.0
) -> np.ndarray:
    """
    Apply vinyl stop/slowdown effect.
    
    Args:
        y: Audio time series
        sr: Sample rate
        stop_duration: Duration of slowdown in seconds
    
    Returns:
        Audio with vinyl stop
    
    Example:
        >>> # Classic vinyl stop
        >>> stopped = apply_vinyl_stop(audio, 44100, stop_duration=2.0)
    """
    stop_samples = int(stop_duration * sr)
    stop_samples = min(stop_samples, len(y) // 2)
    
    if stop_samples < 100:
        return y
    
    # Extract stop section
    section = y[-stop_samples:].copy()
    
    # Create exponential speed curve
    t = np.linspace(0, 1, stop_samples)
    speed_curve = np.exp(-6 * t)
    
    # Resample with decreasing speed
    output_section = []
    pos = 0.0
    
    for i in range(stop_samples):
        idx = int(pos)
        if idx < len(section):
            output_section.append(section[idx])
        pos += speed_curve[i]
        
        if pos >= len(section):
            break
    
    output_section = np.array(output_section, dtype=np.float32)
    
    # Replace end
    output = np.concatenate([y[:-stop_samples], output_section])
    
    return output.astype(np.float32)


def apply_scratch(
    y: np.ndarray,
    sr: int,
    scratch_position: int,
    scratch_duration: float = 0.5,
    scratch_speed: float = 3.0
) -> np.ndarray:
    """
    Apply DJ scratch effect.
    
    Args:
        y: Audio time series
        sr: Sample rate
        scratch_position: Position to scratch (in samples)
        scratch_duration: Scratch duration in seconds
        scratch_speed: Scratch speed multiplier
    
    Returns:
        Audio with scratch effect
    
    Example:
        >>> # DJ scratch
        >>> scratched = apply_scratch(audio, 44100, scratch_position=44100*8,
        >>>                           scratch_duration=1.0, scratch_speed=4.0)
    """
    scratch_samples = int(scratch_duration * sr)
    scratch_samples = min(scratch_samples, len(y) - scratch_position)
    
    # Extract scratch section
    start = max(0, scratch_position)
    section = y[start:start + scratch_samples].copy()
    
    # Create scratch pattern (back and forth)
    n_cycles = int(scratch_speed * 2)
    cycle_samples = len(section) // n_cycles
    
    scratched = []
    for i in range(n_cycles):
        cycle_start = i * cycle_samples
        cycle_end = min((i + 1) * cycle_samples, len(section))
        cycle = section[cycle_start:cycle_end]
        
        if i % 2 == 0:
            # Forward
            scratched.append(cycle)
        else:
            # Backward
            scratched.append(cycle[::-1])
    
    scratched = np.concatenate(scratched)
    
    # Insert back
    output = y.copy()
    insert_len = min(len(scratched), len(output) - start)
    output[start:start + insert_len] = scratched[:insert_len]
    
    return output.astype(np.float32)


def create_silence_gap(
    y: np.ndarray,
    sr: int,
    gap_position: int,
    gap_duration: float = 1.0
) -> np.ndarray:
    """
    Create dramatic silence gap.
    
    Args:
        y: Audio time series
        sr: Sample rate
        gap_position: Position to insert gap (in samples)
        gap_duration: Gap duration in seconds
    
    Returns:
        Audio with silence gap
    
    Example:
        >>> # 1-second silence before drop
        >>> gapped = create_silence_gap(audio, 44100, gap_position=44100*64,
        >>>                             gap_duration=1.0)
    
    Note:
        Silence creates anticipation. Use before drops for maximum impact.
    """
    gap_samples = int(gap_duration * sr)
    silence = np.zeros(gap_samples, dtype=np.float32)
    
    # Insert silence
    before = y[:gap_position]
    after = y[gap_position:]
    
    output = np.concatenate([before, silence, after])
    
    return output.astype(np.float32)


def apply_gate_stutter(
    y: np.ndarray,
    sr: int,
    bpm: float,
    gate_pattern: List[int] = [1, 0, 1, 0, 1, 1, 0, 0]
) -> np.ndarray:
    """
    Apply gated stutter pattern.
    
    Args:
        y: Audio time series
        sr: Sample rate
        bpm: Tempo in BPM
        gate_pattern: Gate pattern (1 = on, 0 = off) for 16th notes
    
    Returns:
        Audio with gate stutter
    
    Example:
        >>> # Create gated rhythm pattern
        >>> gated = apply_gate_stutter(audio, 44100, bpm=128,
        >>>                            gate_pattern=[1, 0, 1, 0, 1, 1, 0, 1])
    
    Note:
        Creates rhythmic gating effect. Popular in EDM build-ups.
    """
    # Calculate 16th note duration
    beat_duration = 60.0 / bpm
    sixteenth_duration = beat_duration / 4.0
    sixteenth_samples = int(sixteenth_duration * sr)
    
    # Calculate pattern length
    pattern_samples = sixteenth_samples * len(gate_pattern)
    pattern_samples = min(pattern_samples, len(y))
    
    # Extract section
    section = y[-pattern_samples:].copy()
    
    # Apply gate pattern
    for i, gate in enumerate(gate_pattern):
        start = i * sixteenth_samples
        end = min((i + 1) * sixteenth_samples, len(section))
        
        if gate == 0:
            section[start:end] = 0.0
    
    # Replace end section
    output = y.copy()
    output[-pattern_samples:] = section
    
    return output.astype(np.float32)


if __name__ == '__main__':
    # Quick test
    print("Transition Manipulations Module")
    print(f"Available functions: {len(__all__)}")
    print("Build-ups: riser, white_noise_riser, filter_riser, buildup")
    print("Drops: impact, drop, reverse_cymbal")
    print("Echo: echo_out, delay_throw, ping_pong_delay")
    print("Rhythmic: stutter_buildup, beat_repeat, half_time, double_time")
    print("Reverse: reverse_buildup, reverse_echo, spinback")
    print("Filters: filter_sweep, highpass_sweep, lowpass_sweep")
    print("Special: vinyl_stop, scratch, silence_gap, gate_stutter")
