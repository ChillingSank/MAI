"""
Audio Effects for Creative Mashup Enhancement

This module provides creative audio effects to add color, character,
and excitement to mashups. All effects are independent and production-ready.

Time-Based Effects:
- apply_reverb: Room/hall/plate reverb
- apply_delay: Echo with feedback and filtering
- apply_echo: Simple echo/repeat
- apply_slapback: Short delay (rockabilly style)

Modulation Effects:
- apply_chorus: Thicken sound with detuned copies
- apply_flanger: Jet/swoosh effect
- apply_phaser: Phase shifting sweep
- apply_tremolo: Amplitude modulation
- apply_vibrato: Pitch modulation

Distortion/Saturation:
- apply_distortion: Overdrive/fuzz/hard clip
- apply_saturation: Warm analog-style saturation
- apply_bitcrush: Lo-fi bit depth reduction
- apply_waveshaper: Custom waveshaping curves

Filter Effects:
- apply_autowah: Envelope-follower filter
- apply_talkbox: Formant-style filtering
- apply_resonant_filter: Sweeping resonant filter

Creative/Special Effects:
- apply_ring_mod: Ring modulation (metallic/robotic)
- apply_vocoder: Robotic voice effect (simplified)
- apply_granular: Granular synthesis texture
- apply_reverse: Reverse audio effect
- apply_stutter: Rhythmic stutter/glitch
- apply_tapestop: Vinyl stop/slowdown effect

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import math
from typing import Tuple, Optional, List, Literal

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

__all__ = [
    'apply_reverb',
    'apply_delay',
    'apply_echo',
    'apply_slapback',
    'apply_chorus',
    'apply_flanger',
    'apply_phaser',
    'apply_tremolo',
    'apply_vibrato',
    'apply_distortion',
    'apply_saturation',
    'apply_bitcrush',
    'apply_waveshaper',
    'apply_autowah',
    'apply_talkbox',
    'apply_resonant_filter',
    'apply_ring_mod',
    'apply_vocoder',
    'apply_granular',
    'apply_reverse',
    'apply_stutter',
    'apply_tapestop',
]


# =============================================================================
# TIME-BASED EFFECTS (REVERB, DELAY, ECHO)
# =============================================================================

def apply_reverb(
    y: np.ndarray,
    sr: int,
    room_size: Literal['small', 'medium', 'large', 'hall'] = 'medium',
    wet: float = 0.3,
    decay: float = 0.5,
    damping: float = 0.5
) -> np.ndarray:
    """
    Apply reverb (room ambience).
    
    Args:
        y: Audio time series
        sr: Sample rate
        room_size: Room size preset
        wet: Wet/dry mix (0.0 = dry only, 1.0 = wet only)
        decay: Decay time (0-1, higher = longer reverb tail)
        damping: High frequency damping (0-1, higher = darker reverb)
    
    Returns:
        Audio with reverb applied
    
    Example:
        >>> # Large hall reverb
        >>> reverb = apply_reverb(audio, 44100, room_size='hall', 
        >>>                       wet=0.4, decay=0.7)
    
    Note:
        This is a Schroeder reverb (comb filters + allpass).
        For production reverb, consider using convolution with impulse responses.
    """
    y = y.astype(np.float32)
    
    # Room size presets (delay times in ms)
    room_configs = {
        'small': [11.73, 15.55, 19.10, 23.73],
        'medium': [29.7, 37.1, 41.1, 43.7],
        'large': [50.0, 56.0, 61.0, 68.0],
        'hall': [89.6, 99.8, 107.5, 116.5]
    }
    
    delays_ms = room_configs.get(room_size, room_configs['medium'])
    delays = [int(sr * ms / 1000.0) for ms in delays_ms]
    
    # Comb filter bank (parallel)
    output = np.zeros_like(y)
    
    for delay_samples in delays:
        # Feedback comb filter
        comb_out = np.zeros_like(y)
        buffer = 0.0
        feedback = decay * 0.7
        
        # Apply damping filter coefficient
        damp_coef = 1.0 - damping * 0.5
        
        for i in range(len(y)):
            if i >= delay_samples:
                # Feedback with damping
                buffer = damp_coef * buffer + (1 - damp_coef) * comb_out[i - delay_samples]
                comb_out[i] = y[i] + feedback * buffer
            else:
                comb_out[i] = y[i]
        
        output += comb_out
    
    # Normalize comb output
    output = output / len(delays)
    
    # Allpass filters for diffusion (series)
    allpass_delays = [int(sr * ms / 1000.0) for ms in [5.0, 6.9]]
    
    for ap_delay in allpass_delays:
        ap_out = np.zeros_like(output)
        g = 0.5
        
        for i in range(len(output)):
            if i >= ap_delay:
                ap_out[i] = -g * output[i] + output[i - ap_delay] + g * ap_out[i - ap_delay]
            else:
                ap_out[i] = output[i]
        
        output = ap_out
    
    # Mix wet and dry
    result = (1.0 - wet) * y + wet * output
    
    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    
    return result.astype(np.float32)


def apply_delay(
    y: np.ndarray,
    sr: int,
    delay_time: float = 0.5,
    feedback: float = 0.4,
    wet: float = 0.3,
    filter_cutoff: Optional[float] = None
) -> np.ndarray:
    """
    Apply delay/echo with feedback.
    
    Args:
        y: Audio time series
        sr: Sample rate
        delay_time: Delay time in seconds
        feedback: Feedback amount (0-1, higher = more repeats)
        wet: Wet/dry mix (0-1)
        filter_cutoff: Optional lowpass filter cutoff (Hz) for dub-style delay
    
    Returns:
        Audio with delay applied
    
    Example:
        >>> # Dub-style delay (filtered feedback)
        >>> delayed = apply_delay(audio, 44100, delay_time=0.375, 
        >>>                       feedback=0.6, filter_cutoff=2000)
    """
    y = y.astype(np.float32)
    
    delay_samples = int(delay_time * sr)
    delay_samples = max(1, min(delay_samples, len(y)))
    
    # Create delay buffer
    output = y.copy()
    delay_buffer = np.zeros(len(y) + delay_samples * 10, dtype=np.float32)
    
    # Optional feedback filter
    if filter_cutoff is not None:
        nyquist = sr / 2.0
        cutoff_norm = min(filter_cutoff / nyquist, 0.99)
        b, a = signal.butter(2, cutoff_norm, btype='lowpass')
        filter_state = signal.lfilter_zi(b, a)
    
    # Process delay with feedback
    for i in range(len(y)):
        # Read from delay buffer
        delayed = delay_buffer[i]
        
        # Mix input with delayed signal
        output[i] = y[i] + wet * delayed
        
        # Write to delay buffer with feedback
        write_pos = i + delay_samples
        if write_pos < len(delay_buffer):
            feedback_signal = delayed * feedback
            
            # Apply filter to feedback if specified
            if filter_cutoff is not None:
                feedback_signal, filter_state = signal.lfilter(
                    b, a, [feedback_signal], zi=filter_state
                )
                feedback_signal = feedback_signal[0]
            
            delay_buffer[write_pos] += y[i] + feedback_signal
    
    # Prevent clipping
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_echo(
    y: np.ndarray,
    sr: int,
    delay_time: float = 0.5,
    repeats: int = 3,
    decay: float = 0.6
) -> np.ndarray:
    """
    Apply simple echo (discrete repeats without feedback loop).
    
    Args:
        y: Audio time series
        sr: Sample rate
        delay_time: Time between echoes in seconds
        repeats: Number of echo repeats
        decay: Volume decay per repeat (0-1)
    
    Returns:
        Audio with echo applied
    
    Example:
        >>> # 3 echoes, 0.5s apart
        >>> echoed = apply_echo(audio, 44100, delay_time=0.5, repeats=3)
    """
    y = y.astype(np.float32)
    
    delay_samples = int(delay_time * sr)
    total_length = len(y) + delay_samples * repeats
    
    output = np.zeros(total_length, dtype=np.float32)
    output[:len(y)] = y
    
    # Add echoes
    for i in range(1, repeats + 1):
        start_idx = delay_samples * i
        end_idx = start_idx + len(y)
        gain = decay ** i
        output[start_idx:end_idx] += y * gain
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_slapback(
    y: np.ndarray,
    sr: int,
    delay_ms: float = 120.0,
    wet: float = 0.5
) -> np.ndarray:
    """
    Apply slapback delay (short single echo - rockabilly/Elvis style).
    
    Args:
        y: Audio time series
        sr: Sample rate
        delay_ms: Delay time in milliseconds (typically 80-150ms)
        wet: Wet/dry mix
    
    Returns:
        Audio with slapback applied
    
    Example:
        >>> # Classic rockabilly slapback
        >>> slapped = apply_slapback(vocal, 44100, delay_ms=120)
    """
    delay_samples = int(sr * delay_ms / 1000.0)
    
    output = y.copy()
    
    if delay_samples < len(y):
        # Add single delayed copy
        output[delay_samples:] += wet * y[:len(y) - delay_samples]
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


# =============================================================================
# MODULATION EFFECTS
# =============================================================================

def apply_chorus(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 1.5,
    depth: float = 0.002,
    voices: int = 3,
    wet: float = 0.5
) -> np.ndarray:
    """
    Apply chorus effect (thickens sound with detuned copies).
    
    Args:
        y: Audio time series
        sr: Sample rate
        rate_hz: LFO rate in Hz (typically 0.5-5 Hz)
        depth: Modulation depth in seconds (typically 0.001-0.005)
        voices: Number of chorus voices (2-4)
        wet: Wet/dry mix
    
    Returns:
        Audio with chorus applied
    
    Example:
        >>> # Rich chorus for synth
        >>> chorused = apply_chorus(synth, 44100, rate_hz=2.0, voices=4)
    
    Note:
        Chorus uses modulated delay lines to create detuning effect.
    """
    y = y.astype(np.float32)
    n = len(y)
    
    # Generate time array
    t = np.arange(n) / sr
    
    output = y.copy()
    
    # Create multiple voices with different LFO phases
    for voice in range(voices):
        # Phase offset for each voice
        phase_offset = (2 * math.pi * voice) / voices
        
        # Generate LFO
        lfo = np.sin(2 * math.pi * rate_hz * t + phase_offset)
        
        # Convert to delay modulation (in samples)
        delay_mod = depth * sr * (lfo + 1.0) / 2.0
        delay_mod = delay_mod + sr * 0.01  # Add base delay (10ms)
        
        # Apply time-varying delay
        voice_output = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            # Calculate read position
            read_pos = i - delay_mod[i]
            
            if read_pos >= 0:
                # Linear interpolation
                read_idx = int(read_pos)
                frac = read_pos - read_idx
                
                if read_idx + 1 < n:
                    voice_output[i] = (1 - frac) * y[read_idx] + frac * y[read_idx + 1]
                elif read_idx < n:
                    voice_output[i] = y[read_idx]
        
        output += voice_output / voices
    
    # Mix wet and dry
    result = (1.0 - wet) * y + wet * output
    
    # Normalize
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    
    return result.astype(np.float32)


def apply_flanger(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 0.5,
    depth: float = 0.002,
    feedback: float = 0.5,
    wet: float = 0.5
) -> np.ndarray:
    """
    Apply flanger effect (jet/swoosh sound).
    
    Args:
        y: Audio time series
        sr: Sample rate
        rate_hz: LFO rate in Hz (typically 0.1-2 Hz)
        depth: Modulation depth in seconds (typically 0.001-0.005)
        feedback: Feedback amount (creates resonance peaks)
        wet: Wet/dry mix
    
    Returns:
        Audio with flanger applied
    
    Example:
        >>> # Jet plane flanger
        >>> flanged = apply_flanger(guitar, 44100, rate_hz=0.5, feedback=0.7)
    """
    y = y.astype(np.float32)
    n = len(y)
    
    # Generate LFO
    t = np.arange(n) / sr
    lfo = np.sin(2 * math.pi * rate_hz * t)
    
    # Convert to delay modulation (very short delays for flanger)
    delay_mod = depth * sr * (lfo + 1.0) / 2.0
    delay_mod = delay_mod + sr * 0.001  # Base delay (1ms)
    
    output = np.zeros(n, dtype=np.float32)
    delay_buffer = np.zeros(n, dtype=np.float32)
    
    # Process with feedback
    for i in range(n):
        # Calculate read position
        read_pos = i - delay_mod[i]
        delayed = 0.0
        
        if read_pos >= 0:
            read_idx = int(read_pos)
            frac = read_pos - read_idx
            
            if read_idx + 1 < n:
                delayed = (1 - frac) * delay_buffer[read_idx] + frac * delay_buffer[read_idx + 1]
            elif read_idx < n:
                delayed = delay_buffer[read_idx]
        
        # Mix with input and feedback
        output[i] = y[i] + wet * delayed
        delay_buffer[i] = y[i] + feedback * delayed
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_phaser(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 0.5,
    depth: float = 1.0,
    stages: int = 4,
    feedback: float = 0.5,
    wet: float = 0.5
) -> np.ndarray:
    """
    Apply phaser effect (phase shifting sweep).
    
    Args:
        y: Audio time series
        sr: Sample rate
        rate_hz: LFO rate in Hz
        depth: Modulation depth (0-1)
        stages: Number of allpass stages (2, 4, 6, or 8)
        feedback: Feedback amount
        wet: Wet/dry mix
    
    Returns:
        Audio with phaser applied
    
    Example:
        >>> # Classic 4-stage phaser
        >>> phased = apply_phaser(synth, 44100, stages=4, rate_hz=0.5)
    """
    y = y.astype(np.float32)
    n = len(y)
    
    # Generate LFO
    t = np.arange(n) / sr
    lfo = np.sin(2 * math.pi * rate_hz * t)
    lfo = (lfo + 1.0) / 2.0  # Normalize to 0-1
    
    # Modulate allpass frequency (200 Hz to 2000 Hz)
    freq = 200.0 + depth * 1800.0 * lfo
    
    output = y.copy()
    
    # Apply allpass filter stages
    for stage in range(stages):
        stage_output = np.zeros(n, dtype=np.float32)
        prev_in = 0.0
        prev_out = 0.0
        
        for i in range(n):
            # Time-varying allpass coefficient
            omega = 2.0 * math.pi * freq[i] / sr
            alpha = (1.0 - math.tan(omega / 2.0)) / (1.0 + math.tan(omega / 2.0))
            
            # Allpass filter
            stage_output[i] = alpha * output[i] + prev_in - alpha * prev_out
            prev_in = output[i]
            prev_out = stage_output[i]
        
        output = stage_output
    
    # Add feedback
    if abs(feedback) > 0.01:
        output = y + feedback * output
    
    # Mix wet/dry
    result = (1.0 - wet) * y + wet * output
    
    # Normalize
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    
    return result.astype(np.float32)


def apply_tremolo(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 5.0,
    depth: float = 0.5,
    waveform: Literal['sine', 'triangle', 'square'] = 'sine'
) -> np.ndarray:
    """
    Apply tremolo effect (amplitude modulation).
    
    Args:
        y: Audio time series
        sr: Sample rate
        rate_hz: Modulation rate in Hz (typically 3-8 Hz)
        depth: Modulation depth (0-1)
        waveform: LFO waveform shape
    
    Returns:
        Audio with tremolo applied
    
    Example:
        >>> # Classic tremolo
        >>> tremolo = apply_tremolo(guitar, 44100, rate_hz=5.0, depth=0.6)
    """
    t = np.arange(len(y)) / sr
    
    # Generate LFO
    if waveform == 'sine':
        lfo = np.sin(2 * math.pi * rate_hz * t)
    elif waveform == 'triangle':
        lfo = 2.0 * np.abs(2.0 * (rate_hz * t % 1.0) - 1.0) - 1.0
    elif waveform == 'square':
        lfo = np.sign(np.sin(2 * math.pi * rate_hz * t))
    else:
        lfo = np.sin(2 * math.pi * rate_hz * t)
    
    # Convert to amplitude modulation (0.5 to 1.0 range, or deeper)
    mod = 1.0 - depth * (1.0 - (lfo + 1.0) / 2.0)
    
    # Apply modulation
    output = y * mod
    
    return output.astype(np.float32)


def apply_vibrato(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 5.0,
    depth: float = 0.002
) -> np.ndarray:
    """
    Apply vibrato effect (pitch modulation).
    
    Args:
        y: Audio time series
        sr: Sample rate
        rate_hz: Modulation rate in Hz (typically 4-7 Hz)
        depth: Modulation depth in seconds (typically 0.001-0.005)
    
    Returns:
        Audio with vibrato applied
    
    Example:
        >>> # Vocal vibrato
        >>> vibrato = apply_vibrato(vocal, 44100, rate_hz=6.0, depth=0.003)
    """
    y = y.astype(np.float32)
    n = len(y)
    
    # Generate LFO
    t = np.arange(n) / sr
    lfo = np.sin(2 * math.pi * rate_hz * t)
    
    # Convert to time delay modulation
    delay_mod = depth * sr * lfo
    
    output = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        # Calculate read position
        read_pos = i - delay_mod[i]
        
        if 0 <= read_pos < n:
            # Linear interpolation
            read_idx = int(read_pos)
            frac = read_pos - read_idx
            
            if read_idx + 1 < n:
                output[i] = (1 - frac) * y[read_idx] + frac * y[read_idx + 1]
            else:
                output[i] = y[read_idx]
    
    return output.astype(np.float32)


# =============================================================================
# DISTORTION & SATURATION
# =============================================================================

def apply_distortion(
    y: np.ndarray,
    drive: float = 10.0,
    tone: Literal['soft', 'hard', 'fuzz'] = 'soft',
    output_gain: float = 1.0
) -> np.ndarray:
    """
    Apply distortion/overdrive effect.
    
    Args:
        y: Audio time series
        drive: Distortion amount (1-100, higher = more distortion)
        tone: Distortion character
            - 'soft': Smooth tube-style overdrive
            - 'hard': Hard clipping
            - 'fuzz': Extreme fuzzy distortion
        output_gain: Output level compensation
    
    Returns:
        Distorted audio
    
    Example:
        >>> # Soft tube overdrive
        >>> driven = apply_distortion(guitar, drive=5.0, tone='soft')
        >>> # Extreme fuzz
        >>> fuzzed = apply_distortion(guitar, drive=20.0, tone='fuzz')
    """
    # Apply input gain
    y_driven = y * drive
    
    if tone == 'soft':
        # Soft clipping (tanh)
        output = np.tanh(y_driven)
    elif tone == 'hard':
        # Hard clipping
        output = np.clip(y_driven, -1.0, 1.0)
    elif tone == 'fuzz':
        # Fuzz (asymmetric clipping + extra harmonics)
        output = np.sign(y_driven) * (1.0 - np.exp(-np.abs(y_driven)))
    else:
        output = np.tanh(y_driven)
    
    # Apply output gain
    output = output * output_gain
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_saturation(
    y: np.ndarray,
    amount: float = 0.5,
    warmth: float = 0.5
) -> np.ndarray:
    """
    Apply analog-style saturation (warm harmonic distortion).
    
    Args:
        y: Audio time series
        amount: Saturation amount (0-1)
        warmth: Add warm even harmonics (0-1)
    
    Returns:
        Saturated audio
    
    Example:
        >>> # Tape saturation
        >>> saturated = apply_saturation(mix, amount=0.6, warmth=0.7)
    
    Note:
        Saturation adds pleasant harmonic distortion, unlike harsh clipping.
        Use for adding analog warmth to digital recordings.
    """
    # Drive signal
    driven = y * (1.0 + amount * 2.0)
    
    # Soft saturation curve
    saturated = driven / (1.0 + np.abs(driven))
    
    # Add even harmonics for warmth
    if warmth > 0.01:
        even_harm = warmth * (saturated ** 2) * np.sign(saturated)
        saturated = saturated + even_harm * 0.3
    
    # Normalize
    peak = np.max(np.abs(saturated))
    if peak > 1.0:
        saturated = saturated / peak * 0.99
    
    return saturated.astype(np.float32)


def apply_bitcrush(
    y: np.ndarray,
    bits: int = 8,
    sample_rate_reduction: Optional[int] = None
) -> np.ndarray:
    """
    Apply bitcrusher effect (lo-fi digital degradation).
    
    Args:
        y: Audio time series
        bits: Bit depth (1-16, lower = more crushed)
        sample_rate_reduction: Optional sample rate reduction factor
    
    Returns:
        Bitcrushed audio
    
    Example:
        >>> # 8-bit video game sound
        >>> crushed = apply_bitcrush(synth, bits=8, sample_rate_reduction=4)
    
    Note:
        Bitcrushing creates lo-fi digital artifacts popular in electronic music.
    """
    # Bit depth reduction
    levels = 2 ** bits
    crushed = np.round(y * levels) / levels
    
    # Sample rate reduction (hold-and-sample)
    if sample_rate_reduction is not None and sample_rate_reduction > 1:
        output = np.zeros_like(crushed)
        for i in range(len(crushed)):
            sample_idx = (i // sample_rate_reduction) * sample_rate_reduction
            if sample_idx < len(crushed):
                output[i] = crushed[sample_idx]
        crushed = output
    
    return crushed.astype(np.float32)


def apply_waveshaper(
    y: np.ndarray,
    shape: Literal['tanh', 'cubic', 'sine', 'custom'] = 'tanh',
    drive: float = 1.0
) -> np.ndarray:
    """
    Apply waveshaping distortion with custom transfer curves.
    
    Args:
        y: Audio time series
        shape: Waveshaping curve
        drive: Input drive amount
    
    Returns:
        Waveshaped audio
    
    Example:
        >>> # Sine waveshaping (warm harmonics)
        >>> shaped = apply_waveshaper(bass, shape='sine', drive=2.0)
    """
    y_driven = y * drive
    
    if shape == 'tanh':
        output = np.tanh(y_driven)
    elif shape == 'cubic':
        # Cubic: y - y³/3
        output = y_driven - (y_driven ** 3) / 3.0
        output = np.clip(output, -1.0, 1.0)
    elif shape == 'sine':
        # Sine shaping
        output = np.sin(y_driven * math.pi / 2.0)
    else:
        output = np.tanh(y_driven)
    
    return output.astype(np.float32)


# =============================================================================
# FILTER EFFECTS
# =============================================================================

def apply_autowah(
    y: np.ndarray,
    sr: int,
    sensitivity: float = 0.5,
    q: float = 5.0,
    mix: float = 1.0
) -> np.ndarray:
    """
    Apply auto-wah (envelope-controlled filter).
    
    Args:
        y: Audio time series
        sr: Sample rate
        sensitivity: Envelope sensitivity (0-1)
        q: Filter resonance (1-20)
        mix: Wet/dry mix
    
    Returns:
        Audio with auto-wah applied
    
    Example:
        >>> # Funky auto-wah
        >>> wah = apply_autowah(guitar, 44100, sensitivity=0.7, q=10.0)
    
    Note:
        Auto-wah uses the input signal's envelope to control filter frequency.
        Classic funk guitar effect.
    """
    # Extract envelope
    envelope = np.abs(signal.hilbert(y))
    
    # Smooth envelope
    from scipy.ndimage import gaussian_filter1d
    envelope = gaussian_filter1d(envelope, sigma=sr // 100)
    
    # Normalize envelope
    env_max = np.max(envelope)
    if env_max > 1e-9:
        envelope = envelope / env_max
    
    # Map envelope to filter frequency (200 Hz to 2000 Hz)
    freq = 200.0 + sensitivity * 1800.0 * envelope
    
    # Simple time-varying resonant filter (simplified)
    output = np.zeros_like(y)
    
    for i in range(len(y)):
        # Create resonant bandpass at envelope frequency
        if i % 512 == 0:  # Update filter every block
            nyquist = sr / 2.0
            f_norm = min(freq[i] / nyquist, 0.99)
            bw = f_norm / q
            
            try:
                sos = signal.butter(2, [max(f_norm - bw/2, 0.01), min(f_norm + bw/2, 0.99)], 
                                   btype='bandpass', output='sos')
                
                # Process block
                block_end = min(i + 512, len(y))
                output[i:block_end] = signal.sosfilt(sos, y[i:block_end])
            except:
                output[i] = y[i]
        elif i % 512 != 0:
            output[i] = y[i]
    
    # Mix
    result = (1.0 - mix) * y + mix * output
    
    return result.astype(np.float32)


def apply_talkbox(
    y: np.ndarray,
    sr: int,
    formants: List[float] = [700, 1220, 2600],
    wet: float = 0.7
) -> np.ndarray:
    """
    Apply talkbox-style formant filtering.
    
    Args:
        y: Audio time series
        sr: Sample rate
        formants: Formant frequencies in Hz
        wet: Wet/dry mix
    
    Returns:
        Audio with talkbox effect
    
    Example:
        >>> # Robotic voice
        >>> talked = apply_talkbox(synth, 44100, formants=[500, 1500, 2500])
    """
    output = np.zeros_like(y)
    
    # Apply bandpass filter at each formant
    for formant in formants:
        nyquist = sr / 2.0
        f_norm = min(formant / nyquist, 0.99)
        bw = f_norm * 0.1  # 10% bandwidth
        
        sos = signal.butter(2, [max(f_norm - bw, 0.01), min(f_norm + bw, 0.99)],
                           btype='bandpass', output='sos')
        
        output += signal.sosfilt(sos, y) / len(formants)
    
    # Mix
    result = (1.0 - wet) * y + wet * output
    
    return result.astype(np.float32)


def apply_resonant_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 1000.0,
    resonance: float = 5.0,
    filter_type: Literal['lowpass', 'highpass', 'bandpass'] = 'lowpass'
) -> np.ndarray:
    """
    Apply resonant filter (synthesizer-style).
    
    Args:
        y: Audio time series
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz
        resonance: Resonance amount (1-20, higher = more resonant)
        filter_type: Filter type
    
    Returns:
        Filtered audio
    
    Example:
        >>> # Resonant lowpass sweep
        >>> filtered = apply_resonant_filter(synth, 44100, cutoff_hz=500, resonance=10.0)
    """
    nyquist = sr / 2.0
    cutoff_norm = min(cutoff_hz / nyquist, 0.99)
    
    # Calculate bandwidth from resonance
    bw = cutoff_norm / resonance
    
    if filter_type == 'bandpass':
        sos = signal.butter(4, [max(cutoff_norm - bw, 0.01), min(cutoff_norm + bw, 0.99)],
                           btype='bandpass', output='sos')
    elif filter_type == 'highpass':
        sos = signal.butter(4, cutoff_norm, btype='highpass', output='sos')
    else:  # lowpass
        sos = signal.butter(4, cutoff_norm, btype='lowpass', output='sos')
    
    output = signal.sosfilt(sos, y)
    
    return output.astype(np.float32)


# =============================================================================
# CREATIVE/SPECIAL EFFECTS
# =============================================================================

def apply_ring_mod(
    y: np.ndarray,
    sr: int,
    carrier_freq: float = 440.0,
    mix: float = 0.5
) -> np.ndarray:
    """
    Apply ring modulation (metallic/robotic effect).
    
    Args:
        y: Audio time series
        sr: Sample rate
        carrier_freq: Carrier frequency in Hz
        mix: Wet/dry mix
    
    Returns:
        Ring modulated audio
    
    Example:
        >>> # Metallic robot voice
        >>> ringed = apply_ring_mod(vocal, 44100, carrier_freq=200, mix=0.7)
    """
    t = np.arange(len(y)) / sr
    carrier = np.sin(2 * math.pi * carrier_freq * t)
    
    # Ring modulation (multiply signals)
    modulated = y * carrier
    
    # Mix
    output = (1.0 - mix) * y + mix * modulated
    
    return output.astype(np.float32)


def apply_vocoder(
    y: np.ndarray,
    carrier: np.ndarray,
    sr: int,
    bands: int = 16
) -> np.ndarray:
    """
    Apply simple vocoder effect (carrier modulated by input envelope).
    
    Args:
        y: Modulator audio (e.g., voice)
        carrier: Carrier audio (e.g., synth)
        sr: Sample rate
        bands: Number of frequency bands
    
    Returns:
        Vocoded audio
    
    Example:
        >>> # Robot voice
        >>> vocoded = apply_vocoder(voice, synth, 44100, bands=16)
    
    Note:
        This is a simplified vocoder. Production vocoders use more
        sophisticated envelope followers and band processing.
    """
    # Ensure same length
    min_len = min(len(y), len(carrier))
    y = y[:min_len]
    carrier = carrier[:min_len]
    
    output = np.zeros(min_len, dtype=np.float32)
    
    # Split into frequency bands
    freqs = np.logspace(np.log10(100), np.log10(sr/2 - 100), bands + 1)
    
    for i in range(bands):
        low_freq = freqs[i]
        high_freq = freqs[i + 1]
        
        nyquist = sr / 2.0
        low_norm = min(low_freq / nyquist, 0.99)
        high_norm = min(high_freq / nyquist, 0.99)
        
        # Filter modulator and carrier
        sos = signal.butter(2, [low_norm, high_norm], btype='bandpass', output='sos')
        
        mod_band = signal.sosfilt(sos, y)
        car_band = signal.sosfilt(sos, carrier)
        
        # Extract envelope from modulator
        envelope = np.abs(signal.hilbert(mod_band))
        
        # Apply envelope to carrier band
        output += car_band * envelope
    
    # Normalize
    output = output / bands
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_granular(
    y: np.ndarray,
    sr: int,
    grain_size: float = 0.05,
    overlap: float = 0.5,
    pitch_shift: float = 0.0,
    scatter: float = 0.0
) -> np.ndarray:
    """
    Apply granular synthesis (texture/glitch effect).
    
    Args:
        y: Audio time series
        sr: Sample rate
        grain_size: Grain duration in seconds
        overlap: Grain overlap (0-1)
        pitch_shift: Pitch shift in semitones
        scatter: Random grain position scatter (0-1)
    
    Returns:
        Granulated audio
    
    Example:
        >>> # Glitchy texture
        >>> granular = apply_granular(pad, 44100, grain_size=0.03, scatter=0.5)
    """
    grain_samples = int(grain_size * sr)
    hop_samples = int(grain_samples * (1.0 - overlap))
    
    # Pitch shift ratio
    pitch_ratio = 2.0 ** (pitch_shift / 12.0)
    
    output = np.zeros(len(y), dtype=np.float32)
    window = np.hanning(grain_samples)
    
    pos = 0
    while pos + grain_samples < len(y):
        # Add scatter
        scatter_offset = int(scatter * np.random.uniform(-grain_samples, grain_samples))
        read_pos = max(0, min(pos + scatter_offset, len(y) - grain_samples))
        
        # Extract grain
        grain = y[read_pos:read_pos + grain_samples]
        
        # Apply window
        grain = grain * window
        
        # Pitch shift grain (simple resampling)
        if abs(pitch_shift) > 0.1:
            grain_len = int(len(grain) / pitch_ratio)
            grain = np.interp(
                np.linspace(0, len(grain) - 1, grain_len),
                np.arange(len(grain)),
                grain
            )
        
        # Write grain
        write_len = min(len(grain), len(output) - pos)
        output[pos:pos + write_len] += grain[:write_len]
        
        pos += hop_samples
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak * 0.99
    
    return output.astype(np.float32)


def apply_reverse(
    y: np.ndarray,
    reverse_length: Optional[float] = None,
    sr: Optional[int] = None
) -> np.ndarray:
    """
    Apply reverse effect.
    
    Args:
        y: Audio time series
        reverse_length: Optional length in seconds to reverse (None = full track)
        sr: Sample rate (required if reverse_length is specified)
    
    Returns:
        Reversed audio
    
    Example:
        >>> # Reverse entire track
        >>> reversed = apply_reverse(audio)
        >>> # Reverse last 2 seconds
        >>> reversed = apply_reverse(audio, reverse_length=2.0, sr=44100)
    """
    if reverse_length is not None and sr is not None:
        reverse_samples = int(reverse_length * sr)
        reverse_samples = min(reverse_samples, len(y))
        
        # Reverse only the end portion
        output = y.copy()
        output[-reverse_samples:] = output[-reverse_samples:][::-1]
        return output
    else:
        # Reverse entire array
        return y[::-1].copy()


def apply_stutter(
    y: np.ndarray,
    sr: int,
    bpm: float,
    stutter_beats: float = 0.25,
    repeats: int = 4
) -> np.ndarray:
    """
    Apply stutter/glitch effect (repeat short slice).
    
    Args:
        y: Audio time series
        sr: Sample rate
        bpm: Tempo in BPM
        stutter_beats: Stutter length in beats (typically 0.25 or 0.125)
        repeats: Number of repeats
    
    Returns:
        Audio with stutter applied
    
    Example:
        >>> # 16th note stutter (4 repeats)
        >>> stuttered = apply_stutter(audio, 44100, bpm=128, 
        >>>                           stutter_beats=0.25, repeats=4)
    """
    if bpm <= 0:
        return y
    
    beat_duration = 60.0 / bpm
    stutter_duration = beat_duration * stutter_beats
    stutter_samples = int(stutter_duration * sr)
    
    # Extract stutter slice from end
    if len(y) >= stutter_samples:
        slice_audio = y[-stutter_samples:]
    else:
        slice_audio = y
    
    # Create stuttered section
    stuttered = np.tile(slice_audio, repeats)
    
    # Replace end with stutter
    if len(y) >= stutter_samples:
        output = np.concatenate([y[:-stutter_samples], stuttered])
    else:
        output = stuttered
    
    return output.astype(np.float32)


def apply_tapestop(
    y: np.ndarray,
    sr: int,
    stop_duration: float = 1.0
) -> np.ndarray:
    """
    Apply tape stop effect (vinyl/tape slowdown).
    
    Args:
        y: Audio time series
        sr: Sample rate
        stop_duration: Duration of slowdown in seconds
    
    Returns:
        Audio with tape stop applied
    
    Example:
        >>> # Vinyl stop at end
        >>> stopped = apply_tapestop(audio, 44100, stop_duration=1.5)
    """
    stop_samples = int(stop_duration * sr)
    stop_samples = min(stop_samples, len(y) // 2)
    
    if stop_samples < 100:
        return y
    
    # Create speed curve (exponential slowdown)
    t = np.linspace(0, 1, stop_samples)
    speed_curve = np.exp(-5 * t)  # Exponential decay
    
    # Resample with varying speed
    stop_section = y[-stop_samples:]
    output_stop = []
    
    pos = 0.0
    for i in range(stop_samples):
        idx = int(pos)
        if idx < len(stop_section):
            output_stop.append(stop_section[idx])
        pos += speed_curve[i]
    
    output_stop = np.array(output_stop, dtype=np.float32)
    
    # Combine
    output = np.concatenate([y[:-stop_samples], output_stop])
    
    return output.astype(np.float32)


if __name__ == '__main__':
    # Quick test
    print("Effects Manipulations Module")
    print(f"Available functions: {len(__all__)}")
    print("Time-based: reverb, delay, echo, slapback")
    print("Modulation: chorus, flanger, phaser, tremolo, vibrato")
    print("Distortion: distortion, saturation, bitcrush, waveshaper")
    print("Filters: autowah, talkbox, resonant_filter")
    print("Creative: ring_mod, vocoder, granular, reverse, stutter, tapestop")
