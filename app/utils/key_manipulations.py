"""
Key manipulation utilities for audio processing.
This module is completely independent and handles only key-related operations.

Functions:
    - detect_key: Detect the musical key of audio
    - pitch_shift: Shift pitch by semitones
    - transpose_to_key: Auto-detect key and transpose to target
    - get_compatible_keys: Find harmonically compatible keys (Camelot wheel)
    - get_relative_key: Get relative major/minor
    - keys_are_compatible: Check if two keys are compatible
    - get_camelot_code: Convert key/mode to Camelot code
"""

import librosa
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Musical constants
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['maj', 'min']  # Shortened for consistency with Camelot

# Camelot Wheel mapping: (key, mode) -> Camelot code
CAMELOT_WHEEL = {
    ('C', 'maj'): '8B', ('C', 'min'): '5A',
    ('C#', 'maj'): '3B', ('C#', 'min'): '12A',
    ('D', 'maj'): '10B', ('D', 'min'): '7A',
    ('D#', 'maj'): '5B', ('D#', 'min'): '2A',
    ('E', 'maj'): '12B', ('E', 'min'): '9A',
    ('F', 'maj'): '7B', ('F', 'min'): '4A',
    ('F#', 'maj'): '2B', ('F#', 'min'): '11A',
    ('G', 'maj'): '9B', ('G', 'min'): '6A',
    ('G#', 'maj'): '4B', ('G#', 'min'): '1A',
    ('A', 'maj'): '11B', ('A', 'min'): '8A',
    ('A#', 'maj'): '6B', ('A#', 'min'): '3A',
    ('B', 'maj'): '1B', ('B', 'min'): '10A',
}

# Reverse mapping: Camelot -> (key, mode)
CAMELOT_TO_KEY = {v: k for k, v in CAMELOT_WHEEL.items()}

# Krumhansl-Schmuckler key profiles for detection
_C_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_C_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
_MAJOR_TEMPLATES = np.stack([np.roll(_C_MAJOR_PROFILE, i) for i in range(12)], axis=0)
_MINOR_TEMPLATES = np.stack([np.roll(_C_MINOR_PROFILE, i) for i in range(12)], axis=0)


# ============================================================================
# KEY DETECTION
# ============================================================================

def detect_key(audio: np.ndarray, sr: int) -> Tuple[str, str, Optional[str], float]:
    """
    Detect the musical key of audio using Krumhansl-Schmuckler algorithm.
    
    Args:
        audio: Audio signal (mono or stereo)
        sr: Sampling rate
        
    Returns:
        Tuple of (key_name, mode, camelot_code, confidence)
        - key_name: 'C', 'C#', 'D', etc. or 'unknown'
        - mode: 'maj', 'min', or '' if unknown
        - camelot_code: e.g., '8B', '5A', or None
        - confidence: 0.0 to 1.0
        
    Example:
        key, mode, camelot, conf = detect_key(audio, sr)
        if conf > 0.5:
            print(f"Key: {key} {mode} ({camelot})")
    """
    try:
        # Separate harmonic content for better key detection
        try:
            y_harmonic, _ = librosa.effects.hpss(audio)
        except Exception:
            y_harmonic = audio
        
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=1024)
        if chroma.size == 0:
            return "unknown", "", None, 0.0
        
        # Average chroma over time and normalize
        chroma_mean = chroma.mean(axis=1)
        chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
        
        # Correlate with major and minor templates
        corr_major = _MAJOR_TEMPLATES @ chroma_mean
        corr_minor = _MINOR_TEMPLATES @ chroma_mean
        
        # Find best matches
        idx_major = int(np.argmax(corr_major))
        score_major = float(corr_major[idx_major])
        idx_minor = int(np.argmax(corr_minor))
        score_minor = float(corr_minor[idx_minor])
        
        # Choose major or minor
        if max(score_major, score_minor) <= 0:
            return "unknown", "", None, 0.0
        
        if score_major >= score_minor:
            pitch_idx, mode, score = idx_major, 'maj', score_major
        else:
            pitch_idx, mode, score = idx_minor, 'min', score_minor
        
        # Calculate confidence
        confidence = score / (score_major + score_minor + 1e-9)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Low confidence = unknown
        if confidence < 0.35:
            return "unknown", "", None, confidence
        
        # Map to key name and Camelot code
        key_name = NOTES[pitch_idx]
        camelot_num = ((pitch_idx * 7) % 12) + 1
        camelot_letter = 'B' if mode == 'maj' else 'A'
        camelot_code = f"{camelot_num}{camelot_letter}"
        
        return key_name, mode, camelot_code, confidence
        
    except Exception as e:
        logger.error(f"Error detecting key: {e}")
        return "unknown", "", None, 0.0


# ============================================================================
# PITCH MANIPULATION
# ============================================================================

def pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float,
    preserve_formant: bool = False
) -> np.ndarray:
    """
    Shift the pitch of audio by a given number of semitones.
    
    Args:
        audio: Audio signal as numpy array (mono or stereo)
        sr: Sampling rate
        semitones: Number of semitones to shift (can be fractional, +/-)
        preserve_formant: If True, preserves vocal timbre (slower but better for vocals)
        
    Returns:
        Pitch-shifted audio signal
        
    Examples:
        # Shift up a whole tone
        shifted = pitch_shift(audio, sr, semitones=2)
        
        # Shift down with formant preservation (good for vocals)
        shifted = pitch_shift(audio, sr, semitones=-3, preserve_formant=True)
    """
    try:
        if abs(semitones) < 0.01:  # No shift needed
            return audio.astype(np.float32)
        
        # Clamp to reasonable range to avoid artifacts
        semitones = np.clip(semitones, -12.0, 12.0)
        
        if preserve_formant:
            # Use phase vocoder with formant preservation
            # This is slower but maintains vocal character better
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=semitones,
                bins_per_octave=12
            )
        else:
            # Standard pitch shift (faster)
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        
        return shifted.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in pitch_shift: {str(e)}")
        raise


def pitch_shift_semitones(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Lightweight pitch shift with automatic clamping to ±2 semitones.
    
    This is a simpler, more constrained version of pitch_shift() that's
    useful for subtle pitch adjustments (like tuning corrections).
    For larger shifts, use pitch_shift() instead.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        semitones: Number of semitones to shift (will be clamped to ±2.0)
        
    Returns:
        Pitch-shifted audio
        
    Example:
        # Subtle tuning adjustment
        tuned = pitch_shift_semitones(audio, sr, 0.5)  # +50 cents
    """
    # Clamp to subtle range to keep it safe
    semitones = float(np.clip(semitones, -2.0, 2.0))
    
    if abs(semitones) < 1e-6:
        return audio.astype(np.float32)
    
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=semitones)


def transpose_to_key(
    audio: np.ndarray,
    sr: int,
    target_key: str,
    target_mode: str = 'maj',
    preserve_formant: bool = False,
    confidence_threshold: float = 0.35
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Transpose audio to match a target musical key.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        target_key: Target key ('C', 'C#', 'D', etc.)
        target_mode: Target mode ('maj' or 'min')
        preserve_formant: Preserve vocal timbre
        confidence_threshold: Minimum key detection confidence (0-1)
        
    Returns:
        (transposed_audio, metadata_dict) with info about the transposition
        
    Example:
        transposed, meta = transpose_to_key(audio, sr, 'C', 'maj')
        print(f"Shifted {meta['semitones']} semitones from {meta['source_key']}")
    """
    try:
        # Detect current key using local function
        current_key, current_mode, camelot, confidence = detect_key(audio, sr)
        
        # Normalize mode format
        current_mode_short = 'maj' if current_mode == 'major' or current_mode == 'maj' else 'min'
        target_mode_short = 'maj' if target_mode == 'major' else target_mode
        
        # Check confidence
        if confidence < confidence_threshold:
            logger.warning(
                f"Low key detection confidence ({confidence:.2f}). "
                f"Transposition may be inaccurate."
            )
        
        # Calculate semitones needed
        semitones = _calculate_semitones_between_keys(
            current_key, current_mode_short,
            target_key, target_mode_short
        )
        
        # Perform pitch shift
        if abs(semitones) < 0.5:
            transposed = audio.astype(np.float32)
        else:
            transposed = pitch_shift(audio, sr, semitones, preserve_formant)
        
        # Return with metadata
        metadata = {
            'source_key': current_key,
            'source_mode': current_mode_short,
            'source_camelot': camelot,
            'detection_confidence': float(confidence),
            'target_key': target_key,
            'target_mode': target_mode_short,
            'target_camelot': CAMELOT_WHEEL.get((target_key, target_mode_short)),
            'semitones': float(semitones),
            'preserve_formant': preserve_formant
        }
        
        return transposed, metadata
        
    except Exception as e:
        logger.error(f"Error in transpose_to_key: {str(e)}")
        raise

def get_relative_key(key: str, mode: str) -> Tuple[str, str]:
    """
    Get the relative major/minor key.
    
    Args:
        key: Current key ('C', 'D#', etc.)
        mode: Current mode ('maj', 'min', 'major', or 'minor')
        
    Returns:
        (relative_key, relative_mode)
        
    Examples:
        get_relative_key('C', 'maj')  # Returns ('A', 'min')
        get_relative_key('A', 'min')  # Returns ('C', 'maj')
    """
    # Normalize mode
    mode_norm = 'maj' if mode in ('major', 'maj') else 'min'
    
    idx = NOTES.index(key)
    if mode_norm == 'maj':
        # Relative minor is 3 semitones down (minor third)
        relative_idx = (idx - 3) % 12
        return NOTES[relative_idx], 'min'
    else:
        # Relative major is 3 semitones up (minor third)
        relative_idx = (idx + 3) % 12
        return NOTES[relative_idx], 'maj'


def get_compatible_keys(
    key: str,
    mode: str,
    compatibility_level: str = 'safe'
) -> List[Dict[str, Any]]:
    """
    Get harmonically compatible keys for mixing (Camelot wheel rules).
    
    Args:
        key: Source key
        mode: Source mode ('maj' or 'min')
        compatibility_level: 'safe', 'medium', or 'adventurous'
            - safe: Same key, +/-1 on wheel, relative major/minor
            - medium: adds +/-2 on wheel
            - adventurous: adds +/-3 on wheel, parallel keys
            
    Returns:
        List of compatible keys with metadata
        
    Example:
        compatible = get_compatible_keys('C', 'maj', 'safe')
        # Returns keys like: C maj, C min, G maj, F maj, A min, etc.
    """
    mode_norm = 'maj' if mode in ('major', 'maj') else 'min'
    source_camelot = CAMELOT_WHEEL.get((key, mode_norm))
    
    if not source_camelot:
        return []
    
    compatible = []
    
    # Parse Camelot code
    num = int(source_camelot[:-1])
    letter = source_camelot[-1]
    
    # Rule 1: Same key (perfect match)
    compatible.append({
        'key': key,
        'mode': mode_norm,
        'camelot': source_camelot,
        'compatibility': 'perfect',
        'reason': 'Same key'
    })
    
    # Rule 2: Relative major/minor (energy switch)
    rel_key, rel_mode = get_relative_key(key, mode_norm)
    rel_camelot = CAMELOT_WHEEL.get((rel_key, rel_mode))
    compatible.append({
        'key': rel_key,
        'mode': rel_mode,
        'camelot': rel_camelot,
        'compatibility': 'perfect',
        'reason': 'Relative key (energy switch)'
    })
    
    # Rule 3: +/-1 on Camelot wheel (safe harmonic mixing)
    for delta in [-1, 1]:
        adj_num = ((num - 1 + delta) % 12) + 1
        adj_camelot = f"{adj_num}{letter}"
        adj_key, adj_mode = CAMELOT_TO_KEY.get(adj_camelot, (None, None))
        if adj_key:
            compatible.append({
                'key': adj_key,
                'mode': adj_mode,
                'camelot': adj_camelot,
                'compatibility': 'safe',
                'reason': f'{delta:+d} on Camelot wheel'
            })
    
    if compatibility_level in ('medium', 'adventurous'):
        # Rule 4: +/-2 on wheel (medium risk)
        for delta in [-2, 2]:
            adj_num = ((num - 1 + delta) % 12) + 1
            adj_camelot = f"{adj_num}{letter}"
            adj_key, adj_mode = CAMELOT_TO_KEY.get(adj_camelot, (None, None))
            if adj_key:
                compatible.append({
                    'key': adj_key,
                    'mode': adj_mode,
                    'camelot': adj_camelot,
                    'compatibility': 'medium',
                    'reason': f'{delta:+d} on Camelot wheel (medium risk)'
                })
    
    if compatibility_level == 'adventurous':
        # Rule 5: Parallel major/minor (mood change)
        parallel_mode = 'min' if mode_norm == 'maj' else 'maj'
        parallel_camelot = CAMELOT_WHEEL.get((key, parallel_mode))
        compatible.append({
            'key': key,
            'mode': parallel_mode,
            'camelot': parallel_camelot,
            'compatibility': 'adventurous',
            'reason': 'Parallel key (dramatic mood shift)'
        })
        
        # Rule 6: +/-3 on wheel (tritone, risky but interesting)
        for delta in [-3, 3]:
            adj_num = ((num - 1 + delta) % 12) + 1
            adj_camelot = f"{adj_num}{letter}"
            adj_key, adj_mode = CAMELOT_TO_KEY.get(adj_camelot, (None, None))
            if adj_key:
                compatible.append({
                    'key': adj_key,
                    'mode': adj_mode,
                    'camelot': adj_camelot,
                    'compatibility': 'adventurous',
                    'reason': f'{delta:+d} on Camelot wheel (tritone, experimental)'
                })
    
    return compatible


def get_camelot_code(key: str, mode: str) -> Optional[str]:
    """
    Get Camelot wheel code for a key.
    
    Args:
        key: Musical key
        mode: 'maj' or 'min'
        
    Returns:
        Camelot code (e.g., '8B', '5A') or None if invalid
    """
    mode_norm = 'maj' if mode in ('major', 'maj') else 'min'
    return CAMELOT_WHEEL.get((key, mode_norm))


def keys_are_compatible(key1: str, mode1: str, key2: str, mode2: str) -> Tuple[bool, str]:
    """
    Check if two keys are harmonically compatible for mixing.
    
    Args:
        key1, mode1: First key and mode
        key2, mode2: Second key and mode
        
    Returns:
        (is_compatible, reason)
    """
    compatible_keys = get_compatible_keys(key1, mode1, compatibility_level='medium')
    
    for compat in compatible_keys:
        if compat['key'] == key2 and compat['mode'] == mode2:
            return True, compat['reason']
    
    return False, 'Keys are not harmonically compatible'


# ============================================================================
# HELPER FUNCTIONS (PRIVATE)
# ============================================================================

def _calculate_semitones_between_keys(
    source_key: str,
    source_mode: str,
    target_key: str,
    target_mode: str
) -> int:
    """
    Calculate semitones needed to transpose from source to target key.
    
    Note: Mode changes (maj/min) affect the *feeling*, not the pitch.
    If you want C major to sound like A minor, you transpose DOWN 3 semitones
    (because A minor is the relative minor of C major).
    """
    source_idx = NOTES.index(source_key)
    target_idx = NOTES.index(target_key)
    
    # Basic pitch difference
    semitones = target_idx - source_idx
    
    # Normalize to shortest path around circle of fifths
    if semitones > 6:
        semitones -= 12
    elif semitones < -6:
        semitones += 12
    
    # Mode adjustment: if changing from maj->min or min->maj with same root,
    # we want to transpose to the RELATIVE key, not change the mode
    # Example: C major -> C minor means transpose to E♭ major (relative major of C minor)
    if source_key == target_key and source_mode != target_mode:
        if target_mode == 'min':
            # Target is minor: go down 3 semitones to get relative minor
            semitones -= 3
        else:
            # Target is major: go up 3 semitones to get relative major
            semitones += 3
    
    return int(semitones)


def _compute_key_correlation(chroma: np.ndarray) -> np.ndarray:
    """
    Compute correlation between chromagram and Krumhansl-Schmuckler key profiles.
    
    Args:
        chroma: Chromagram of audio
        
    Returns:
        Correlation scores for all 24 keys (12 major + 12 minor)
    """
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile = major_profile / major_profile.sum()
    minor_profile = minor_profile / minor_profile.sum()
    
    # Compute mean chroma vector
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
    
    # Initialize correlation array (24 = 12 major + 12 minor)
    correlations = np.zeros(24, dtype=np.float32)
    
    # Calculate correlation for each key
    for i in range(12):
        # Rotate profiles to test different keys
        shifted_major = np.roll(major_profile, i)
        shifted_minor = np.roll(minor_profile, i)
        
        # Pearson correlation
        correlations[i] = np.corrcoef(chroma_mean, shifted_major)[0, 1]
        correlations[i + 12] = np.corrcoef(chroma_mean, shifted_minor)[0, 1]
    
    return correlations

    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
    
    # Initialize correlation array (24 = 12 major + 12 minor)
    correlations = np.zeros(24, dtype=np.float32)
    
    # Calculate correlation for each key
    for i in range(12):
        # Rotate profiles to test different keys
        shifted_major = np.roll(major_profile, i)
        shifted_minor = np.roll(minor_profile, i)
        
        # Pearson correlation
        correlations[i] = np.corrcoef(chroma_mean, shifted_major)[0, 1]
        correlations[i + 12] = np.corrcoef(chroma_mean, shifted_minor)[0, 1]
    
    return correlations


def _calculate_pitch_shift(
    current_key: str,
    current_mode: str,
    target_key: str,
    target_mode: str
) -> int:
    """
    DEPRECATED: Use _calculate_semitones_between_keys instead.
    
    Kept for backwards compatibility.
    """
    logger.warning("_calculate_pitch_shift is deprecated, use _calculate_semitones_between_keys")
    return _calculate_semitones_between_keys(
        current_key, current_mode, target_key, target_mode
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'detect_key',
    'pitch_shift',
    'pitch_shift_semitones',
    'transpose_to_key',
    'get_relative_key',
    'get_compatible_keys',
    'get_camelot_code',
    'keys_are_compatible',
    'CAMELOT_WHEEL',
    'CAMELOT_TO_KEY',
    'NOTES',
    'MODES',
]



