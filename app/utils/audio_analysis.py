"""
Audio Analysis & Feature Extraction for Intelligent Mashup Creation

This module extracts comprehensive audio features to enable intelligent
mashup decisions. Acts as the "brain" that analyzes tracks and suggests
optimal mixing strategies.

Core Analysis:
- analyze_track: Complete audio fingerprint (all features)
- compare_tracks: Similarity scoring for mashup compatibility
- find_mix_points: Suggest optimal transition points

Spectral Features:
- get_spectral_features: Rolloff, flux, contrast, bandwidth
- get_frequency_balance: Bass/mid/high energy distribution
- get_brightness: Spectral centroid analysis

Harmonic Features:
- get_harmonic_features: Harmonicity, tonal vs noise ratio
- get_pitch_features: Pitch range, stability
- estimate_chords: Basic chord detection

Rhythm Features:
- get_rhythm_features: Onset rate, tempo stability, groove strength
- get_beat_consistency: How consistent beats are
- detect_syncopation: Rhythmic complexity

Structural Features:
- detect_energy_profile: Track energy over time
- find_transitions: Detect energy changes/transitions
- estimate_sections: Rough intro/verse/chorus detection

Stereo Features:
- get_stereo_features: Width, correlation, balance
- analyze_spatial_field: Stereo distribution

Comparison & Matching:
- calculate_similarity: Overall similarity score
- match_features: Feature-by-feature comparison
- suggest_mashup_strategy: Intelligent mixing recommendations

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import math
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import librosa

logger = logging.getLogger(__name__)

__all__ = [
    'analyze_track',
    'compare_tracks',
    'find_mix_points',
    'get_spectral_features',
    'get_frequency_balance',
    'get_brightness',
    'get_harmonic_features',
    'get_pitch_features',
    'estimate_chords',
    'get_rhythm_features',
    'get_beat_consistency',
    'detect_syncopation',
    'detect_energy_profile',
    'find_transitions',
    'estimate_sections',
    'get_stereo_features',
    'analyze_spatial_field',
    'calculate_similarity',
    'match_features',
    'suggest_mashup_strategy',
]


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_track(
    y: np.ndarray,
    sr: int,
    include_advanced: bool = True
) -> Dict[str, Any]:
    """
    Complete audio fingerprint extraction (all features).
    
    Args:
        y: Audio time series (mono)
        sr: Sample rate
        include_advanced: If True, compute expensive features (chords, etc.)
    
    Returns:
        Dictionary with all audio features:
        - Basic: duration, sample_rate
        - Temporal: bpm, beat_count, tempo_stability
        - Spectral: brightness, rolloff, flux, contrast, bandwidth
        - Harmonic: key, mode, harmonicity, pitch_range
        - Energy: rms, peak, crest_factor, dynamic_range
        - Frequency: bass_energy, mid_energy, high_energy
        - Rhythm: onset_rate, beat_strength, syncopation
        - Stereo: width, correlation (if stereo)
        - Structure: energy_profile, transitions
    
    Example:
        >>> features = analyze_track(audio, 44100)
        >>> print(f"Key: {features['key']} {features['mode']}")
        >>> print(f"BPM: {features['bpm']:.1f}")
        >>> print(f"Brightness: {features['brightness']:.0f} Hz")
    
    Note:
        This is the main entry point for audio analysis.
        Use this to build a feature database for intelligent mashup matching.
    """
    logger.info("Starting comprehensive track analysis...")
    
    features = {}
    
    # Basic info
    features['duration'] = len(y) / sr
    features['sample_rate'] = sr
    features['samples'] = len(y)
    
    # Temporal features (BPM, beats)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        features['bpm'] = float(tempo)
        features['beat_count'] = len(beats)
        
        # Tempo stability
        if len(beats) > 2:
            intervals = np.diff(beats)
            features['tempo_stability'] = float(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-9))
        else:
            features['tempo_stability'] = 0.5
    except Exception as e:
        logger.warning(f"Tempo detection failed: {e}")
        features['bpm'] = 120.0
        features['beat_count'] = 0
        features['tempo_stability'] = 0.0
    
    # Spectral features
    spectral = get_spectral_features(y, sr)
    features.update(spectral)
    
    # Frequency balance
    freq_balance = get_frequency_balance(y, sr)
    features.update(freq_balance)
    
    # Harmonic features
    harmonic = get_harmonic_features(y, sr)
    features.update(harmonic)
    
    # Energy features
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    features['rms'] = float(rms)
    features['rms_db'] = float(20.0 * np.log10(rms + 1e-9))
    features['peak'] = float(peak)
    features['peak_db'] = float(20.0 * np.log10(peak + 1e-9))
    features['crest_factor'] = float(peak / (rms + 1e-9))
    features['crest_factor_db'] = features['peak_db'] - features['rms_db']
    
    # Dynamic range
    rms_frames = librosa.feature.rms(y=y)[0]
    if len(rms_frames) > 0:
        loud_db = 20.0 * np.log10(np.percentile(rms_frames, 95) + 1e-9)
        quiet_db = 20.0 * np.log10(np.percentile(rms_frames, 5) + 1e-9)
        features['dynamic_range_db'] = float(loud_db - quiet_db)
    else:
        features['dynamic_range_db'] = 0.0
    
    # Rhythm features
    rhythm = get_rhythm_features(y, sr)
    features.update(rhythm)
    
    # Stereo features (if applicable)
    if y.ndim > 1:
        stereo = get_stereo_features(y, sr)
        features.update(stereo)
    else:
        features['is_stereo'] = False
        features['stereo_width'] = 0.0
        features['stereo_correlation'] = 1.0
    
    # Energy profile and structure
    energy_prof = detect_energy_profile(y, sr)
    features['energy_profile'] = energy_prof
    
    transitions = find_transitions(y, sr)
    features['transition_count'] = len(transitions)
    features['transitions'] = transitions
    
    # Advanced features (optional, more expensive)
    if include_advanced:
        try:
            sections = estimate_sections(y, sr)
            features['sections'] = sections
        except Exception as e:
            logger.warning(f"Section detection failed: {e}")
            features['sections'] = []
        
        try:
            if features.get('harmonicity', 0) > 0.3:
                chords = estimate_chords(y, sr)
                features['chord_changes'] = len(chords)
                features['chords'] = chords
            else:
                features['chord_changes'] = 0
                features['chords'] = []
        except Exception as e:
            logger.warning(f"Chord detection failed: {e}")
            features['chord_changes'] = 0
            features['chords'] = []
    
    logger.info(f"Analysis complete: {features['bpm']:.1f} BPM, {features['key']} {features['mode']}")
    
    # Convert numpy arrays to lists for JSON serialization
    features = _make_json_serializable(features)
    
    return features


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and other non-serializable types to JSON-compatible types.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
    
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def compare_tracks(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compare two tracks for mashup compatibility.
    
    Args:
        features_a: Features from first track (from analyze_track)
        features_b: Features from second track
        weights: Optional feature weights (default: equal weights)
    
    Returns:
        Dictionary with comparison results:
        - overall_similarity: 0-1 score (higher = more compatible)
        - bpm_compatible: Boolean
        - key_compatible: Boolean
        - timbre_similarity: 0-1 (spectral similarity)
        - energy_match: 0-1 (similar loudness/dynamics)
        - recommendations: List of suggestions
    
    Example:
        >>> feat_a = analyze_track(track1, 44100)
        >>> feat_b = analyze_track(track2, 44100)
        >>> comparison = compare_tracks(feat_a, feat_b)
        >>> if comparison['overall_similarity'] > 0.7:
        >>>     print("Great mashup potential!")
    """
    if weights is None:
        weights = {
            'bpm': 0.25,
            'key': 0.20,
            'timbre': 0.20,
            'energy': 0.15,
            'rhythm': 0.10,
            'frequency': 0.10,
        }
    
    scores = {}
    recommendations = []
    
    # BPM compatibility
    bpm_a = features_a.get('bpm', 120)
    bpm_b = features_b.get('bpm', 120)
    bpm_ratio = max(bpm_a, bpm_b) / (min(bpm_a, bpm_b) + 1e-9)
    
    if abs(bpm_a - bpm_b) < 3:
        scores['bpm'] = 1.0
        scores['bpm_compatible'] = True
    elif abs(bpm_a - bpm_b/2) < 3 or abs(bpm_a - bpm_b*2) < 3:
        scores['bpm'] = 0.9
        scores['bpm_compatible'] = True
        recommendations.append("BPMs are half/double-time related")
    elif abs(bpm_a - bpm_b) < 10:
        scores['bpm'] = 0.7
        scores['bpm_compatible'] = True
    else:
        scores['bpm'] = max(0.0, 1.0 - abs(bpm_a - bpm_b) / 50.0)
        scores['bpm_compatible'] = False
        recommendations.append(f"Large BPM difference: {abs(bpm_a - bpm_b):.1f} BPM")
    
    # Key compatibility (simplified - in reality would use Camelot wheel)
    key_a = features_a.get('key', 'unknown')
    key_b = features_b.get('key', 'unknown')
    
    if key_a == key_b:
        scores['key'] = 1.0
        scores['key_compatible'] = True
    elif key_a == 'unknown' or key_b == 'unknown':
        scores['key'] = 0.5
        scores['key_compatible'] = False
        recommendations.append("Key detection uncertain")
    else:
        scores['key'] = 0.3
        scores['key_compatible'] = False
        recommendations.append(f"Different keys: {key_a} vs {key_b} - consider transposing")
    
    # Timbre similarity (spectral features)
    brightness_a = features_a.get('brightness', 2000)
    brightness_b = features_b.get('brightness', 2000)
    brightness_diff = abs(brightness_a - brightness_b) / max(brightness_a, brightness_b)
    scores['timbre'] = float(1.0 - brightness_diff)
    
    if scores['timbre'] > 0.8:
        recommendations.append("Similar timbres - will blend smoothly")
    elif scores['timbre'] < 0.5:
        recommendations.append("Very different timbres - consider EQ matching")
    
    # Energy match
    rms_a = features_a.get('rms_db', -20)
    rms_b = features_b.get('rms_db', -20)
    energy_diff = abs(rms_a - rms_b)
    scores['energy'] = float(max(0.0, 1.0 - energy_diff / 20.0))
    
    if energy_diff > 6:
        recommendations.append(f"Loudness difference: {energy_diff:.1f} dB - normalize volumes")
    
    # Rhythm similarity
    onset_a = features_a.get('onset_rate', 0)
    onset_b = features_b.get('onset_rate', 0)
    if onset_a > 0 and onset_b > 0:
        rhythm_ratio = min(onset_a, onset_b) / max(onset_a, onset_b)
        scores['rhythm'] = float(rhythm_ratio)
    else:
        scores['rhythm'] = 0.5
    
    # Frequency balance (complementary is good!)
    bass_a = features_a.get('bass_energy', 0.33)
    bass_b = features_b.get('bass_energy', 0.33)
    
    # Complementary bass is actually GOOD for mashups
    bass_complement = 1.0 - abs(bass_a - bass_b)
    scores['frequency'] = float(bass_complement)
    
    if bass_a > 0.5 and bass_b < 0.3:
        recommendations.append("Perfect: Track A is bass-heavy, Track B is bass-light!")
    elif bass_a < 0.3 and bass_b > 0.5:
        recommendations.append("Perfect: Track B is bass-heavy, Track A is bass-light!")
    elif bass_a > 0.5 and bass_b > 0.5:
        recommendations.append("Both tracks bass-heavy - consider hi-pass filtering one")
    
    # Calculate weighted overall similarity
    overall = sum(scores.get(k, 0.5) * weights.get(k, 0) 
                  for k in weights.keys())
    
    result = {
        'overall_similarity': float(overall),
        'component_scores': scores,
        'recommendations': recommendations,
        'bpm_difference': float(abs(bpm_a - bpm_b)),
        'energy_difference_db': float(energy_diff),
        'mashup_potential': 'excellent' if overall > 0.8 else 
                          'good' if overall > 0.6 else
                          'fair' if overall > 0.4 else 'challenging'
    }
    
    return result


def find_mix_points(
    y: np.ndarray,
    sr: int,
    num_points: int = 3,
    prefer_low_energy: bool = True
) -> List[Dict[str, float]]:
    """
    Suggest optimal transition/mix points in a track.
    
    Args:
        y: Audio time series
        sr: Sample rate
        num_points: Number of mix points to return
        prefer_low_energy: If True, prefer quieter sections (easier transitions)
    
    Returns:
        List of mix points with timing and confidence:
        [{'time': seconds, 'beat': beat_index, 'confidence': 0-1}, ...]
    
    Example:
        >>> mix_points = find_mix_points(audio, 44100, num_points=5)
        >>> for point in mix_points:
        >>>     print(f"Mix at {point['time']:.1f}s (confidence: {point['confidence']:.2f})")
    """
    transitions = find_transitions(y, sr)
    
    # Get energy profile
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    # Find beats
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    except:
        beats = np.arange(0, len(y)/sr, 0.5)  # Fallback
    
    mix_points = []
    
    # Score each beat position
    for i, beat_time in enumerate(beats[4:-4]):  # Skip beginning and end
        # Calculate confidence based on:
        # 1. Low energy (if preferred)
        # 2. Proximity to transitions
        # 3. On-beat timing
        
        # Energy score
        time_idx = np.searchsorted(times, beat_time)
        time_idx = min(time_idx, len(rms) - 1)
        energy = rms[time_idx]
        
        if prefer_low_energy:
            energy_score = 1.0 - (energy / (np.max(rms) + 1e-9))
        else:
            energy_score = energy / (np.max(rms) + 1e-9)
        
        # Transition proximity score
        trans_score = 0.0
        for trans in transitions:
            time_diff = abs(beat_time - trans['time'])
            if time_diff < 2.0:  # Within 2 seconds
                trans_score = max(trans_score, 1.0 - time_diff / 2.0)
        
        # On-beat bonus
        beat_score = 1.0 if i % 4 == 0 else 0.7  # Prefer downbeats
        
        # Combined confidence
        confidence = (energy_score * 0.4 + trans_score * 0.4 + beat_score * 0.2)
        
        mix_points.append({
            'time': float(beat_time),
            'beat': int(i),
            'confidence': float(confidence),
            'energy': float(energy)
        })
    
    # Sort by confidence and return top N
    mix_points.sort(key=lambda x: x['confidence'], reverse=True)
    
    return mix_points[:num_points]


# =============================================================================
# SPECTRAL FEATURES
# =============================================================================

def get_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract spectral features (timbre characteristics).
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with:
        - brightness: Mean spectral centroid (Hz)
        - brightness_std: Variability in brightness
        - rolloff: Mean spectral rolloff (85% energy point)
        - flux: Mean spectral flux (how quickly spectrum changes)
        - contrast: Mean spectral contrast (peaks vs valleys)
        - bandwidth: Mean spectral bandwidth
    """
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    
    # Spectral flux (rate of spectral change)
    spec = np.abs(librosa.stft(y))
    flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    return {
        'brightness': float(np.mean(centroid)),
        'brightness_std': float(np.std(centroid)),
        'rolloff': float(np.mean(rolloff)),
        'rolloff_std': float(np.std(rolloff)),
        'flux': float(np.mean(flux)),
        'contrast': float(np.mean(contrast)),
        'contrast_std': float(np.std(contrast)),
        'bandwidth': float(np.mean(bandwidth)),
    }


def get_frequency_balance(
    y: np.ndarray,
    sr: int,
    bass_cutoff: float = 200.0,
    high_cutoff: float = 2000.0
) -> Dict[str, float]:
    """
    Analyze bass/mid/high frequency energy distribution.
    
    Args:
        y: Audio time series
        sr: Sample rate
        bass_cutoff: Bass-mid crossover frequency (Hz)
        high_cutoff: Mid-high crossover frequency (Hz)
    
    Returns:
        Dictionary with:
        - bass_energy: 0-1 (proportion of energy in bass)
        - mid_energy: 0-1 (proportion in mids)
        - high_energy: 0-1 (proportion in highs)
        - bass_db: Bass level in dB
        - mid_db: Mid level in dB
        - high_db: High level in dB
    """
    from scipy import signal as scipy_signal
    
    nyquist = sr / 2.0
    
    # Design filters
    sos_low = scipy_signal.butter(4, bass_cutoff / nyquist, btype='lowpass', output='sos')
    sos_mid = scipy_signal.butter(4, [bass_cutoff / nyquist, high_cutoff / nyquist], 
                                  btype='bandpass', output='sos')
    sos_high = scipy_signal.butter(4, high_cutoff / nyquist, btype='highpass', output='sos')
    
    # Apply filters
    y_low = scipy_signal.sosfilt(sos_low, y)
    y_mid = scipy_signal.sosfilt(sos_mid, y)
    y_high = scipy_signal.sosfilt(sos_high, y)
    
    # Calculate RMS energy
    rms_low = np.sqrt(np.mean(y_low**2))
    rms_mid = np.sqrt(np.mean(y_mid**2))
    rms_high = np.sqrt(np.mean(y_high**2))
    
    total_energy = rms_low + rms_mid + rms_high + 1e-9
    
    return {
        'bass_energy': float(rms_low / total_energy),
        'mid_energy': float(rms_mid / total_energy),
        'high_energy': float(rms_high / total_energy),
        'bass_db': float(20.0 * np.log10(rms_low + 1e-9)),
        'mid_db': float(20.0 * np.log10(rms_mid + 1e-9)),
        'high_db': float(20.0 * np.log10(rms_high + 1e-9)),
    }


def get_brightness(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Get detailed brightness analysis.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with brightness metrics
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    return {
        'brightness_mean': float(np.mean(centroid)),
        'brightness_std': float(np.std(centroid)),
        'brightness_min': float(np.min(centroid)),
        'brightness_max': float(np.max(centroid)),
        'brightness_range': float(np.max(centroid) - np.min(centroid)),
    }


# =============================================================================
# HARMONIC FEATURES
# =============================================================================

def get_harmonic_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract harmonic/tonal features.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with:
        - key: Detected key (e.g., 'C', 'F#')
        - mode: 'major' or 'minor'
        - harmonicity: 0-1 (how tonal vs noisy)
        - key_confidence: 0-1 (key detection confidence)
    """
    # Detect key (simplified - copy from key_manipulations logic)
    try:
        # Separate harmonic and percussive
        y_harm, y_perc = librosa.effects.hpss(y)
        
        # Calculate harmonicity (harmonic vs percussive energy)
        harm_energy = np.sqrt(np.mean(y_harm**2))
        perc_energy = np.sqrt(np.mean(y_perc**2))
        total_energy = harm_energy + perc_energy + 1e-9
        harmonicity = harm_energy / total_energy
        
        # Detect key using chroma
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=1024)
        if chroma.size > 0:
            chroma_mean = chroma.mean(axis=1)
            chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
            
            # Simplified key detection
            pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = int(np.argmax(chroma_mean))
            key = pitches[key_idx]
            
            # Rough major/minor estimation
            minor_idx = (key_idx + 3) % 12  # Relative minor
            if chroma_mean[minor_idx] > chroma_mean[key_idx] * 0.8:
                mode = 'minor'
            else:
                mode = 'major'
            
            key_confidence = float(np.max(chroma_mean))
        else:
            key = 'unknown'
            mode = ''
            key_confidence = 0.0
        
    except Exception as e:
        logger.warning(f"Harmonic analysis failed: {e}")
        key = 'unknown'
        mode = ''
        harmonicity = 0.5
        key_confidence = 0.0
    
    return {
        'key': key,
        'mode': mode,
        'harmonicity': float(harmonicity),
        'key_confidence': float(key_confidence),
    }


def get_pitch_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze pitch characteristics.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with pitch statistics
    """
    try:
        # Extract pitch using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filter to voiced regions
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            return {
                'pitch_mean': float(np.nanmean(f0_voiced)),
                'pitch_std': float(np.nanstd(f0_voiced)),
                'pitch_min': float(np.nanmin(f0_voiced)),
                'pitch_max': float(np.nanmax(f0_voiced)),
                'pitch_range': float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced)),
                'voiced_ratio': float(np.sum(voiced_flag) / len(voiced_flag)),
            }
        else:
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0,
                'pitch_range': 0.0,
                'voiced_ratio': 0.0,
            }
    except Exception as e:
        logger.warning(f"Pitch analysis failed: {e}")
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_min': 0.0,
            'pitch_max': 0.0,
            'pitch_range': 0.0,
            'voiced_ratio': 0.0,
        }


def estimate_chords(
    y: np.ndarray,
    sr: int,
    hop_length: int = 4096
) -> List[Dict[str, Any]]:
    """
    Basic chord progression estimation.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length for analysis
    
    Returns:
        List of chord changes: [{'time': seconds, 'chord': 'C'}, ...]
    
    Note:
        This is a simplified chord detection. For production use,
        consider using a dedicated chord recognition library.
    """
    try:
        # Extract chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        
        # Simple chord templates (major triads only)
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        chords = []
        prev_chord = None
        
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            # Find dominant pitch
            root = int(np.argmax(frame_chroma))
            chord_name = pitches[root]
            
            # Detect chord changes
            if chord_name != prev_chord:
                time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                chords.append({
                    'time': float(time),
                    'chord': chord_name,
                })
                prev_chord = chord_name
        
        return chords
    
    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")
        return []


# =============================================================================
# RHYTHM FEATURES
# =============================================================================

def get_rhythm_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract rhythm/groove characteristics.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary with:
        - onset_rate: Onsets per second
        - beat_strength_mean: Average beat strength
        - beat_strength_std: Beat strength variability
        - groove_strength: 0-1 (rhythmic consistency)
    """
    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512
    )
    
    duration = len(y) / sr
    onset_rate = len(onset_frames) / duration if duration > 0 else 0
    
    # Beat strength
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units='frames')
        if len(beats) > 0:
            beat_strengths = onset_env[np.clip(beats, 0, len(onset_env)-1)]
            beat_strength_mean = float(np.mean(beat_strengths))
            beat_strength_std = float(np.std(beat_strengths))
        else:
            beat_strength_mean = 0.0
            beat_strength_std = 0.0
    except:
        beat_strength_mean = 0.0
        beat_strength_std = 0.0
    
    # Groove strength (consistency of beats)
    groove_strength = get_beat_consistency(y, sr)
    
    return {
        'onset_rate': float(onset_rate),
        'beat_strength_mean': float(beat_strength_mean),
        'beat_strength_std': float(beat_strength_std),
        'groove_strength': float(groove_strength),
    }


def get_beat_consistency(y: np.ndarray, sr: int) -> float:
    """
    Measure how consistent/tight the beats are (0-1).
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Consistency score (1.0 = perfect groove, 0.0 = no consistent beats)
    """
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        
        if len(beats) < 3:
            return 0.0
        
        # Analyze inter-beat intervals
        intervals = np.diff(beats)
        
        if len(intervals) == 0:
            return 0.0
        
        # Low variance = high consistency
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval < 1e-9:
            return 0.0
        
        consistency = 1.0 - min(std_interval / mean_interval, 1.0)
        
        return float(consistency)
    
    except Exception as e:
        logger.warning(f"Beat consistency calculation failed: {e}")
        return 0.0


def detect_syncopation(y: np.ndarray, sr: int) -> float:
    """
    Detect rhythmic syncopation/complexity (0-1).
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Syncopation score (higher = more syncopated/complex)
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=512, units='frames'
        )
        
        if len(beats) < 2:
            return 0.0
        
        # Detect onsets between beats (off-beat onsets = syncopation)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512
        )
        
        off_beat_count = 0
        for onset in onset_frames:
            # Check if onset is NOT on a beat
            min_dist = np.min(np.abs(beats - onset))
            if min_dist > 2:  # Threshold in frames
                off_beat_count += 1
        
        syncopation = off_beat_count / (len(onset_frames) + 1)
        
        return float(min(syncopation, 1.0))
    
    except Exception as e:
        logger.warning(f"Syncopation detection failed: {e}")
        return 0.0


# =============================================================================
# STRUCTURAL FEATURES
# =============================================================================

def detect_energy_profile(
    y: np.ndarray,
    sr: int,
    resolution: float = 1.0
) -> np.ndarray:
    """
    Get energy profile over time (for visualizing track structure).
    
    Args:
        y: Audio time series
        sr: Sample rate
        resolution: Time resolution in seconds
    
    Returns:
        Array of RMS values over time
    """
    hop_length = int(resolution * sr)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    return rms.astype(np.float32)


def find_transitions(
    y: np.ndarray,
    sr: int,
    sensitivity: float = 1.5
) -> List[Dict[str, float]]:
    """
    Detect energy transitions (build-ups, drops, breaks).
    
    Args:
        y: Audio time series
        sr: Sample rate
        sensitivity: Detection sensitivity (higher = more transitions)
    
    Returns:
        List of transitions: [{'time': seconds, 'magnitude': change}, ...]
    """
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    # Smooth RMS
    from scipy.ndimage import gaussian_filter1d
    rms_smooth = gaussian_filter1d(rms, sigma=5)
    
    # Find changes
    rms_diff = np.diff(rms_smooth)
    
    # Threshold for significant changes
    threshold = sensitivity * np.std(rms_diff)
    
    transitions = []
    for i in range(1, len(rms_diff)):
        if abs(rms_diff[i]) > threshold:
            transitions.append({
                'time': float(times[i]),
                'magnitude': float(rms_diff[i]),
                'type': 'buildup' if rms_diff[i] > 0 else 'drop'
            })
    
    return transitions


def estimate_sections(
    y: np.ndarray,
    sr: int
) -> List[Dict[str, Any]]:
    """
    Intelligent musical structure detection using beat analysis and repetition finding.
    Creates 8-10 meaningful sections (intro/verse/chorus/bridge/outro) instead of just 3.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        List of sections: [{'start': s, 'end': s, 'type': 'intro', 'confidence': 0.8}, ...]
    """
    try:
        duration = len(y) / sr
        
        # Get beats for alignment
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Calculate features for section detection
        # 1. RMS energy for intro/outro detection
        rms = librosa.feature.rms(y=y, hop_length=2048)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=2048)
        from scipy.ndimage import gaussian_filter1d
        rms_smooth = gaussian_filter1d(rms, sigma=10)
        
        # 2. Chroma features for finding repeated sections (choruses)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        
        # 3. Self-similarity matrix to detect repetitions
        # Use smaller hop for better resolution
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=1024)
        similarity = np.dot(mfcc.T, mfcc) / (np.linalg.norm(mfcc, axis=0)[:, None] * np.linalg.norm(mfcc, axis=0)[None, :] + 1e-8)
        
        sections = []
        
        # STEP 1: Detect intro (first 10-30 seconds with lower energy or buildup)
        intro_candidates = beat_times[beat_times < min(duration * 0.15, 30.0)]
        if len(intro_candidates) > 0:
            intro_end_idx = np.searchsorted(times, intro_candidates[-1])
            intro_rms = rms_smooth[:intro_end_idx]
            avg_rms = np.mean(rms_smooth)
            
            # Intro if energy builds up or is consistently lower
            if len(intro_rms) > 0 and (np.mean(intro_rms) < avg_rms * 0.85 or 
                                        (intro_rms[-1] > intro_rms[0] * 1.3)):  # Buildup
                intro_end = float(intro_candidates[-1])
                sections.append({
                    'start': 0.0,
                    'end': intro_end,
                    'type': 'intro',
                    'confidence': 0.85
                })
        
        # STEP 2: Detect outro (last 10-30 seconds with fade or lower energy)
        outro_candidates = beat_times[beat_times > max(duration * 0.85, duration - 30.0)]
        if len(outro_candidates) > 0:
            outro_start_idx = np.searchsorted(times, outro_candidates[0])
            outro_rms = rms_smooth[outro_start_idx:]
            avg_rms = np.mean(rms_smooth)
            
            # Outro if energy fades or is consistently lower
            if len(outro_rms) > 0 and (np.mean(outro_rms) < avg_rms * 0.85 or
                                        (outro_rms[-1] < outro_rms[0] * 0.7)):  # Fade
                outro_start = float(outro_candidates[0])
                sections.append({
                    'start': outro_start,
                    'end': float(duration),
                    'type': 'outro',
                    'confidence': 0.85
                })
        
        # STEP 3: Divide middle section into 6-8 segments using beats
        mid_start = sections[0]['end'] if sections else 0.0
        mid_end = sections[-1]['start'] if len(sections) > 0 and sections[-1]['type'] == 'outro' else duration
        mid_duration = mid_end - mid_start
        
        # Target 15-30 second sections (typical verse/chorus length)
        target_section_length = 20.0  # seconds
        num_sections = max(6, min(8, int(mid_duration / target_section_length)))
        
        # Get beats in the middle section
        mid_beats = beat_times[(beat_times >= mid_start) & (beat_times < mid_end)]
        
        if len(mid_beats) > num_sections:
            # Divide into equal beat groups
            beats_per_section = len(mid_beats) // num_sections
            section_boundaries = [mid_beats[i * beats_per_section] for i in range(num_sections)]
            section_boundaries.append(mid_end)
            
            # STEP 4: Use self-similarity to label sections (find repeated patterns = chorus)
            chorus_indices = []
            for i in range(len(section_boundaries) - 1):
                start_time = section_boundaries[i]
                end_time = section_boundaries[i + 1]
                
                # Get similarity scores for this section vs all others
                start_frame = int(start_time * sr / 1024)
                end_frame = int(end_time * sr / 1024)
                
                if start_frame < similarity.shape[0] and end_frame < similarity.shape[0]:
                    section_similarity = similarity[start_frame:end_frame, :]
                    
                    # Check for high similarity with other sections (repeated = likely chorus)
                    avg_similarity_with_others = []
                    for j in range(len(section_boundaries) - 1):
                        if i != j:
                            other_start = int(section_boundaries[j] * sr / 1024)
                            other_end = int(section_boundaries[j + 1] * sr / 1024)
                            if other_start < similarity.shape[1] and other_end < similarity.shape[1]:
                                similarity_score = np.mean(section_similarity[:, other_start:other_end])
                                avg_similarity_with_others.append(similarity_score)
                    
                    if len(avg_similarity_with_others) > 0:
                        avg_sim = np.mean(avg_similarity_with_others)
                        if avg_sim > 0.7:  # High repetition = chorus
                            chorus_indices.append(i)
            
            # STEP 5: Label sections intelligently
            section_labels = []
            for i in range(len(section_boundaries) - 1):
                if i in chorus_indices:
                    section_labels.append('chorus')
                elif i == 0:
                    section_labels.append('verse')  # First section after intro
                elif i == len(section_boundaries) - 2:
                    section_labels.append('verse')  # Last section before outro
                elif len(chorus_indices) > 0 and i > max(chorus_indices):
                    # After last chorus = bridge or final verse
                    section_labels.append('bridge' if i == max(chorus_indices) + 1 else 'verse')
                else:
                    section_labels.append('verse')  # Default to verse
            
            # Create section objects
            for i in range(len(section_boundaries) - 1):
                sections.append({
                    'start': float(section_boundaries[i]),
                    'end': float(section_boundaries[i + 1]),
                    'type': section_labels[i],
                    'confidence': 0.75
                })
        else:
            # Fallback: just create 3 equal sections if not enough beats
            third = mid_duration / 3
            sections.extend([
                {'start': float(mid_start), 'end': float(mid_start + third), 'type': 'verse', 'confidence': 0.6},
                {'start': float(mid_start + third), 'end': float(mid_start + 2*third), 'type': 'chorus', 'confidence': 0.6},
                {'start': float(mid_start + 2*third), 'end': float(mid_end), 'type': 'verse', 'confidence': 0.6}
            ])
        
        # Sort by start time
        sections.sort(key=lambda x: x['start'])
        
        # Final validation: ensure no overlaps and no gaps
        for i in range(len(sections) - 1):
            if sections[i]['end'] > sections[i + 1]['start']:
                sections[i]['end'] = sections[i + 1]['start']
        
        logger.info(f"Detected {len(sections)} sections: {[s['type'] for s in sections]}")
        return sections
    
    except Exception as e:
        logger.warning(f"Section estimation failed: {e}, using fallback")
        # Fallback: simple 3-section structure
        duration = len(y) / sr
        return [
            {'start': 0.0, 'end': min(20.0, duration * 0.15), 'type': 'intro', 'confidence': 0.5},
            {'start': min(20.0, duration * 0.15), 'end': max(duration - 20.0, duration * 0.85), 'type': 'verse', 'confidence': 0.5},
            {'start': max(duration - 20.0, duration * 0.85), 'end': float(duration), 'type': 'outro', 'confidence': 0.5}
        ]


# =============================================================================
# STEREO FEATURES
# =============================================================================

def get_stereo_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze stereo field characteristics.
    
    Args:
        y: Stereo audio [2, n_samples] or [n_samples, 2]
        sr: Sample rate
    
    Returns:
        Dictionary with:
        - is_stereo: True
        - stereo_width: 0-1 (0=mono, 1=wide)
        - stereo_correlation: -1 to 1 (1=mono, 0=decorrelated, -1=out-of-phase)
        - left_right_balance: -1 to 1 (-1=all left, 0=centered, 1=all right)
    """
    # Handle input shape
    if y.ndim == 1:
        return {
            'is_stereo': False,
            'stereo_width': 0.0,
            'stereo_correlation': 1.0,
            'left_right_balance': 0.0,
        }
    
    if y.shape[0] == 2:
        left, right = y[0], y[1]
    else:
        left, right = y[:, 0], y[:, 1]
    
    # Calculate correlation
    correlation = float(np.corrcoef(left, right)[0, 1])
    
    # Calculate width (inverse of correlation)
    width = (1.0 - abs(correlation))
    
    # Calculate balance
    left_rms = np.sqrt(np.mean(left**2))
    right_rms = np.sqrt(np.mean(right**2))
    total_rms = left_rms + right_rms + 1e-9
    balance = (right_rms - left_rms) / total_rms
    
    return {
        'is_stereo': True,
        'stereo_width': float(width),
        'stereo_correlation': float(correlation),
        'left_right_balance': float(balance),
    }


def analyze_spatial_field(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detailed stereo field analysis.
    
    Args:
        y: Stereo audio
        sr: Sample rate
    
    Returns:
        Dictionary with detailed stereo analysis
    """
    basic = get_stereo_features(y, sr)
    
    if not basic['is_stereo']:
        return basic
    
    if y.shape[0] == 2:
        left, right = y[0], y[1]
    else:
        left, right = y[:, 0], y[:, 1]
    
    # Mid-side analysis
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    
    mid_rms = np.sqrt(np.mean(mid**2))
    side_rms = np.sqrt(np.mean(side**2))
    
    basic['mid_energy'] = float(mid_rms)
    basic['side_energy'] = float(side_rms)
    basic['mid_side_ratio'] = float(mid_rms / (side_rms + 1e-9))
    
    return basic


# =============================================================================
# COMPARISON & MATCHING
# =============================================================================

def calculate_similarity(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
    feature_name: str
) -> float:
    """
    Calculate similarity for a specific feature (0-1).
    
    Args:
        features_a: Features from first track
        features_b: Features from second track
        feature_name: Name of feature to compare
    
    Returns:
        Similarity score (1.0 = identical, 0.0 = very different)
    """
    val_a = features_a.get(feature_name, 0)
    val_b = features_b.get(feature_name, 0)
    
    if val_a == 0 and val_b == 0:
        return 1.0
    
    # Normalized difference
    max_val = max(abs(val_a), abs(val_b))
    if max_val < 1e-9:
        return 1.0
    
    diff = abs(val_a - val_b) / max_val
    similarity = 1.0 - min(diff, 1.0)
    
    return float(similarity)


def match_features(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
    feature_list: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compare multiple features between tracks.
    
    Args:
        features_a: Features from first track
        features_b: Features from second track
        feature_list: List of features to compare (default: common features)
    
    Returns:
        Dictionary of feature similarities
    """
    if feature_list is None:
        feature_list = [
            'bpm', 'brightness', 'rms_db', 'onset_rate',
            'bass_energy', 'mid_energy', 'high_energy'
        ]
    
    similarities = {}
    for feature in feature_list:
        if feature in features_a and feature in features_b:
            similarities[feature] = calculate_similarity(
                features_a, features_b, feature
            )
    
    return similarities


def suggest_mashup_strategy(
    features_a: Dict[str, Any],
    features_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Suggest intelligent mashup mixing strategy based on features.
    
    Args:
        features_a: Features from first track
        features_b: Features from second track
    
    Returns:
        Dictionary with mixing recommendations:
        - crossfade_type: Suggested crossfade type
        - crossfade_duration: Suggested duration in beats
        - eq_strategy: EQ recommendations
        - volume_adjustments: Gain recommendations
        - key_adjustment: Transposition needed
        - bpm_adjustment: Time stretch needed
    """
    comparison = compare_tracks(features_a, features_b)
    
    suggestions = {
        'overall_compatibility': comparison['overall_similarity'],
        'mashup_potential': comparison['mashup_potential'],
    }
    
    # Crossfade type
    energy_a = features_a.get('rms_db', -20)
    energy_b = features_b.get('rms_db', -20)
    
    if abs(energy_a - energy_b) < 3:
        suggestions['crossfade_type'] = 'equal_power'
        suggestions['crossfade_duration'] = 16  # beats
    else:
        suggestions['crossfade_type'] = 'adaptive'
        suggestions['crossfade_duration'] = 32
    
    # Bass strategy
    bass_a = features_a.get('bass_energy', 0.33)
    bass_b = features_b.get('bass_energy', 0.33)
    
    if bass_a > 0.5 and bass_b > 0.5:
        suggestions['eq_strategy'] = 'frequency_split_crossfade'
        suggestions['bass_swap'] = True
    elif abs(bass_a - bass_b) > 0.3:
        suggestions['eq_strategy'] = 'complementary'
        suggestions['bass_swap'] = False
    else:
        suggestions['eq_strategy'] = 'standard'
        suggestions['bass_swap'] = False
    
    # Volume adjustments
    suggestions['gain_a_db'] = 0.0
    suggestions['gain_b_db'] = float(energy_a - energy_b)
    
    # Key adjustment
    key_a = features_a.get('key', 'unknown')
    key_b = features_b.get('key', 'unknown')
    
    if key_a != key_b and key_a != 'unknown' and key_b != 'unknown':
        suggestions['transpose_track_b'] = True
        suggestions['target_key'] = key_a
    else:
        suggestions['transpose_track_b'] = False
    
    # BPM adjustment
    bpm_a = features_a.get('bpm', 120)
    bpm_b = features_b.get('bpm', 120)
    
    if abs(bpm_a - bpm_b) > 5:
        suggestions['time_stretch_track_b'] = True
        suggestions['target_bpm'] = bpm_a
    else:
        suggestions['time_stretch_track_b'] = False
    
    return suggestions


if __name__ == '__main__':
    # Quick test
    print("Audio Analysis Module")
    print(f"Available functions: {len(__all__)}")
    print("Core: analyze_track, compare_tracks, find_mix_points")
    print("Features: spectral, harmonic, rhythm, structure, stereo")
