"""
Audio State Manager

Manages audio buffers, metadata, and sections during mashup creation.
Tracks what operations have been applied and stores intermediate results.

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TRACK METADATA
# =============================================================================

@dataclass
class TrackMetadata:
    """Metadata for a track."""
    
    name: str
    original_key: Optional[str] = None
    current_key: Optional[str] = None
    original_bpm: Optional[float] = None
    current_bpm: Optional[float] = None
    duration: float = 0.0
    loudness_lufs: float = 0.0
    operations_applied: List[str] = field(default_factory=list)


# =============================================================================
# AUDIO STATE
# =============================================================================

class AudioState:
    """
    Manages audio buffers and metadata during mashup creation.
    
    This class keeps track of:
    - Original audio for both tracks
    - Processed audio after operations
    - Extracted sections
    - Intermediate mixing results
    - Metadata and operation history
    """
    
    def __init__(self, sr: int = 44100):
        """
        Initialize audio state.
        
        Args:
            sr: Sample rate
        """
        self.sr = sr
        
        # Original audio
        self.track_a_original: Optional[np.ndarray] = None
        self.track_b_original: Optional[np.ndarray] = None
        
        # Processed audio (after preprocessing)
        self.track_a: Optional[np.ndarray] = None
        self.track_b: Optional[np.ndarray] = None
        
        # Metadata
        self.track_a_meta = TrackMetadata(name="Track A")
        self.track_b_meta = TrackMetadata(name="Track B")
        
        # Sections (extracted parts of tracks)
        self.sections: Dict[str, np.ndarray] = {}
        
        # Final mashup
        self.mashup: Optional[np.ndarray] = None
        
        # Memory tracking
        self.memory_usage_mb: float = 0.0
    
    def set_track_a(self, audio: np.ndarray, name: str = "Track A"):
        """Set Track A audio."""
        self.track_a_original = audio.copy()
        self.track_a = audio.copy()
        self.track_a_meta.name = name
        self.track_a_meta.duration = len(audio) / self.sr
        self._update_memory_usage()
        logger.info(f"Track A loaded: {len(audio)} samples ({self.track_a_meta.duration:.2f}s)")
    
    def set_track_b(self, audio: np.ndarray, name: str = "Track B"):
        """Set Track B audio."""
        self.track_b_original = audio.copy()
        self.track_b = audio.copy()
        self.track_b_meta.name = name
        self.track_b_meta.duration = len(audio) / self.sr
        self._update_memory_usage()
        logger.info(f"Track B loaded: {len(audio)} samples ({self.track_b_meta.duration:.2f}s)")
    
    def update_track_a(self, audio: np.ndarray, operation: str):
        """Update Track A after an operation."""
        self.track_a = audio.copy()
        self.track_a_meta.operations_applied.append(operation)
        self.track_a_meta.duration = len(audio) / self.sr
        self._update_memory_usage()
        logger.info(f"Track A updated after {operation}")
    
    def update_track_b(self, audio: np.ndarray, operation: str):
        """Update Track B after an operation."""
        self.track_b = audio.copy()
        self.track_b_meta.operations_applied.append(operation)
        self.track_b_meta.duration = len(audio) / self.sr
        self._update_memory_usage()
        logger.info(f"Track B updated after {operation}")
    
    def add_section(self, name: str, audio: np.ndarray):
        """
        Store an extracted section.
        
        Args:
            name: Section identifier (e.g., 'intro_a', 'chorus_b')
            audio: Audio data
        """
        self.sections[name] = audio.copy()
        self._update_memory_usage()
        logger.info(f"Section '{name}' stored: {len(audio)} samples")
    
    def get_section(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a stored section.
        
        Args:
            name: Section identifier
        
        Returns:
            Audio data or None if not found
        """
        return self.sections.get(name)
    
    def set_mashup(self, audio: np.ndarray):
        """Set the final mashup."""
        self.mashup = audio.copy()
        self._update_memory_usage()
        logger.info(f"Mashup finalized: {len(audio)} samples ({len(audio)/self.sr:.2f}s)")
    
    def get_track(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get track audio by ID.
        
        Args:
            track_id: 'a', 'b', 'track_a', 'track_b', or section name
        
        Returns:
            Audio data or None
        """
        track_id = track_id.lower()
        
        if track_id in ['a', 'track_a']:
            return self.track_a
        elif track_id in ['b', 'track_b']:
            return self.track_b
        elif track_id in self.sections:
            return self.sections[track_id]
        else:
            logger.warning(f"Track ID '{track_id}' not found")
            return None
    
    def reset_track_a(self):
        """Reset Track A to original."""
        if self.track_a_original is not None:
            self.track_a = self.track_a_original.copy()
            self.track_a_meta.operations_applied = []
            logger.info("Track A reset to original")
    
    def reset_track_b(self):
        """Reset Track B to original."""
        if self.track_b_original is not None:
            self.track_b = self.track_b_original.copy()
            self.track_b_meta.operations_applied = []
            logger.info("Track B reset to original")
    
    def clear_sections(self):
        """Clear all stored sections."""
        self.sections.clear()
        self._update_memory_usage()
        logger.info("All sections cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            'track_a': {
                'name': self.track_a_meta.name,
                'duration': self.track_a_meta.duration,
                'operations_applied': len(self.track_a_meta.operations_applied),
                'has_audio': self.track_a is not None
            },
            'track_b': {
                'name': self.track_b_meta.name,
                'duration': self.track_b_meta.duration,
                'operations_applied': len(self.track_b_meta.operations_applied),
                'has_audio': self.track_b is not None
            },
            'sections_stored': len(self.sections),
            'section_names': list(self.sections.keys()),
            'mashup_ready': self.mashup is not None,
            'memory_usage_mb': self.memory_usage_mb
        }
    
    def _update_memory_usage(self):
        """Update memory usage estimate."""
        total_bytes = 0
        
        if self.track_a_original is not None:
            total_bytes += self.track_a_original.nbytes
        if self.track_b_original is not None:
            total_bytes += self.track_b_original.nbytes
        if self.track_a is not None:
            total_bytes += self.track_a.nbytes
        if self.track_b is not None:
            total_bytes += self.track_b.nbytes
        
        for section in self.sections.values():
            total_bytes += section.nbytes
        
        if self.mashup is not None:
            total_bytes += self.mashup.nbytes
        
        self.memory_usage_mb = total_bytes / (1024 * 1024)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def concatenate_sections(sections: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple audio sections.
    
    Args:
        sections: List of audio arrays
    
    Returns:
        Concatenated audio
    """
    if not sections:
        return np.array([], dtype=np.float32)
    
    return np.concatenate([s.astype(np.float32) for s in sections])


def blend_sections(
    sections: List[np.ndarray],
    gains: Optional[List[float]] = None
) -> np.ndarray:
    """
    Blend multiple sections with different gains.
    
    Args:
        sections: List of audio arrays (must be same length)
        gains: Gain for each section (default: equal mix)
    
    Returns:
        Blended audio
    """
    if not sections:
        return np.array([], dtype=np.float32)
    
    # Filter out empty sections
    sections = [s for s in sections if s is not None and len(s) > 0]
    
    if not sections:
        logger.warning("All sections are empty, cannot blend")
        return np.array([], dtype=np.float32)
    
    if gains is None:
        gains = [1.0 / len(sections)] * len(sections)
    
    # Ensure all sections are same length
    min_len = min(len(s) for s in sections)
    
    if min_len == 0:
        logger.warning("Sections have zero length, cannot blend")
        return np.array([], dtype=np.float32)
    
    sections = [s[:min_len] for s in sections]
    
    # Blend
    result = np.zeros(min_len, dtype=np.float32)
    for section, gain in zip(sections, gains):
        result += section.astype(np.float32) * gain
    
    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    
    return result


if __name__ == '__main__':
    print("Audio State Manager Module")
    print("Manages audio buffers and metadata during mashup creation")
    
    # Test
    state = AudioState(sr=44100)
    test_audio = np.random.randn(44100 * 10).astype(np.float32)  # 10 seconds
    state.set_track_a(test_audio, "Test Track A")
    state.set_track_b(test_audio, "Test Track B")
    
    summary = state.get_summary()
    print(f"\nTest state created:")
    print(f"  Track A: {summary['track_a']['duration']:.2f}s")
    print(f"  Track B: {summary['track_b']['duration']:.2f}s")
    print(f"  Memory usage: {summary['memory_usage_mb']:.2f} MB")
