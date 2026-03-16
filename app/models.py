from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator

class AnalysisResult(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int
    bpm: float
    bpm_confidence: float
    n_beats: int
    note: Optional[str] = None
    error: Optional[str] = None


class MixOptions(BaseModel):
    tempo_match: bool = True
    crossfade_beats: int = Field(0, description="0 = auto based on BPM gap; otherwise beats")
    mode: Literal["snappy", "auto", "smooth"] = "smooth"
    use_stems: bool = True
    tease_bars: int = 0

    @validator("crossfade_beats")
    def _non_negative(cls, v):
        if v < 0:
            raise ValueError("crossfade_beats must be >= 0 (0 means auto).")
        return v

    @validator("tease_bars")
    def _tease_nonneg(cls, v):
        if v < 0:
            raise ValueError("tease_bars must be >= 0.")
        return v


class MixResult(BaseModel):
    out_wav: str
    out_mp3: Optional[str] = None
    sr: int
    length_sec: float
    track_a_bpm: float
    track_b_bpm_effective: float
    applied_stretch_ratio: float
    mp3_encode_ok: Optional[bool] = None
    encode_error: Optional[str] = None
    decision: Optional[Dict[str, Any]] = None
