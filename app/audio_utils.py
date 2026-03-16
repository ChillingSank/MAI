# app/audio_utils.py
from __future__ import annotations
import os
import io
import math
import subprocess
import shutil
import tempfile
from typing import Tuple, Optional, List, Dict

import numpy as np
import librosa
import soundfile as sf

# -------------------------------
# [BASIC] constants / paths
# -------------------------------
SR_DEFAULT = int(os.getenv("SAMPLE_RATE", 44100))
TMP_DIR = os.getenv("TMP_DIR", "./tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# ffmpeg path (env or PATH)
FFMPEG = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")


# -------------------------------
# [DECODE] compressed -> temp WAV
# -------------------------------
def _ffmpeg_decode_to_wav(src_path: str, dst_sr: int) -> str:
    """
    Decode any compressed audio (mp3/m4a/ogg/opus/etc.) to a temporary mono WAV
    at the target sample rate, using ffmpeg. Returns the WAV temp path.
    """
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH (needed to decode compressed audio)")
    tmp_wav = src_path + ".wav"
    cmd = [
        FFMPEG, "-nostdin", "-y",
        "-i", src_path,
        "-ac", "1", "-ar", str(dst_sr),
        tmp_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return tmp_wav


# -------------------------------
# [LOAD] main loader
# -------------------------------
def load_audio(file_bytes: bytes, sr: int = SR_DEFAULT, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Loads audio as float32 mono at `sr`. For compressed formats we always:
      bytes -> temp file -> ffmpeg -> temp WAV -> sf.read
    This avoids librosa/audioread issues on Python 3.13.
    """
    ext = (os.path.splitext(filename or "")[1] or "").lower()
    needs_ffmpeg = {".mp3", ".m4a", ".aac", ".wma", ".mp4", ".mkv", ".ogg", ".opus", ".flac"}

    # Fast path for WAV from memory
    if ext == ".wav":
        y, _sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if _sr != sr:
            y = librosa.resample(y, orig_sr=_sr, target_sr=sr)
        return y.astype(np.float32), sr

    # Everything else -> write temp input, decode with ffmpeg to wav, read wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext if ext else ".bin") as tmp_in:
        tmp_in.write(file_bytes)
        in_path = tmp_in.name

    wav_path = None
    try:
        wav_path = _ffmpeg_decode_to_wav(in_path, sr)
        y, _ = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), sr
    finally:
        try:
            os.remove(in_path)
        except OSError:
            pass
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


# -------------------------------
# [SILENCE]
# -------------------------------
def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt


# -------------------------------
# [ANALYZE] bpm
# -------------------------------
def estimate_bpm(y: np.ndarray, sr: int) -> Tuple[float, float, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(tempo)
    conf = float(np.clip((onset_env.mean() / (onset_env.std() + 1e-6)), 0.0, 5.0))
    return bpm, conf, beats


# -------------------------------
# [TEMPO_MATCHING] stretch
# -------------------------------
def time_stretch_to_match_bpm(y: np.ndarray, sr: int, src_bpm: float, tgt_bpm: float) -> Tuple[np.ndarray, float]:
    if src_bpm <= 0 or tgt_bpm <= 0:
        return y, 1.0
    ratio = float(tgt_bpm / src_bpm)
    if abs(1.0 - ratio) < 0.02:
        return y, 1.0
    # clamp extreme ratios to avoid artifacts
    ratio = float(np.clip(ratio, 0.5, 2.0))
    y_st = librosa.effects.time_stretch(y, rate=ratio)
    return y_st, ratio


# -------------------------------
# [CROSSFADE] equal-power
# -------------------------------
def equal_power_crossfade(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float) -> np.ndarray:
    a = a.flatten()
    b = b.flatten()
    n = int(crossfade_sec * sr)
    n = max(1, n)

    a_head = a[:-n] if len(a) > n else np.zeros(0, dtype=np.float32)
    a_tail = a[-n:]
    b_head = b[:n]
    b_tail = b[n:]

    t = np.linspace(0, 1, n, endpoint=True).astype(np.float32)
    gain_a = np.cos(t * math.pi / 2.0) ** 2
    gain_b = np.sin(t * math.pi / 2.0) ** 2

    overlap = a_tail * gain_a + b_head * gain_b
    out = np.concatenate([a_head, overlap, b_tail])
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


# -------------------------------
# [IO] write / convert
# -------------------------------
def write_wav(y: np.ndarray, sr: int, path: str) -> str:
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def wav_to_mp3(wav_path: str, mp3_path: str) -> Optional[str]:
    if not FFMPEG:
        return None
    try:
        cmd = [
            FFMPEG, "-nostdin", "-y",
            "-i", wav_path,
            "-vn",
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            mp3_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return mp3_path
    except Exception:
        return None


# -------------------------------
# [UTIL] beats -> seconds
# -------------------------------
def beats_to_seconds(n_beats: int, bpm: float) -> float:
    if bpm <= 0:
        return 0.0
    return (60.0 / bpm) * n_beats


# ======================================================================
# Key detection (Krumhansl) + Camelot mapping  [USED BY INDEXER/MIX]
# ======================================================================
_C_MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
_C_MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)
_MAJOR_TEMPLATES = np.stack([np.roll(_C_MAJOR_PROFILE, i) for i in range(12)], axis=0)
_MINOR_TEMPLATES = np.stack([np.roll(_C_MINOR_PROFILE, i) for i in range(12)], axis=0)
_PITCHES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _camelot_code(pitch_index: int, mode: str) -> str:
    """
    Map pitch class + mode to Camelot code.
    Major -> B, Minor -> A ; number = ((pitch*7)%12)+1
    """
    num = ((pitch_index * 7) % 12) + 1
    let = 'B' if mode == 'maj' else 'A'
    return f"{num}{let}"

def detect_key(y: np.ndarray, sr: int) -> Tuple[str, str, Optional[str], float]:
    """
    Detect musical key using chroma + Krumhansl profiles.
    Returns: (key_name, mode['maj'|'min'|'' if unknown], camelot or None, confidence[0..1])
    """
    # Focus on harmonic content
    try:
        y_h, _ = librosa.effects.hpss(y)
    except Exception:
        y_h = y
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=1024)
    if chroma.size == 0:
        return "unknown", "", None, 0.0

    v = chroma.mean(axis=1)
    v = v / (v.sum() + 1e-9)

    corr_maj = _MAJOR_TEMPLATES @ v
    corr_min = _MINOR_TEMPLATES @ v

    i_maj = int(np.argmax(corr_maj)); s_maj = float(corr_maj[i_maj])
    i_min = int(np.argmax(corr_min)); s_min = float(corr_min[i_min])

    if max(s_maj, s_min) <= 0:
        return "unknown", "", None, 0.0

    if s_maj >= s_min:
        pitch_i, mode, score = i_maj, 'maj', s_maj
    else:
        pitch_i, mode, score = i_min, 'min', s_min

    conf = score / (s_maj + s_min + 1e-9)
    if conf < 0.35:
        return "unknown", "", None, float(conf)

    key_name = _PITCHES[pitch_i]
    camelot = _camelot_code(pitch_i, mode)
    return key_name, mode, camelot, float(np.clip(conf, 0.0, 1.0))


# -------- Energy (RMS) metrics --------
def rms_metrics(y: np.ndarray, sr: int, curve_rate_hz: float = 2.0) -> Tuple[float, np.ndarray]:
    """
    Returns (rms_overall, rms_curve) where rms_curve is downsampled to ~curve_rate_hz.
    """
    # Per-frame RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512, center=True)[0]
    # Overall RMS (linear)
    rms_overall = float(np.sqrt(np.mean(y**2)) + 1e-12)
    # Downsample curve to target rate
    frames_per_sec = sr / 512.0
    step = max(1, int(round(frames_per_sec / max(curve_rate_hz, 0.1))))
    rms_curve = rms[::step].astype(np.float32)
    return float(rms_overall), rms_curve


# -------- Beat strength (onset energy at beats) --------
def beat_strength(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Approximate 'punch' using onset strength sampled at beat frames.
    Returns (mean, std) of beat salience.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames", hop_length=512)
    if len(beats) == 0:
        return 0.0, 0.0
    vals = onset_env[np.clip(beats.astype(int), 0, len(onset_env)-1)]
    return float(np.mean(vals)), float(np.std(vals))


# -------- Spectral centroid (brightness) --------
def spectral_centroid_metrics(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Returns (mean_hz, std_hz) of spectral centroid.
    """
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(sc)), float(np.std(sc))


# -------- Zero-crossing rate (noisiness/voicing) --------
def zcr_metrics(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Returns (mean, std) zero-crossing rate (per frame, [0..1]).
    """
    z = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(z)), float(np.std(z))


# -------- Loudness in dBFS --------
def loudness_dbfs(y: np.ndarray) -> float:
    """
    Integrated loudness relative to full-scale from linear RMS.
    dbFS = 20 * log10(rms), clamped to [-80, 0] approx.
    """
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    db = 20.0 * math.log10(rms)
    return float(max(-80.0, min(0.0, db)))


# -------- Beat utilities for bar windows --------
def beat_frames(y: np.ndarray, sr: int, hop_length: int = 512):
    # Returns beat frame indices (librosa frames), and a converter to samples
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    def frames_to_samples(frames):
        return librosa.frames_to_samples(frames, hop_length=hop_length)
    return tempo, beats, frames_to_samples

def window_from_beats(y: np.ndarray, sr: int, start_b: int, n_beats: int, hop_length: int = 512) -> tuple[int, int]:
    # Returns (start_sample, end_sample) for a beat-synchronous window
    _, beats, f2s = beat_frames(y, sr, hop_length)
    if len(beats) == 0:
        return 0, min(len(y), int((60.0/120.0)*n_beats*sr))  # crude fallback
    start_idx = int(np.clip(start_b, 0, max(0, len(beats)-1)))
    end_idx = int(np.clip(start_idx + n_beats, start_idx, len(beats)-1))
    s = int(f2s(beats[start_idx]))
    e = int(f2s(beats[end_idx])) if end_idx < len(beats) else len(y)
    return s, e

# -------- Outro/Intro finders (lightweight heuristics) --------
def find_outro_window(y: np.ndarray, sr: int, bpm: float, beats_span: int = 16) -> tuple[int, int]:
    """
    Pick a smooth outro near the end: last `beats_span` beats with steady/decreasing RMS.
    Falls back gracefully if analysis is weak.
    """
    _, rms_curve = rms_metrics(y, sr, curve_rate_hz=4.0)
    # use the last ~30 seconds worth of curve to avoid early sections
    tail = rms_curve[-min(len(rms_curve), 4*30):] if len(rms_curve) else rms_curve
    # prefer lower-variance windows: smooth == steady
    if len(tail) >= 8:
        win = max(1, len(tail)//8)
        var = np.array([np.var(tail[i:i+win]) for i in range(0, len(tail)-win)])
        _ = int(np.argmax(np.linspace(0,1,len(var)) * (var.max() - var) ))
    # Map this position to beats near the end
    _, beats, _ = beat_frames(y, sr)
    if len(beats) < beats_span + 1:
        # fallback to last N seconds
        fade_s = max(2.0, min(8.0, 60.0/max(bpm,1.0)*beats_span))
        start = max(0, int(len(y) - fade_s*sr))
        return start, len(y)
    # choose the last beat index minus the span
    start_b = max(0, len(beats) - (beats_span+1))
    s, e = window_from_beats(y, sr, start_b, beats_span)
    return s, e

def find_intro_window(y: np.ndarray, sr: int, bpm: float, beats_span: int = 16) -> tuple[int, int]:
    """
    Pick a clean intro: first `beats_span` beats after the first strong downbeat.
    """
    _, beats, f2s = beat_frames(y, sr)
    if len(beats) < beats_span + 1:
        # fallback to first N seconds
        fade_s = max(2.0, min(8.0, 60.0/max(bpm,1.0)*beats_span))
        return 0, min(len(y), int(fade_s*sr))
    # skip the very first beat if it’s a pickup; start at beat 1
    start_b = 1 if len(beats) > (beats_span+1) else 0
    s, e = window_from_beats(y, sr, start_b, beats_span)
    return s, e


# ======================================================================
# EXTRA UTILITIES used by mixer (crossfades, EQ, FX, mastering)
# ======================================================================

# [JIRA MIX-103] Adaptive fade curves
def gain_curve(kind: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (gain_a, gain_b) for 'linear' | 'log' | 's' | 'equal'
    """
    t = np.linspace(0, 1, n, dtype=np.float32)
    if kind == "linear":
        return 1.0 - t, t
    if kind == "log":
        return np.sqrt(1.0 - t), np.sqrt(t)
    if kind == "s":  # S-curve
        s = 3*t*t - 2*t*t*t
        return 1.0 - s, s
    # equal-power default
    return np.cos(t * math.pi / 2.0) ** 2, np.sin(t * math.pi / 2.0) ** 2


def adaptive_crossfade(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float, curve: str = "equal") -> np.ndarray:
    """
    Generalized crossfade with selectable curve.
    """
    a = a.flatten(); b = b.flatten()
    n = max(1, int(crossfade_sec * sr))
    a_head = a[:-n] if len(a) > n else np.zeros(0, dtype=np.float32)
    a_tail = a[-n:]; b_head = b[:n]; b_tail = b[n:]
    ga, gb = gain_curve(curve, n)
    overlap = a_tail * ga + b_head * gb
    out = np.concatenate([a_head, overlap, b_tail])
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0: out = out / peak
    return out.astype(np.float32)


# [JIRA MIX-101] EQ-based crossfade (multi-band) + bass swap
def _fft_masks(n_fft: int, sr: int, split_freqs: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.linspace(0, sr/2, n_fft//2 + 1, dtype=np.float32)
    f1, f2 = split_freqs
    low = (f <= f1).astype(np.float32)
    mid = ((f > f1) & (f <= f2)).astype(np.float32)
    high = (f > f2).astype(np.float32)
    return low, mid, high

def _apply_mask_time(y: np.ndarray, mask_1side: np.ndarray) -> np.ndarray:
    n = len(y)
    n_fft = int(2**np.ceil(np.log2(max(1024, n))))
    Y = np.fft.rfft(y, n=n_fft)
    m = mask_1side
    if len(m) != len(Y):
        m = np.interp(np.arange(len(Y)), np.linspace(0, len(Y)-1, num=len(mask_1side)), mask_1side)
    y_out = np.fft.irfft(Y * m, n=n_fft)[:n]
    return y_out.astype(np.float32)

def eq_crossfade(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float, split_freqs: Tuple[float, float] = (200.0, 2000.0), curve: str = "equal", bass_swap: bool = True) -> np.ndarray:
    """
    Multi-band crossfade: fades L/M/H with separate gain curves; optional 'bass swap'
    (duck outgoing bass while bringing in incoming bass).
    """
    a = a.flatten(); b = b.flatten()
    n = max(1, int(crossfade_sec * sr))
    a_head = a[:-n] if len(a) > n else np.zeros(0, dtype=np.float32)
    a_tail = a[-n:]; b_head = b[:n]; b_tail = b[n:]
    ga, gb = gain_curve(curve, n)

    low, mid, high = _fft_masks(4096, sr, split_freqs)
    aL = _apply_mask_time(a_tail, low); aM = _apply_mask_time(a_tail, mid); aH = _apply_mask_time(a_tail, high)
    bL = _apply_mask_time(b_head, low); bM = _apply_mask_time(b_head, mid); bH = _apply_mask_time(b_head, high)

    if bass_swap:
        ol = aL * (ga * 0.5) + bL * gb
    else:
        ol = aL * ga + bL * gb
    om = aM * ga + bM * gb
    oh = aH * ga + bH * gb

    overlap = np.clip(ol + om + oh, -1.0, 1.0)
    out = np.concatenate([a_head, overlap, b_tail])
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0: out = out / peak
    return out.astype(np.float32)


# [JIRA MIX-102] Energy-aware fade length helper
def energy_aware_crossfade_beats(rms_curve: np.ndarray, bpm: float, base_beats: int = 16, min_beats: int = 8, max_beats: int = 32) -> int:
    """
    Decide fade beats based on recent RMS variance:
      - high variance / high energy -> shorter fades
      - low variance / breakdown -> longer fades
    """
    if bpm <= 0 or rms_curve is None or len(rms_curve) < 8:
        return base_beats
    tail = rms_curve[-min(len(rms_curve), 64):]
    v = float(np.var(tail))
    # Normalize variance to [0,1] using robust scale
    p90 = float(np.percentile(rms_curve, 90)) + 1e-9
    norm = np.clip(v / (p90**2 + 1e-9), 0.0, 1.0)
    beats = base_beats - int(norm * (base_beats - min_beats))
    return int(np.clip(beats, min_beats, max_beats))


# [JIRA MIX-202] Downbeat/bar grid (approx 4/4)
def estimate_bar_downbeats(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Approximate downbeats by assuming 4/4 and taking every 4th detected beat.
    Returns sample indices of bar-starts.
    """
    _, beats, _ = beat_frames(y, sr, hop_length=hop_length)
    if len(beats) == 0:
        return np.array([], dtype=int)
    # assume first beat is bar start; every 4th beat = downbeat
    bar_idxs = np.arange(0, len(beats), 4, dtype=int)
    samples = librosa.frames_to_samples(beats[bar_idxs], hop_length=hop_length)
    return samples.astype(int)


# [JIRA MIX-201] Section detection (intro/chorus/drop/outro heuristics)
def detect_sections(y: np.ndarray, sr: int) -> Dict[str, Tuple[int, int]]:
    """
    Lightweight heuristic segmentation:
      - compute RMS + spectral flux; pick sustained high-energy windows as 'chorus/drop'
      - first low-energy window as 'intro', last steady window as 'outro'
    Returns dict with sample ranges.
    """
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    flux = np.clip(np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0)), 0, None)
    rms = librosa.feature.rms(S=S)[0]

    def smooth(x, k=8):
        k = max(1, k)
        w = np.ones(k)/k
        return np.convolve(x, w, mode="same")

    rms_s = smooth(rms, 16); flux_s = smooth(flux, 16)
    score = 0.6 * rms_s[1:] + 0.4 * flux_s  # align sizes

    frames_per_sec = sr / hop
    win_frames = int(max(frames_per_sec * 8.0, 64))  # ~8s
    if len(score) < win_frames + 1:
        return {"full": (0, len(y))}
    best_i = int(np.argmax(np.convolve(score, np.ones(win_frames), mode="valid")))
    s_f = best_i; e_f = min(len(score)-1, best_i + win_frames)
    s = int(librosa.frames_to_samples(s_f, hop_length=hop))
    e = int(librosa.frames_to_samples(e_f, hop_length=hop))

    # intro: first low-energy ~6s
    intro_f = int(max(0, np.argmin(smooth(rms_s, 64)) - frames_per_sec*3))
    intro_s = 0
    intro_e = int(min(s, librosa.frames_to_samples(intro_f + int(frames_per_sec*6), hop_length=hop)))

    # outro: last low-variance ~6s
    outro_e = len(y)
    outro_s = max(int(e), int(len(y) - sr*6))

    return {
        "intro": (intro_s, intro_e),
        "chorus": (s, e),
        "drop": (s, e),
        "outro": (outro_s, outro_e),
    }


# [JIRA MIX-301] Pitch-shift micro-adjust (±1 semitone typical)
def pitch_shift_semitones(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    semitones = float(np.clip(semitones, -2.0, 2.0))  # keep subtle by default
    if abs(semitones) < 1e-6:
        return y.astype(np.float32)
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=semitones)


# [JIRA MIX-302] Harmonic EQ morphing (simple midrange ducking)
def harmonic_eq_morph(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float, curve: str = "equal", mid_band: Tuple[float, float] = (300.0, 3000.0), duck_db: float = 4.0) -> np.ndarray:
    """
    During overlap, duck mid-band of outgoing track slightly to reduce clashes.
    """
    a = a.flatten(); b = b.flatten()
    n = max(1, int(crossfade_sec * sr))
    a_head = a[:-n] if len(a) > n else np.zeros(0, dtype=np.float32)
    a_tail = a[-n:]; b_head = b[:n]; b_tail = b[n:]
    ga, gb = gain_curve(curve, n)

    low, mid, high = _fft_masks(4096, sr, (mid_band[0], mid_band[1]))
    a_mid = _apply_mask_time(a_tail, mid)
    a_sides = _apply_mask_time(a_tail, low + high)
    b_all = b_head

    duck = 10 ** (-duck_db / 20.0)
    overlap = (a_mid * ga * duck + a_sides * ga) + (b_all * gb)
    out = np.concatenate([a_head, overlap, b_tail])
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0: out = out / peak
    return out.astype(np.float32)


# [JIRA MIX-401] Filter sweep (LP or HP) over a segment
def filter_sweep(y: np.ndarray, sr: int, kind: str = "lp", start_hz: float = 20000.0, end_hz: float = 200.0) -> np.ndarray:
    """
    Apply a simple 1st-order time-varying filter sweep ('lp' or 'hp').
    """
    y = y.astype(np.float32).copy()
    n = len(y)
    freqs = np.linspace(start_hz, end_hz, n).astype(np.float32)
    x = 0.0
    out = np.zeros_like(y)
    for i in range(n):
        fc = float(np.clip(freqs[i], 40.0, sr/2 - 100.0))
        # one-pole coefficient
        alpha = float(np.exp(-2.0 * math.pi * fc / sr))
        x = alpha * x + (1 - alpha) * y[i]
        if kind == "lp":
            out[i] = x
        else:  # high-pass via input - lowpass
            out[i] = y[i] - x
    # normalize lightly
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


# [JIRA MIX-402] Reverb tail (very light Schroeder-style)
def reverb_tail(y: np.ndarray, sr: int, wet: float = 0.15, decay: float = 0.4) -> np.ndarray:
    y = y.astype(np.float32)
    # simple multi-tap feedback delay network
    delays = [int(sr*0.011), int(sr*0.013), int(sr*0.017), int(sr*0.019)]
    buf = np.zeros((len(delays), len(y)), dtype=np.float32)
    out = y.copy()
    for idx, d in enumerate(delays):
        g = decay * (0.9 ** idx)
        for i in range(d, len(y)):
            buf[idx, i] = y[i] + g * buf[idx, i-d]
        out += wet * buf[idx]
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


# [JIRA MIX-403] Echo freeze (repeat last beat with decay)
def echo_freeze(y: np.ndarray, sr: int, bpm: float, repeats: int = 4, decay: float = 0.6) -> np.ndarray:
    if bpm <= 0:
        return y
    beat_samps = int(sr * 60.0 / bpm)
    seg = y[-beat_samps:] if len(y) >= beat_samps else y
    out = y.astype(np.float32).copy()
    for k in range(1, repeats + 1):
        add = (seg * (decay ** k)).astype(np.float32)
        out = np.concatenate([out, add])
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


# [JIRA MIX-404] Stutter cuts (repeat short slice before drop)
def stutter_slice(y: np.ndarray, sr: int, bpm: float, fraction: float = 0.25, repeats: int = 3) -> np.ndarray:
    if bpm <= 0:
        return y
    beat_samps = int(sr * 60.0 / bpm)
    slen = int(max(1, beat_samps * fraction))
    seg = y[-slen:] if len(y) >= slen else y
    rep = np.tile(seg, repeats)
    out = np.concatenate([y[:-slen], rep])
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


# [JIRA MIX-501] Simple compressor + limiter
def compress_limit(y: np.ndarray, sr: int, thresh_db: float = -12.0, ratio: float = 2.0, attack_ms: float = 5.0, release_ms: float = 50.0, out_ceiling_db: float = -0.8) -> np.ndarray:
    """
    Basic peak-like compressor then soft limiter.
    """
    y = y.astype(np.float32)
    # envelope
    attack = math.exp(-1.0 / (sr * (attack_ms/1000.0) + 1e-9))
    release = math.exp(-1.0 / (sr * (release_ms/1000.0) + 1e-9))
    env = 0.0
    out = np.zeros_like(y)
    thr = 10 ** (thresh_db / 20.0)
    for i, s in enumerate(np.abs(y)):
        if s > env:
            env = attack * env + (1-attack) * s
        else:
            env = release * env + (1-release) * s
        gain = 1.0
        if env > thr:
            over = env / thr
            comp = over ** (1.0 - 1.0/ratio)
            gain = 1.0 / comp
        out[i] = y[i] * gain
    # limiter (soft clip)
    ceiling = 10 ** (out_ceiling_db / 20.0)
    out = np.tanh(out / max(ceiling, 1e-6)) * ceiling
    return out.astype(np.float32)


# [JIRA MIX-502] Loudness normalization (RMS target)
def loudness_normalize(y: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    """
    Simple RMS-normalization to a target integrated loudness (approx).
    """
    cur = loudness_dbfs(y)
    gain_db = float(target_dbfs - cur)
    gain = 10 ** (gain_db / 20.0)
    out = y.astype(np.float32) * gain
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak  # avoid clipping
    return out.astype(np.float32)


# -------------------------------
# [MISSING FUNCTIONS] Stubs for mix_engine.py compatibility
# -------------------------------
def ensure_stereo(y: np.ndarray) -> np.ndarray:
    """Convert mono to stereo if needed."""
    if y.ndim == 1:
        return np.stack([y, y], axis=0)
    return y


def to_mono(y: np.ndarray) -> np.ndarray:
    """Convert stereo to mono."""
    if y.ndim == 2:
        return np.mean(y, axis=0)
    return y


def bpm_gap_pct(bpm1: float, bpm2: float) -> float:
    """Calculate percentage gap between two BPMs."""
    return abs(bpm1 - bpm2) / max(bpm1, bpm2) * 100.0


def canonicalize_bpm(bpm: float) -> float:
    """Normalize BPM to standard range (e.g., 90-180)."""
    while bpm < 90:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


def refine_alignment_perband(a: np.ndarray, b: np.ndarray, sr: int) -> int:
    """Refine alignment offset between two audio signals."""
    # Simple cross-correlation based alignment
    correlation = np.correlate(a[:sr*10], b[:sr*10], mode='full')
    offset = np.argmax(correlation) - len(a[:sr*10]) + 1
    return int(offset)


def dj_ms_crossfade(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float) -> np.ndarray:
    """Mid-side crossfade for DJ mixing."""
    return equal_power_crossfade(a, b, sr, crossfade_sec)


def dj_eq_crossfade(a: np.ndarray, b: np.ndarray, sr: int, crossfade_sec: float) -> np.ndarray:
    """EQ-based crossfade for DJ mixing."""
    return equal_power_crossfade(a, b, sr, crossfade_sec)


def spectral_flux(y: np.ndarray, sr: int) -> np.ndarray:
    """Calculate spectral flux."""
    stft = librosa.stft(y)
    mag = np.abs(stft)
    flux = np.diff(mag, axis=1)
    return np.sum(flux, axis=0)


def band_energy(y: np.ndarray, sr: int, fmin: float = 20.0, fmax: float = 200.0) -> np.ndarray:
    """Calculate energy in a frequency band."""
    stft = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_mag = np.abs(stft[mask, :])
    return np.sum(band_mag ** 2, axis=0)


def lufs_normalize(y: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
    """LUFS-based loudness normalization (approximation)."""
    return loudness_normalize(y, target_dbfs=target_lufs)


def true_peak_guard(y: np.ndarray, max_db: float = -1.0) -> np.ndarray:
    """Guard against true peaks exceeding threshold."""
    max_val = 10 ** (max_db / 20.0)
    peak = np.max(np.abs(y))
    if peak > max_val:
        y = y * (max_val / peak)
    return y


def separate_stems_window(y: np.ndarray, sr: int, start_sec: float, duration_sec: float) -> Dict[str, np.ndarray]:
    """Stub for stem separation (requires Demucs)."""
    # Return empty stems as placeholder
    segment = y[int(start_sec*sr):int((start_sec+duration_sec)*sr)]
    return {
        'vocals': segment * 0.0,
        'drums': segment * 0.0,
        'bass': segment * 0.0,
        'other': segment
    }
