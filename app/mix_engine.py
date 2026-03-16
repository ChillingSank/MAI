# app/mix_engine.py
from typing import Tuple, Dict, Any, List
import shutil
import numpy as np
import librosa

from .audio_utils import (
    ensure_stereo, to_mono, trim_silence,
    estimate_bpm, bpm_gap_pct, canonicalize_bpm,
    time_stretch_to_match_bpm, refine_alignment_perband,
    dj_ms_crossfade, dj_eq_crossfade, gain_curve,
    spectral_flux, band_energy,
    lufs_normalize, compress_limit, true_peak_guard,
    separate_stems_window,
)

# -------------------------------
# helpers
# -------------------------------
def crossfade_seconds(bpm: float, beats: int) -> float:
    return max(2.0, (60.0 / max(bpm, 1.0)) * max(1, beats))


def choose_beats_by_gap(gap: float, mode: str) -> int:
    base = 32 if gap <= 3.0 else (16 if gap <= 10.0 else 8)
    if mode == "snappy": return 16 if base == 32 else 8
    if mode == "smooth": return 32 if (base == 16 and gap <= 6.0) else base
    return base


def apply_shift(b: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0: return b
    if shift > 0:
        pad = np.zeros((shift, b.shape[1]), dtype=b.dtype)
        return np.vstack([pad, b])
    cut = min(b.shape[0], -shift)
    return b[cut:]


def rms_var(y: np.ndarray) -> float:
    m = to_mono(ensure_stereo(y))
    if m.size < 2048:
        return 0.0
    rms = librosa.feature.rms(y=m, frame_length=2048, hop_length=512, center=True)[0]
    return float(np.var(rms))


def _demucs_available() -> bool:
    return bool(shutil.which("demucs"))


# -------------------------------
# stems overlap renderer
# -------------------------------
def render_with_stems_overlap(a: np.ndarray, b: np.ndarray, sr: int, sec: float,
                              curve="s", tease_bars: int = 0) -> np.ndarray:
    a = ensure_stereo(a); b = ensure_stereo(b)
    n = int(max(1, sec * sr))
    a_head = a[:-n] if len(a) > n else a[:0]
    a_tail = a[-n:]; b_head = b[:n]; b_tail = b[n:]

    stems_a = separate_stems_window(a_tail, sr)
    stems_b = separate_stems_window(b_head, sr)

    ga, gb = gain_curve(curve, n); ga = ga[:, None]; gb = gb[:, None]
    t = np.linspace(0, 1, n).astype(np.float32)

    a_bass_w = (1.0 - t)[:, None] * 0.85
    b_bass_w = (t)[:, None]
    a_drm_w  = (1.0 - t)[:, None] * 0.85
    b_drm_w  = (t)[:, None] * 0.95
    a_voc_w  = (1.0 - t)[:, None] * 0.8
    b_voc_w  = gb
    a_oth_w  = (1.0 - t)[:, None] * 0.9
    b_oth_w  = gb

    overlap = (
        stems_a["bass"]   * a_bass_w * ga +
        stems_a["drums"]  * a_drm_w  * ga +
        stems_a["vocals"] * a_voc_w  * ga +
        stems_a["other"]  * a_oth_w  * ga +
        stems_b["bass"]   * b_bass_w * gb +
        stems_b["drums"]  * b_drm_w  * gb +
        stems_b["vocals"] * b_voc_w  * gb +
        stems_b["other"]  * b_oth_w  * gb
    )

    if tease_bars > 0:
        # very light pre-tease of B's vocals into the end of A_head
        beats_per_sec = 2.0
        tease_len = int(sr * max(0.5, 0.5 * tease_bars / max(beats_per_sec, 1e-6)))
        v = stems_b["vocals"]
        tlen = min(tease_len, v.shape[0])
        if tlen > 0 and a_head.shape[0] >= tlen:
            pre = a_head.copy()
            fade = np.linspace(0.0, 0.6, tlen, dtype=np.float32)[:, None]
            pre[-tlen:] = np.clip(pre[-tlen:] + v[:tlen] * fade, -1.0, 1.0)
            a_head = pre

    out = np.vstack([a_head, overlap, b_tail])
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0: out = out / peak
    return out.astype(np.float32)


# -------------------------------
# candidate scoring
# -------------------------------
def score_overlap(a_tail: np.ndarray, b_head: np.ndarray, sr: int) -> float:
    flux = spectral_flux(np.vstack([a_tail, b_head]))
    mid_clash = band_energy(a_tail + b_head, sr, 300, 3000)
    low_conflict = band_energy(a_tail + b_head, sr, 30, 140)
    return float(flux + 0.75 * mid_clash + 0.5 * low_conflict)


# -------------------------------
# “silent dip” prevention
# -------------------------------
def _rms_head(x: np.ndarray, n: int) -> float:
    if n <= 0 or x.shape[0] <= 0: return 0.0
    m = to_mono(ensure_stereo(x[:n]))
    return float(np.sqrt(np.mean(m**2)) + 1e-12)


def guard_shift_for_audibility(a: np.ndarray, b: np.ndarray, sr: int, overlap_sec: float, shift: int) -> int:
    """
    If B's head is too quiet after 'shift', search a small window around it for a louder start.
    Clamp large positive shifts to avoid silent overlap.
    """
    n = int(max(1, overlap_sec * sr))

    def _apply(x, s):
        if s == 0: return x
        if s > 0:
            pad = np.zeros((s, x.shape[1]), dtype=x.dtype)
            return np.vstack([pad, x])
        cut = min(x.shape[0], -s)
        return x[cut:]

    b0 = _apply(b, shift)
    r0 = _rms_head(b0, n)
    r0_db = 20*np.log10(r0)

    if r0_db < -35.0:
        step = max(1, int(0.05 * sr))   # 50 ms
        span = int(0.75 * sr)           # ±0.75 s
        best_s, best_r = shift, r0
        for delta in range(-span, span+1, step):
            s = shift + delta
            r = _rms_head(_apply(b, s), n)
            if r > best_r:
                best_r, best_s = r, s
        if best_r > r0 * 2.0:  # ≈ +6 dB improvement
            shift = best_s
        # final clamp: don't allow positive shift to exceed half the overlap
        max_delay = n // 2
        if shift > max_delay:
            shift = max_delay
    return shift


def auto_skip_quiet_intro(y: np.ndarray, sr: int, top_db: float = 35.0, max_skip_sec: float = 2.0) -> np.ndarray:
    """
    Skip a very quiet head (up to max_skip_sec) so the overlap hits content.
    """
    m = to_mono(ensure_stereo(y))
    intervals = librosa.effects.split(m, top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals):
        start = int(intervals[0, 0])
        max_skip = int(max_skip_sec * sr)
        if 0 < start <= max_skip:
            return y[start:]
    return y


# -------------------------------
# AUTO PLANNER
# -------------------------------
def auto_plan(y1: np.ndarray, y2: np.ndarray, sr: int) -> Dict[str, Any]:
    bpm1, _, _ = estimate_bpm(y1, sr)
    bpm2, _, _ = estimate_bpm(y2, sr)
    bpm2_c = canonicalize_bpm(bpm1, bpm2)
    gap = bpm_gap_pct(bpm1, bpm2_c)

    dur_a = len(y1) / sr
    dur_b = len(y2) / sr
    short = (dur_a < 20 or dur_b < 20)
    energy_var = rms_var(y1) + rms_var(y2)

    tempo_match = gap <= 14.0
    if short or energy_var > 0.003:
        mode = "snappy" if gap > 8.0 else "auto"
    else:
        mode = "smooth"

    beats = choose_beats_by_gap(gap, mode)
    if short and beats > 16:
        beats = 8
    elif short and beats > 8:
        beats = 8

    use_stems = _demucs_available()
    prefer_stems = use_stems
    tease_bars = 1 if (use_stems and not short and gap <= 10.0 and mode != "snappy") else 0

    return {
        "tempo_match": tempo_match,
        "crossfade_beats": int(beats),
        "mode": mode,
        "use_stems": use_stems,
        "tease_bars": int(tease_bars),
        "prefer_stems": bool(prefer_stems),
        "bpm_a": float(bpm1),
        "bpm_b": float(bpm2),
        "bpm_b_canon": float(bpm2_c),
        "gap_pct": float(gap),
        "durations": {"a_sec": dur_a, "b_sec": dur_b},
        "energy_var": float(energy_var),
    }


# -------------------------------
# MAIN: prepare_and_mix (Auto)
# -------------------------------
def prepare_and_mix(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    tempo_match: bool = True,
    crossfade_beats: int = 0,
    mode: str = "smooth",
    use_stems: bool = True,
    tease_bars: int = 0,
    prefer_stems: bool = False,
) -> Tuple[np.ndarray, float, float, float, Dict[str, Any]]:

    # trim and precondition
    y1 = ensure_stereo(trim_silence(ensure_stereo(y1)))
    y2 = ensure_stereo(trim_silence(ensure_stereo(y2)))
    # NEW: skip ultra-quiet B intro (up to 2 s)
    y2 = auto_skip_quiet_intro(y2, sr, top_db=35.0, max_skip_sec=2.0)

    # analysis
    bpm1, _, _ = estimate_bpm(y1, sr)
    bpm2, _, _ = estimate_bpm(y2, sr)
    bpm2_c = canonicalize_bpm(bpm1, bpm2)
    gap = bpm_gap_pct(bpm1, bpm2_c)

    # stretching
    do_tm = tempo_match and (gap <= 14.0)
    y2_tm, ratio = (y2, 1.0)
    adj_bpm2 = bpm2_c
    if do_tm and bpm1 > 0 and bpm2_c > 0:
        y2_tm, ratio = time_stretch_to_match_bpm(y2, sr, bpm2_c, bpm1)
        adj_bpm2 = bpm1

    # crossfade length
    beats = int(crossfade_beats) if crossfade_beats and crossfade_beats > 0 else choose_beats_by_gap(gap, mode)
    cf_sec = crossfade_seconds(bpm1 if bpm1 > 0 else 120.0, beats)

    # ensure sufficient length
    need = int(cf_sec * sr) + 1
    if len(y1) < need or len(y2_tm) < need:
        if beats > 16: beats = 16
        if beats > 8 and (len(y1) < need or len(y2_tm) < need): beats = 8
        cf_sec = crossfade_seconds(bpm1 if bpm1 > 0 else 120.0, beats)
        need = int(cf_sec * sr) + 1

    if len(y1) < need: y1 = np.pad(y1, ((0, need-len(y1)), (0,0)))
    if len(y2_tm) < need: y2_tm = np.pad(y2_tm, ((0, need-len(y2_tm)), (0,0)))
    if len(y2) < need: y2 = np.pad(y2, ((0, need-len(y2)), (0,0)))

    # alignment (per-band correlation)
    s_tm = refine_alignment_perband(y1, y2_tm, sr)
    s_or = refine_alignment_perband(y1, y2,    sr)

    # NEW: guard shifts to avoid silent B heads in the overlap
    s_tm = guard_shift_for_audibility(y1, y2_tm, sr, cf_sec, s_tm)
    s_or = guard_shift_for_audibility(y1, y2,    sr, cf_sec, s_or)

    aA, bA = y1, apply_shift(y2_tm, s_tm)  # tempo-matched
    aB, bB = y1, apply_shift(y2,    s_or)  # original tempo

    # candidates
    n = int(cf_sec * sr)
    cands: List[Dict[str, Any]] = []

    mix1 = dj_ms_crossfade(aA, bA, sr, cf_sec, curve="s" if mode!="snappy" else "linear")
    cands.append({"name": "ms_tm", "mixed": mix1,
                  "score": score_overlap(aA[-n:], bA[:n], sr),
                  "tm": True, "shift": s_tm, "renderer": "dj_ms"})

    mix2 = dj_eq_crossfade(aA, bA, sr, cf_sec, curve="s" if mode!="snappy" else "linear",
                           split_freqs=(180.0, 2400.0), bass_swap=True, hp_prog=(90.0, 25.0))
    cands.append({"name": "eq_tm", "mixed": mix2,
                  "score": score_overlap(aA[-n:], bA[:n], sr),
                  "tm": True, "shift": s_tm, "renderer": "dj_eq"})

    if gap > 10.0 or not do_tm:
        mix3 = dj_ms_crossfade(aB, bB, sr, cf_sec, curve="s")
        cands.append({"name": "ms_or", "mixed": mix3,
                      "score": score_overlap(aB[-n:], bB[:n], sr),
                      "tm": False, "shift": s_or, "renderer": "dj_ms"})

    stem_error = None
    if use_stems:
        try:
            a_use, b_use = (aA, bA) if do_tm else (aB, bB)
            mix4 = render_with_stems_overlap(a_use, b_use, sr, cf_sec, curve="s", tease_bars=max(0, int(tease_bars)))
            cands.append({"name": "stems", "mixed": mix4,
                          "score": score_overlap(a_use[-n:], b_use[:n], sr),
                          "tm": do_tm, "shift": (s_tm if do_tm else s_or), "renderer": "stems"})
        except Exception as e:
            stem_error = str(e)

    # gentle bias toward stems when close
    if use_stems:
        for c in cands:
            if c.get("renderer") == "stems":
                c["score"] *= 0.90

    # pick best
    best = min(cands, key=lambda x: x["score"])

    # master
    mixed = lufs_normalize(best["mixed"], target_lufs=-14.0)
    mixed = compress_limit(mixed, sr, thresh_db=-10.0, ratio=2.2, attack_ms=4.0, release_ms=55.0, out_ceiling_db=-0.9)
    mixed = true_peak_guard(mixed, sr, margin_db=1.0)

    dbg = {
        "bpm_a": round(float(bpm1), 2),
        "bpm_b": round(float(bpm2), 2),
        "bpm_b_canon": round(float(bpm2_c), 2),
        "gap_pct": round(float(gap), 2),
        "chosen_crossfade_beats": int(beats),
        "crossfade_seconds": round(float(cf_sec), 3),
        "strategy": ("tempo_matched" if best["tm"] else "original_tempo"),
        "overlap_renderer": best["renderer"],
        "alignment_shift_samples": int(best["shift"]),
        "score": round(float(best["score"]), 5),
        "mode": mode,
        "stem_error": stem_error,
    }
    # ratio is stretch factor used to bring B to A; if best path is not TM, keep 1.0
    ratio_out = 1.0 if not best["tm"] else (bpm1 / max(bpm2_c, 1e-9))
    return mixed.astype(np.float32), bpm1, (bpm1 if best["tm"] else bpm2_c), ratio_out, dbg
