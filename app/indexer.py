# app/indexer.py
from __future__ import annotations
import os, re, io, json, sqlite3, subprocess, shutil, time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import numpy as np

# Optional tag reader (best-effort)
try:
    from mutagen import File as MutagenFile
except Exception:
    MutagenFile = None  # tags will be basic only

# Optional FastAPI router (so you can trigger indexing via API)
try:
    from fastapi import APIRouter, Body
    router = APIRouter()
except Exception:
    router = None  # still usable via CLI

# Reuse your audio analysis utils
from .audio_utils import (
    load_audio, trim_silence, estimate_bpm, detect_key, rms_metrics, beat_strength,
    spectral_centroid_metrics, zcr_metrics, loudness_dbfs, SR_DEFAULT
)

# -----------------------
# Config
# -----------------------
SUPPORTED_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma"}
STEMS_CACHE_ROOT = Path(os.getenv("STEMS_CACHE", "./stems_cache")).resolve()
DEFAULT_DB_PATH = Path("./library_index.sqlite").resolve()

# -----------------------
# DB helpers
# -----------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE,
  filename TEXT,
  title TEXT,
  artist TEXT,
  album TEXT,
  genre TEXT,
  year INTEGER,
  duration_sec REAL,
  sample_rate INTEGER,
  bpm REAL,
  bpm_half REAL,
  bpm_double REAL,
  key_name TEXT,
  key_mode TEXT,
  camelot TEXT,
  key_confidence REAL,
  loudness_dbfs REAL,
  energy_variance REAL,
  beat_strength_mean REAL,
  beat_strength_std REAL,
  spectral_centroid_mean REAL,
  spectral_centroid_std REAL,
  zcr_mean REAL,
  zcr_std REAL,
  tags TEXT,
  stems_dir TEXT,
  stems_vocals_rms REAL,
  stems_drums_rms REAL,
  stems_bass_rms REAL,
  stems_other_rms REAL,
  indexed_at TEXT
);
"""

def db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(SCHEMA_SQL)
    return conn

def upsert_track(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join([":" + k for k in row.keys()])
    # on conflict(path) do update set ...
    update_set = ", ".join([f"{k}=excluded.{k}" for k in row.keys() if k not in ("path",)])
    sql = f"INSERT INTO tracks ({cols}) VALUES ({placeholders}) ON CONFLICT(path) DO UPDATE SET {update_set}"
    conn.execute(sql, row)

# -----------------------
# Small helpers
# -----------------------
def canonical_bpm_set(bpm: float) -> Tuple[float, float, float]:
    """Store 0.5x, 1x, 2x for quick canonical matching later."""
    if bpm <= 0: return 0.0, 0.0, 0.0
    return round(bpm/2.0, 2), round(bpm, 2), round(bpm*2.0, 2)

def tokenize_tags(filename: str, id3: Dict[str, Any]) -> List[str]:
    txt = " ".join([
        filename,
        str(id3.get("title") or ""),
        str(id3.get("artist") or ""),
        str(id3.get("album") or ""),
        str(id3.get("genre") or ""),
    ]).lower()
    toks = set(re.split(r"[^a-z0-9#+]+", txt))
    # some domain-specific synonyms
    extra = []
    for key in ("indian", "hind", "bollywood", "punjabi", "desi"):
        if key in txt: extra.append(key)
    for key in ("remix", "edit", "bootleg", "mashup", "cover", "instrumental", "acapella"):
        if key in txt: extra.append(key)
    toks.update(extra)
    toks.discard("")
    return sorted(toks)

def read_id3_tags(path: Path) -> Dict[str, Any]:
    if MutagenFile is None:
        return {}
    try:
        mf = MutagenFile(str(path))
        if mf is None: return {}
        meta = {}
        # very defensive: many formats/keys
        def first(x):
            if x is None: return None
            if isinstance(x, (list, tuple)) and x: return str(x[0])
            return str(x)
        # Common tag routes
        for key_guess in ("title", "TIT2"):
            v = mf.tags.get(key_guess) if getattr(mf, "tags", None) else None
            if v is not None:
                meta["title"] = first(getattr(v, "text", v))
                break
        for key_guess in ("artist", "TPE1"):
            v = mf.tags.get(key_guess) if getattr(mf, "tags", None) else None
            if v is not None:
                meta["artist"] = first(getattr(v, "text", v))
                break
        for key_guess in ("album", "TALB"):
            v = mf.tags.get(key_guess) if getattr(mf, "tags", None) else None
            if v is not None:
                meta["album"] = first(getattr(v, "text", v))
                break
        for key_guess in ("genre", "TCON"):
            v = mf.tags.get(key_guess) if getattr(mf, "tags", None) else None
            if v is not None:
                meta["genre"] = first(getattr(v, "text", v))
                break
        for key_guess in ("date", "TDRC", "TYER", "YEAR"):
            v = mf.tags.get(key_guess) if getattr(mf, "tags", None) else None
            if v is not None:
                try:
                    y = re.search(r"\d{4}", first(getattr(v, "text", v)) or "")
                    meta["year"] = int(y.group()) if y else None
                except Exception:
                    pass
                break
        return meta
    except Exception:
        return {}

# -----------------------
# Stems (optional)
# -----------------------
def stems_for_track(path: Path, demucs_model: str = "htdemucs") -> Tuple[Optional[Path], Dict[str, float]]:
    """
    Run Demucs once per track into STEMS_CACHE_ROOT / model / <trackname> and compute RMS per stem.
    Returns (stems_dir, energies). If demucs missing/failed, returns (None, {}).
    """
    demucs_bin = shutil.which("demucs")
    if demucs_bin is None:
        return None, {}

    out_root = STEMS_CACHE_ROOT / demucs_model
    out_root.mkdir(parents=True, exist_ok=True)

    # Demucs naming: subfolder with the track name; handle collisions by using stem
    track_base = path.stem
    stems_dir = out_root / track_base

    # If already exists with stems, reuse
    expected = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    if stems_dir.exists() and all((stems_dir / s).exists() for s in expected):
        return stems_dir, _rms_per_stem(stems_dir)

    # Run demucs (full file). If your library is big, consider doing this later on-demand.
    try:
        cmd = [
            demucs_bin, "-n", demucs_model,
            "-o", str(out_root),
            str(path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except Exception:
        return None, {}

    if stems_dir.exists():
        return stems_dir, _rms_per_stem(stems_dir)
    # Some demucs versions add an extra folder level with model name:
    alt_dir = out_root / demucs_model / track_base
    if alt_dir.exists():
        return alt_dir, _rms_per_stem(alt_dir)

    # Try to find a folder containing expected stems
    for p in out_root.rglob("*"):
        if p.is_dir() and all((p / s).exists() for s in expected):
            return p, _rms_per_stem(p)

    return None, {}

def _rms_wav(path: Path) -> float:
    import soundfile as sf
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return float(np.sqrt(np.mean(y**2)) + 1e-12)
    except Exception:
        return 0.0

def _rms_per_stem(stems_dir: Path) -> Dict[str, float]:
    d = {}
    for name in ("vocals", "drums", "bass", "other"):
        f = stems_dir / f"{name}.wav"
        d[name] = _rms_wav(f) if f.exists() else 0.0
    return d

# -----------------------
# Core indexing
# -----------------------
def analyze_file(path: Path, sr: int = SR_DEFAULT, sample_window_sec: int = 90) -> Dict[str, Any]:
    """
    Load audio (full), analyze trimmed content; if very long, we still compute quickly.
    For huge files, consider slicing a middle window – but for small POC libraries, full is fine.
    """
    raw = path.read_bytes()
    y, sr = load_audio(raw, sr, filename=path.name)
    y = trim_silence(y)
    dur = float(len(y) / sr)

    # If very long, focus on the central window to speed up indexing
    if dur > sample_window_sec * 1.5:
        win = int(sample_window_sec * sr)
        start = int(max(0, (len(y) - win) // 2))
        y = y[start:start+win]
        dur = float(len(y) / sr)

    bpm, conf, beats = estimate_bpm(y, sr)
    key_name, key_mode, camelot, key_conf = detect_key(y, sr)
    rms_overall, rms_curve = rms_metrics(y, sr, curve_rate_hz=2.0)
    bs_mean, bs_std = beat_strength(y, sr)
    sc_mean, sc_std = spectral_centroid_metrics(y, sr)
    zcr_mean, zcr_std = zcr_metrics(y, sr)
    dbfs = loudness_dbfs(y)

    energy_var = float(np.var(rms_curve)) if rms_curve is not None and len(rms_curve) else 0.0
    bpm_half, bpm_full, bpm_double = canonical_bpm_set(float(bpm))

    return {
        "duration_sec": round(dur, 3),
        "sample_rate": sr,
        "bpm": round(float(bpm), 2),
        "bpm_half": bpm_half,
        "bpm_double": bpm_double,
        "bpm_confidence": round(float(conf), 3),
        "n_beats": int(len(beats)),
        "key_name": key_name,
        "key_mode": key_mode,
        "camelot": camelot,
        "key_confidence": round(float(key_conf), 3),
        "rms_overall": round(float(rms_overall), 6),
        "energy_variance": round(energy_var, 6),
        "beat_strength_mean": round(float(bs_mean), 6),
        "beat_strength_std": round(float(bs_std), 6),
        "spectral_centroid_mean_hz": round(float(sc_mean), 2),
        "spectral_centroid_std_hz": round(float(sc_std), 2),
        "zcr_mean": round(float(zcr_mean), 6),
        "zcr_std": round(float(zcr_std), 6),
        "dbfs_integrated": round(float(dbfs), 2),
    }

def index_library(
    library_dir: Path,
    db_path: Path = DEFAULT_DB_PATH,
    include_exts: Optional[List[str]] = None,
    do_stems: bool = False,
    demucs_model: str = "htdemucs",
    sample_window_sec: int = 90,
) -> Dict[str, Any]:
    t0 = time.time()
    lib = Path(library_dir).resolve()
    dbp = Path(db_path).resolve()
    if not lib.exists():
        raise FileNotFoundError(f"Library folder not found: {lib}")

    if include_exts is None:
        exts = SUPPORTED_EXTS
    else:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in include_exts}

    files = [p for p in lib.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    conn = db_connect(dbp)
    added = 0
    errors: List[str] = []

    for idx, path in enumerate(sorted(files, key=lambda p: p.name.lower())):
        try:
            id3 = read_id3_tags(path)
            features = analyze_file(path, sample_window_sec=sample_window_sec)

            stems_dir = None
            stems_energy = {}
            if do_stems:
                stems_dir, stems_energy = stems_for_track(path, demucs_model=demucs_model)

            tags = tokenize_tags(path.name, id3)
            row = {
                "path": str(path),
                "filename": path.name,
                "title": id3.get("title") or path.stem,
                "artist": id3.get("artist"),
                "album": id3.get("album"),
                "genre": id3.get("genre"),
                "year": id3.get("year"),
                "duration_sec": features["duration_sec"],
                "sample_rate": features["sample_rate"],
                "bpm": features["bpm"],
                "bpm_half": features["bpm_half"],
                "bpm_double": features["bpm_double"],
                "key_name": features["key_name"],
                "key_mode": features["key_mode"],
                "camelot": features["camelot"],
                "key_confidence": features["key_confidence"],
                "loudness_dbfs": features["dbfs_integrated"],
                "energy_variance": features["energy_variance"],
                "beat_strength_mean": features["beat_strength_mean"],
                "beat_strength_std": features["beat_strength_std"],
                "spectral_centroid_mean": features["spectral_centroid_mean_hz"],
                "spectral_centroid_std": features["spectral_centroid_std_hz"],
                "zcr_mean": features["zcr_mean"],
                "zcr_std": features["zcr_std"],
                "tags": ",".join(tags),
                "stems_dir": str(stems_dir) if stems_dir else None,
                "stems_vocals_rms": stems_energy.get("vocals"),
                "stems_drums_rms": stems_energy.get("drums"),
                "stems_bass_rms": stems_energy.get("bass"),
                "stems_other_rms": stems_energy.get("other"),
                "indexed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            upsert_track(conn, row)
            added += 1
            if (idx + 1) % 10 == 0:
                conn.commit()
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    conn.commit()
    conn.close()
    dt = time.time() - t0
    return {
        "scanned": len(files),
        "indexed": added,
        "errors": errors,
        "db_path": str(dbp),
        "elapsed_sec": round(dt, 2),
        "stems_cache": str(STEMS_CACHE_ROOT),
    }

# -----------------------
# FastAPI endpoint (optional)
# -----------------------
if router is not None:
    @router.post("/index_library")
    def index_library_api(
        library_dir: str = Body(..., embed=True),
        db_path: Optional[str] = Body(None, embed=True),
        do_stems: bool = Body(False, embed=True),
        demucs_model: str = Body("htdemucs", embed=True),
        sample_window_sec: int = Body(90, embed=True),
        include_exts: Optional[List[str]] = Body(None, embed=True),
    ):
        result = index_library(
            Path(library_dir),
            Path(db_path) if db_path else DEFAULT_DB_PATH,
            include_exts=include_exts,
            do_stems=do_stems,
            demucs_model=demucs_model,
            sample_window_sec=sample_window_sec,
        )
        return result

# -----------------------
# CLI usage
# -----------------------
def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Index a local music library into SQLite with audio features (and optional stems).")
    ap.add_argument("--library", required=True, help="Path to the folder containing your songs")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite DB (default: ./library_index.sqlite)")
    ap.add_argument("--exts", nargs="*", default=None, help="File extensions to include, e.g. mp3 wav flac")
    ap.add_argument("--stems", action="store_true", help="Also run Demucs once per track and cache stems+energies")
    ap.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    ap.add_argument("--sample-window-sec", type=int, default=90, help="If file is long, analyze ~this many seconds (center)")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
    res = index_library(
        Path(args.library),
        Path(args.db),
        include_exts=args.exts,
        do_stems=args.stems,
        demucs_model=args.demucs_model,
        sample_window_sec=args.sample_window_sec,
    )
    print(json.dumps(res, indent=2))
