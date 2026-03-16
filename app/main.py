# app/main.py
import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .audio_utils import (
    load_audio, write_wav, wav_to_mp3, SR_DEFAULT
)
from .mix_engine import prepare_and_mix, auto_plan
from .indexer import router as indexer_router  # Phase 1: library indexer API
from .routers.ai_mashup_router import router as ai_mashup_router  # Phase 4: AI-powered mashups (V2)

app = FastAPI(
    title="MAI Mashup API",
    version="2.0.0",
    description="Music AI Mashup System with LLM-powered decision making"
)

# Mount static files for the web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

OUT_DIR = os.getenv("TMP_DIR", "./tmp")
os.makedirs(OUT_DIR, exist_ok=True)


def _which(name: str) -> Optional[str]:
    try:
        return shutil.which(name)
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI homepage"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return HTMLResponse("<h1>MAI Mashup API</h1><p>Web UI not found. Visit /docs for API documentation.</p>")


@app.get("/health")
def health():
    ffmpeg = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
    demucs = os.getenv("DEMUCS_BINARY") or _which("demucs")
    return {
        "status": "ok",
        "tmp_dir": os.path.abspath(OUT_DIR),
        "ffmpeg_found": bool(ffmpeg),
        "ffmpeg_path": ffmpeg,
        "demucs_found": bool(demucs),
        "demucs_path": demucs,
    }


# ---- Include Phase 1 indexer routes (/index_library) ----
app.include_router(indexer_router)

# ---- Include Phase 4 AI Mashup V2 routes (/api/v2) ----
# These routes use ONLY the new LLM system (app/llm/* + app/utils/*)
# Zero dependencies on old mix_engine.py or audio_utils.py
app.include_router(ai_mashup_router)


# ---------- Auto-mix (JSON) ----------
@app.post("/mix_auto")
async def mix_auto(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    raw_a = await file_a.read()
    raw_b = await file_b.read()

    y1, sr = load_audio(raw_a, SR_DEFAULT, file_a.filename)
    y2, _  = load_audio(raw_b, SR_DEFAULT, file_b.filename)

    # Let the engine pick optimal settings
    plan = auto_plan(y1, y2, sr)

    mixed, bpm1, bpm2_eff, ratio, dbg = prepare_and_mix(
        y1, y2, sr,
        tempo_match=plan["tempo_match"],
        crossfade_beats=plan["crossfade_beats"],
        mode=plan["mode"],
        use_stems=plan["use_stems"],
        tease_bars=plan["tease_bars"],
        prefer_stems=plan["prefer_stems"],
    )

    job_id = uuid.uuid4().hex[:12]
    wav_path = os.path.join(OUT_DIR, f"mix_{job_id}.wav")
    mp3_path = os.path.join(OUT_DIR, f"mix_{job_id}.mp3")
    write_wav(mixed, sr, wav_path)
    mp3_done = wav_to_mp3(wav_path, mp3_path)

    return JSONResponse({
        "out_wav": wav_path,
        "out_mp3": mp3_done,
        "sr": sr,
        "length_sec": round(len(mixed) / sr, 3),
        "track_a_bpm": round(float(bpm1), 2),
        "track_b_bpm_effective": round(float(bpm2_eff), 2),
        "applied_stretch_ratio": round(float(ratio), 5),
        "mp3_encode_ok": bool(mp3_done),
        "encode_error": None if mp3_done else "ffmpeg encode failed",
        "auto_plan": plan,
        "decision": {
            "strategy": dbg.get("strategy"),
            "overlap_renderer": dbg.get("overlap_renderer"),
            "alignment_shift_samples": dbg.get("alignment_shift_samples"),
            "score": dbg.get("score"),
            "chosen_crossfade_beats": dbg.get("chosen_crossfade_beats"),
            "crossfade_seconds": dbg.get("crossfade_seconds"),
        },
    })


# ---------- Auto-mix (UI with download button) ----------
@app.post("/mix_auto_ui", response_class=HTMLResponse)
async def mix_auto_ui(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    raw_a = await file_a.read()
    raw_b = await file_b.read()

    y1, sr = load_audio(raw_a, SR_DEFAULT, file_a.filename)
    y2, _  = load_audio(raw_b, SR_DEFAULT, file_b.filename)

    plan = auto_plan(y1, y2, sr)

    mixed, bpm1, bpm2_eff, ratio, dbg = prepare_and_mix(
        y1, y2, sr,
        tempo_match=plan["tempo_match"],
        crossfade_beats=plan["crossfade_beats"],
        mode=plan["mode"],
        use_stems=plan["use_stems"],
        tease_bars=plan["tease_bars"],
        prefer_stems=plan["prefer_stems"],
    )

    job_id = uuid.uuid4().hex[:12]
    wav_path = os.path.join(OUT_DIR, f"mix_{job_id}.wav")
    mp3_path = os.path.join(OUT_DIR, f"mix_{job_id}.mp3")
    write_wav(mixed, sr, wav_path)
    mp3_done = wav_to_mp3(wav_path, mp3_path) or wav_path
    dl = f"/download?path={mp3_done}"

    meta = f"""
      <ul>
        <li>BPM A: {round(float(bpm1),2)}</li>
        <li>BPM B (effective): {round(float(bpm2_eff),2)}</li>
        <li>Gap %: {round(float(plan.get('gap_pct', 0.0)),2)}</li>
        <li>Crossfade: {dbg.get('chosen_crossfade_beats')} beats ({dbg.get('crossfade_seconds')}s)</li>
        <li>Mode: {plan.get('mode')}</li>
        <li>Renderer: {dbg.get('overlap_renderer')}</li>
        <li>Strategy: {dbg.get('strategy')}</li>
        <li>Stems used: {plan.get('use_stems')}</li>
      </ul>
    """

    return HTMLResponse(f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>MAI – Auto Mix Result</title>
      <style>
        body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:24px;max-width:860px;margin:auto}}
        .btn{{display:inline-block;padding:10px 16px;border-radius:10px;background:#111;color:#fff;text-decoration:none}}
        .card{{border:1px solid #ddd;border-radius:12px;padding:20px;margin-bottom:18px}}
        .meta{{color:#444;margin-top:18px}}
        code{{background:#f6f8fa;padding:2px 6px;border-radius:6px}}
      </style>
    </head>
    <body>
      <div class="card">
        <h2>Mashup ready</h2>
        <a class="btn" href="{dl}">Download MP3</a>
        <div class="meta">{meta}</div>
        <p style="margin-top:18px"><a href="/">Make another</a></p>
      </div>
    </body>
    </html>
    """)


# ---------- Hardened download ----------
@app.get("/download")
async def download(path: str):
    base = os.path.realpath(OUT_DIR)
    real = os.path.realpath(path)
    if not (real == base or real.startswith(base + os.sep)):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.exists(real):
        return JSONResponse({"error": "not found"}, status_code=404)
    filename = os.path.basename(real)
    media = "audio/mpeg" if filename.lower().endswith(".mp3") else "application/octet-stream"
    return FileResponse(real, media_type=media, filename=filename)


# ---------- Root UI (Auto only) ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>MAI – Auto Mashup</title>
          <style>
            body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:24px;max-width:860px;margin:auto}
            .card{border:1px solid #ddd;border-radius:12px;padding:20px;margin-bottom:18px}
            .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
            input[type=file]{padding:8px;border:1px solid #ccc;border-radius:8px}
            button{padding:10px 16px;border-radius:10px;border:0;background:#111;color:#fff;cursor:pointer}
            .hint{color:#666;font-size:0.9rem;margin-top:8px}
            code{background:#f6f8fa;padding:2px 6px;border-radius:6px}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>Auto Mix (Recommended)</h2>
            <form action="/mix_auto_ui" method="post" enctype="multipart/form-data">
              <div class="row" style="margin-bottom:12px">
                <input name="file_a" type="file" accept="audio/*" required>
                <input name="file_b" type="file" accept="audio/*" required>
              </div>
              <button type="submit">Create Auto Mix</button>
              <div class="hint">No settings needed — the engine picks tempo match, stems, crossfade, mode, and vocal tease automatically.</div>
            </form>
          </div>

          <div class="card">
            <h3>Phase 1: Index your local library (optional)</h3>
            <p>If you want “type a vibe → auto-pick songs” later, index a folder of tracks once using the API:</p>
            <pre><code>POST /index_library
Content-Type: application/json

{ "library_dir": "C:\\\\Users\\\\you\\\\Music\\\\library", "do_stems": false, "sample_window_sec": 90 }</code></pre>
            <p>or via CLI:</p>
            <pre><code>python -m app.indexer --library "C:\\Users\\you\\Music\\library"</code></pre>
          </div>
        </body>
        </html>
        """
    )
