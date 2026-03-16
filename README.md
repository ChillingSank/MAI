setup

python -m venv .venv
# windows
.\.venv\Scripts\Activate.ps1
# mac/linux
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

run

uvicorn app.main:app --reload --port 8000

endpoints

GET  /health
POST /analyze   form-data: file=<audio>
POST /mix       form-data: file_a=<audio>, file_b=<audio>, tempo_match=true, crossfade_beats=16
GET  /download?path=./tmp/mix_xxx.wav  (or .mp3)

notes

requires ffmpeg on PATH for mp3 export
silence trim removes head and tail silence
equal-power crossfade; length driven by track A bpm × crossfade_beats

# one-time setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install -r requirements.txt
copy .env.example .env   # then edit .env to set FFMPEG_BINARY path

# run the API
uvicorn app.main:app --port 8000
# open http://127.0.0.1:8000/docs
# download http://127.0.0.1:8000/download?path=./tmp......

What the params do

tempo_match: time-stretch B to A’s BPM (when it sounds better).

crossfade_beats:

0 = auto: based on BPM gap %

≤3% → 32 beats

3–10% → 16 beats

10% → 8 beats (you can change to 4 if you want)

mode:

snappy: shorter overlaps, cautious about stretching

auto: neutral

smooth: longer overlaps when safe, more willing to stretch

How the code works (quick tour)

app/main.py — FastAPI endpoints

/analyze: reads each file, trims silence, estimates BPM.

/mix: core flow

load both tracks (via ffmpeg), mono @ 44.1k

trim silence

estimate BPMs

BPM gap rule → choose 8/16/32 beats (or use the user’s value)

(optional) tempo-match B → A

beat alignment → snap B to A’s downbeat; try ±2 beats and pick smoothest

evaluate both strategies (tempo-matched vs original tempo) using a spectral-flux clash score
→ pick the cleaner one (with mode biases)

render equal-power crossfade → write WAV → (if ffmpeg) also MP3

return file paths + a decision block explaining exactly what it did

app/audio_utils.py — low-level audio ops

load_audio → always uses ffmpeg → WAV → soundfile (3.13-safe)

estimate_bpm, trim_silence, time_stretch_to_match_bpm

equal_power_crossfade (Apple-style curve)

write_wav, wav_to_mp3

app/mix_engine.py — brains of the mixer

Tags in comments so it’s easy to read/grep:

[BPM_GAP] rules for 8/16/32 beats

[BEAT_ALIGNMENT] downbeat snapping (tries ±2 beats)

[TEMPO_MATCHING] time-stretch decision

[CROSSFADE] fade seconds helper + render

[QUALITY_SCORE] spectral flux to pick the smoother option

mode="snappy|auto|smooth" to nudge behavior without code changes

5) Testing guide (quick)

Close BPM pair (≤3%): expect 32-beat overlap to be smooth

Medium gap (3–10%): expect 16-beat default

Big gap (>10%): expect 8-beat (or set to 4 for ultra short)

Try mode=snappy vs mode=smooth to feel the difference.

If it still feels messy, set tempo_match=false for that pair.

6) Common issues (and fixes)

500 + “ffmpeg not found”
Set FFMPEG_BINARY in .env with the full path to ffmpeg.exe, then restart.

Downloads don’t open
Use forward slashes in the URL, e.g.:
http://127.0.0.1:8000/download?path=./tmp/mix_XXXX.mp3

Beat alignment seems 0
Some intros are beat-sparse. Try a section with clear drums, or use mode=snappy to shorten overlaps, or set tempo_match=false.