# AI Mashup API V2 - Documentation

## Overview

The V2 API provides **AI-powered mashup creation** using Large Language Models (LLMs) to intelligently analyze and mix audio tracks.

**Key Features:**
- 🤖 LLM-powered decision making (OpenAI, Anthropic, local models)
- 🎵 5 mashup styles (4 predefined + custom instructions)
- 📊 Real-time progress via WebSocket
- ✅ Plan preview without execution
- 🔒 Complete independence from old system

## Architecture

```
V1 API (Legacy):          V2 API (New):
/mix_auto                 /api/v2/ai-mashup
    ↓                         ↓
mix_engine.py             app/llm/* (LLM system)
audio_utils.py            app/utils/* (Utilities)
```

**V2 Dependencies:**
- ✅ Uses: `app/llm/*` (new LLM system)
- ✅ Uses: `app/utils/*` (refactored utilities)
- ❌ **Does NOT use**: `mix_engine.py`, `audio_utils.py`

## Base URL

```
http://localhost:8000/api/v2
```

## Authentication

Currently no authentication required. API keys for LLM providers should be set via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

---

## Endpoints

### 1. Health Check

**GET** `/api/v2/health`

Check system health and available features.

**Response:**
```json
{
  "status": "healthy",
  "llm_providers_available": ["openai", "anthropic", "local"],
  "operations_count": 90,
  "utils_modules": [
    "key_manipulations",
    "bpm_manipulations",
    "volume_manipulations",
    "mixing_manipulations",
    "audio_analysis",
    "effects_manipulations",
    "transition_manipulations"
  ],
  "errors": null
}
```

---

### 2. Create AI Mashup

**POST** `/api/v2/ai-mashup`

Create an AI-powered mashup from two audio files.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_a` | File | ✅ Yes | - | First audio track (MP3, WAV, etc.) |
| `file_b` | File | ✅ Yes | - | Second audio track |
| `mashup_style` | String | No | `dj_mix` | Style: `dj_mix`, `creative_mashup`, `remix`, `quick_mix` |
| `custom_instructions` | String | No | None | Custom text instructions (overrides style) |
| `llm_provider` | String | No | `openai` | LLM provider: `openai`, `anthropic`, `local` |
| `llm_model` | String | No | None | Specific model (uses default if omitted) |
| `temperature` | Float | No | `0.7` | Creativity (0-1) |
| `track_a_name` | String | No | `Track A` | Display name for Track A |
| `track_b_name` | String | No | `Track B` | Display name for Track B |

**Response:**
```json
{
  "task_id": "abc123def456",
  "status": "queued",
  "message": "Mashup task created and queued for processing",
  "websocket_url": "/ws/ai-mashup/abc123def456",
  "status_url": "/api/v2/ai-mashup/abc123def456"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v2/ai-mashup \
  -F "file_a=@song_a.mp3" \
  -F "file_b=@song_b.mp3" \
  -F "mashup_style=dj_mix" \
  -F "track_a_name=Shape of You" \
  -F "track_b_name=Closer"
```

**Example (Python):**
```python
import requests

files = {
    'file_a': open('song_a.mp3', 'rb'),
    'file_b': open('song_b.mp3', 'rb')
}

data = {
    'mashup_style': 'dj_mix',
    'track_a_name': 'Shape of You',
    'track_b_name': 'Closer',
    'llm_provider': 'openai'
}

response = requests.post(
    'http://localhost:8000/api/v2/ai-mashup',
    files=files,
    data=data
)

result = response.json()
task_id = result['task_id']
print(f"Task created: {task_id}")
```

---

### 3. Get Task Status

**GET** `/api/v2/ai-mashup/{task_id}`

Check the status of a mashup task.

**Response (Processing):**
```json
{
  "task_id": "abc123def456",
  "status": "processing",
  "progress": 45,
  "message": "Executing mix plan step 3...",
  "created_at": "2025-10-11T10:30:00Z",
  "completed_at": null,
  "mashup_url": null,
  "duration_seconds": null,
  "execution_log": null,
  "errors": null
}
```

**Response (Completed):**
```json
{
  "task_id": "abc123def456",
  "status": "completed",
  "progress": 100,
  "message": "Mashup completed successfully!",
  "created_at": "2025-10-11T10:30:00Z",
  "completed_at": "2025-10-11T10:32:15Z",
  "mashup_url": "/api/v2/download/abc123def456",
  "duration_seconds": 245.3,
  "execution_log": [
    {
      "step": 1,
      "operation": "transpose_to_key",
      "status": "success",
      "message": "Transposed Track A to D minor"
    },
    {
      "step": 2,
      "operation": "time_stretch_to_bpm",
      "status": "success",
      "message": "Stretched Track B to 128 BPM"
    }
  ],
  "errors": null
}
```

**Example:**
```bash
curl http://localhost:8000/api/v2/ai-mashup/abc123def456
```

---

### 4. Download Mashup

**GET** `/api/v2/download/{task_id}`

Download the completed mashup file.

**Response:**
- Content-Type: `audio/wav`
- File: `ai_mashup_{task_id}.wav`

**Example:**
```bash
curl -O http://localhost:8000/api/v2/download/abc123def456
```

---

### 5. Preview Plan (Without Execution)

**POST** `/api/v2/ai-mashup/preview`

Generate and preview a mashup plan without actually executing it.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_a` | File | ✅ Yes | First audio track |
| `file_b` | File | ✅ Yes | Second audio track |
| `mashup_style` | String | No | Style (default: `dj_mix`) |
| `custom_instructions` | String | No | Custom instructions |
| `llm_provider` | String | No | LLM provider (default: `openai`) |
| `track_a_name` | String | No | Track A name |
| `track_b_name` | String | No | Track B name |

**Response:**
```json
{
  "status": "success",
  "plan": {
    "compatibility_analysis": {
      "key_compatibility": 85,
      "bpm_compatibility": 90,
      "energy_match": 75,
      "overall_score": 83
    },
    "preprocessing": [
      {
        "track": "a",
        "operation": "transpose_to_key",
        "parameters": {"target_key": "D minor"},
        "reason": "Match harmonic compatibility"
      }
    ],
    "mix_plan": [
      {
        "step": 1,
        "action": "Play intro from Track A",
        "operation": "extract_section",
        "parameters": {"start": 0, "end": 16},
        "timing": "0:00-0:16"
      }
    ],
    "creative_enhancements": [
      {
        "effect": "reverb",
        "target": "mix",
        "parameters": {"room_size": 0.5, "wet": 0.3}
      }
    ]
  },
  "estimated_duration": 245.0,
  "validation_warnings": null,
  "errors": null
}
```

**Use Cases:**
- Preview what the LLM will do before committing
- Iterate on custom instructions
- Understand the generated plan
- Debug unexpected behavior

---

### 6. WebSocket Progress Updates

**WebSocket** `/ws/ai-mashup/{task_id}`

Real-time progress updates during mashup creation.

**Messages Sent:**

**Connected:**
```json
{
  "type": "connected",
  "task_id": "abc123",
  "status": "processing",
  "progress": 0,
  "message": "Starting..."
}
```

**Progress Update:**
```json
{
  "type": "progress",
  "task_id": "abc123",
  "progress": 45,
  "message": "Executing mix step 3..."
}
```

**Completed:**
```json
{
  "type": "completed",
  "task_id": "abc123",
  "mashup_url": "/api/v2/download/abc123",
  "duration": 245.3
}
```

**Failed:**
```json
{
  "type": "failed",
  "task_id": "abc123",
  "errors": ["Error message"]
}
```

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/ai-mashup/abc123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`${data.progress}%: ${data.message}`);
  } else if (data.type === 'completed') {
    console.log('Mashup ready:', data.mashup_url);
    ws.close();
  } else if (data.type === 'failed') {
    console.error('Failed:', data.errors);
    ws.close();
  }
};
```

**Example (Python):**
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    
    if data['type'] == 'progress':
        print(f"{data['progress']}%: {data['message']}")
    elif data['type'] == 'completed':
        print(f"Mashup ready: {data['mashup_url']}")
        ws.close()
    elif data['type'] == 'failed':
        print(f"Failed: {data['errors']}")
        ws.close()

ws = websocket.WebSocketApp(
    'ws://localhost:8000/ws/ai-mashup/abc123',
    on_message=on_message
)
ws.run_forever()
```

---

### 7. List Available Operations

**GET** `/api/v2/operations`

Get a list of all available audio operations that the LLM can use.

**Response:**
```json
{
  "operations": [
    "detect_key",
    "transpose_to_key",
    "pitch_shift",
    "detect_bpm",
    "time_stretch_to_bpm",
    "normalize_lufs",
    "crossfade",
    "reverb",
    "delay",
    "...90+ more operations"
  ],
  "count": 90
}
```

---

## Mashup Styles

### 1. DJ Mix (`dj_mix`)
- Professional, seamless transitions
- Camelot wheel harmonic mixing
- Beat-matched crossfades
- Club-ready output

### 2. Creative Mashup (`creative_mashup`)
- Experimental combinations
- Creative effects and processing
- Risk-taking approach
- Unique results

### 3. Remix (`remix`)
- Feature one track prominently
- Enhance with elements from the other
- Polished, radio-ready sound

### 4. Quick Mix (`quick_mix`)
- Simple, fast (max 5 steps)
- Minimal processing
- Good for quick tests

### 5. Custom Instructions
Use `custom_instructions` for complete control:

```python
custom_instructions = """
Create a high-energy festival mashup:
- Heavy bass drops every 16 bars
- Dramatic buildups with filter sweeps
- Use Track A's vocals over Track B's instrumental
- Keep energy consistently high (8/10)
- Target 128 BPM for maximum club impact
- Duration: 3-4 minutes
"""
```

---

## Error Handling

All endpoints return standard error responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid audio file format",
  "detail": "Track A could not be loaded",
  "task_id": "abc123"
}
```

**Common Errors:**
- `400 Bad Request`: Invalid parameters or file format
- `404 Not Found`: Task ID not found
- `500 Internal Server Error`: Server error (check logs)

---

## Complete Workflow Example

```python
import requests
import websocket
import json
import time

# 1. Create mashup task
files = {
    'file_a': open('song_a.mp3', 'rb'),
    'file_b': open('song_b.mp3', 'rb')
}

data = {
    'custom_instructions': 'Create a high-energy festival mashup with heavy bass drops',
    'llm_provider': 'openai'
}

response = requests.post(
    'http://localhost:8000/api/v2/ai-mashup',
    files=files,
    data=data
)

task_id = response.json()['task_id']
print(f"Task created: {task_id}")

# 2. Connect to WebSocket for real-time updates
def on_message(ws, message):
    data = json.loads(message)
    print(f"[{data['type']}] {data.get('message', '')}")
    
    if data['type'] in ['completed', 'failed']:
        ws.close()

ws = websocket.WebSocketApp(
    f'ws://localhost:8000/ws/ai-mashup/{task_id}',
    on_message=on_message
)

# Run WebSocket in background
import threading
threading.Thread(target=ws.run_forever, daemon=True).start()

# 3. Poll status (alternative to WebSocket)
while True:
    status = requests.get(f'http://localhost:8000/api/v2/ai-mashup/{task_id}').json()
    
    if status['status'] == 'completed':
        print(f"Completed! Duration: {status['duration_seconds']}s")
        
        # 4. Download mashup
        mashup = requests.get(f'http://localhost:8000{status["mashup_url"]}')
        with open(f'mashup_{task_id}.wav', 'wb') as f:
            f.write(mashup.content)
        
        print(f"Mashup saved to mashup_{task_id}.wav")
        break
    
    elif status['status'] == 'failed':
        print(f"Failed: {status['errors']}")
        break
    
    time.sleep(2)
```

---

## Configuration

Environment variables:

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Output directory
export AI_MASHUP_OUTPUT_DIR="./tmp/ai_mashups"

# Sample rate (default: 44100)
export DEFAULT_SAMPLE_RATE="44100"
```

---

## Comparison: V1 vs V2

| Feature | V1 (`/mix_auto`) | V2 (`/api/v2/ai-mashup`) |
|---------|------------------|--------------------------|
| **Decision Making** | Rule-based | LLM-powered |
| **Customization** | Limited presets | Full natural language |
| **Progress Tracking** | No | Yes (WebSocket) |
| **Plan Preview** | No | Yes |
| **Async Processing** | No | Yes |
| **Operations** | ~20 | 90+ |
| **Music Theory** | Basic | Advanced (Camelot wheel) |
| **Dependencies** | Old scripts | New LLM system only |

---

## Troubleshooting

### "LLM provider not available"
- Ensure API key is set: `export OPENAI_API_KEY="..."`
- Check health endpoint: `GET /api/v2/health`

### "Task stuck in processing"
- Check server logs for errors
- Task might be waiting for LLM response (can take 10-30s)
- Use WebSocket to see real-time progress

### "Plan validation failed"
- LLM generated invalid plan
- Try simplifying custom instructions
- Use predefined style first

### "No mashup audio generated"
- Check execution log in task status
- Specific operation may have failed
- See server logs for details

---

## Next Steps

- 📖 See [USAGE_EXAMPLES.md](../llm/USAGE_EXAMPLES.md) for programmatic usage
- 📋 See [planning/plan.md](../llm/planning/plan.md) for implementation details
- 🧪 Run tests: `pytest tests/integration/test_api.py` (when available)
- 🚀 Deploy: See deployment guide (coming soon)

---

**Version**: 2.0.0  
**Last Updated**: October 11, 2025  
**Status**: Phase 4 Complete ✅
