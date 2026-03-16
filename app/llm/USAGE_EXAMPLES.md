# AI Mashup Producer - Usage Examples

This document provides examples of how to use the AI Mashup Producer with both predefined styles and custom instructions.

## Installation

Make sure you have the required dependencies:
```bash
pip install openai anthropic librosa soundfile numpy
```

Set your API key:
```bash
export OPENAI_API_KEY="your-api-key"
# or
export ANTHROPIC_API_KEY="your-api-key"
```

## Basic Usage

### 1. Using Predefined Styles

```python
from app.llm.producer import create_mashup
import librosa

# Load tracks
track_a, sr = librosa.load("song_a.mp3")
track_b, _ = librosa.load("song_b.mp3", sr=sr)

# Progress callback (optional)
def progress(message, percentage):
    print(f"[{percentage}%] {message}")

# Create mashup with DJ mix style
mashup, result = create_mashup(
    track_a, track_b,
    track_a_name="Shape of You",
    track_b_name="Closer",
    mashup_style='dj_mix',
    progress_callback=progress
)

# Save result
if result.status == 'success':
    import soundfile as sf
    sf.write("mashup_dj_mix.wav", mashup, sr)
    print(f"Mashup created! Duration: {result.total_duration:.2f}s")
else:
    print(f"Failed: {result.errors}")
```

### 2. Using Custom Instructions (NEW!)

```python
from app.llm.producer import create_mashup
import librosa

track_a, sr = librosa.load("song_a.mp3")
track_b, _ = librosa.load("song_b.mp3", sr=sr)

# Create mashup with custom instructions
mashup, result = create_mashup(
    track_a, track_b,
    track_a_name="Track A",
    track_b_name="Track B",
    custom_instructions="""
    Create a high-energy festival mashup with the following characteristics:
    - Heavy bass drops every 16 bars
    - Build tension with filter sweeps and risers
    - Use Track A's vocals over Track B's instrumental
    - Add dramatic pauses before drops
    - Keep energy level consistently high throughout
    - Target duration: 3-4 minutes
    """,
    progress_callback=lambda msg, pct: print(f"[{pct}%] {msg}")
)

if result.status == 'success':
    import soundfile as sf
    sf.write("festival_mashup.wav", mashup, sr)
```

## Predefined Styles

### DJ Mix
Professional, seamless transitions like a club DJ:
```python
mashup, result = create_mashup(
    track_a, track_b,
    mashup_style='dj_mix'
)
```

### Creative Mashup
Experimental, artistic combinations:
```python
mashup, result = create_mashup(
    track_a, track_b,
    mashup_style='creative_mashup'
)
```

### Remix
Feature Track A, enhance with elements from Track B:
```python
mashup, result = create_mashup(
    track_a, track_b,
    mashup_style='remix'
)
```

### Quick Mix
Simple, fast mashup with minimal processing:
```python
mashup, result = create_mashup(
    track_a, track_b,
    mashup_style='quick_mix'
)
```

## Custom Instructions Examples

### Example 1: Chill Vibes
```python
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Create a chill, laid-back mashup perfect for background music.
    Use smooth crossfades, gentle EQ adjustments, and subtle reverb.
    Keep the energy low and consistent. No dramatic changes or drops.
    """
)
```

### Example 2: Workout Mix
```python
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Create an intense workout mashup with driving energy.
    Match BPMs to 128-140 range for running.
    Use punchy transitions and maintain high energy throughout.
    Add compression for extra punch.
    """
)
```

### Example 3: Nostalgic Blend
```python
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Blend these two tracks in a nostalgic way.
    Use vinyl crackle effects and slight lo-fi processing.
    Create smooth, dreamy transitions.
    Add subtle chorus and reverb for a vintage feel.
    """
)
```

### Example 4: Dance Floor Banger
```python
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Create an explosive dance floor mashup.
    Use hard cuts and dramatic transitions.
    Build energy with filter sweeps and white noise.
    Create tension before drops with drum fills.
    Target 128 BPM for maximum club impact.
    """
)
```

### Example 5: Vocal Mashup
```python
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Extract and feature vocals from Track A.
    Use instrumental sections from Track B as the backing track.
    Apply gentle pitch correction to vocals if needed.
    Use Track B's rhythm as the foundation.
    Create a cohesive vocal-focused mashup.
    """
)
```

## Advanced Usage

### Using Different LLM Providers

#### OpenAI (default)
```python
mashup, result = create_mashup(
    track_a, track_b,
    llm_provider='openai',
    llm_model='gpt-4-turbo-preview'  # optional, uses default if not specified
)
```

#### Anthropic Claude
```python
mashup, result = create_mashup(
    track_a, track_b,
    llm_provider='anthropic',
    llm_model='claude-3-opus-20240229'
)
```

#### Local Model (Ollama/LM Studio)
```python
mashup, result = create_mashup(
    track_a, track_b,
    llm_provider='local',
    llm_model='mixtral:8x7b'
)
```

### Using the Producer Class Directly

For more control, use the `AIMashupProducer` class:

```python
from app.llm.producer import AIMashupProducer
import librosa

# Initialize producer
producer = AIMashupProducer(
    sr=44100,
    llm_provider='openai',
    temperature=0.7  # Higher = more creative (0-1)
)

# Load tracks
track_a, _ = librosa.load("song_a.mp3", sr=44100)
track_b, _ = librosa.load("song_b.mp3", sr=44100)

# Create mashup
result = producer.create_mashup(
    track_a_audio=track_a,
    track_b_audio=track_b,
    track_a_name="Song A",
    track_b_name="Song B",
    custom_instructions="Your custom instructions here",
    progress_callback=lambda msg, pct: print(f"[{pct}%] {msg}")
)

# Get audio and save
mashup_audio = producer.get_mashup_audio()
if mashup_audio is not None:
    import soundfile as sf
    sf.write("mashup.wav", mashup_audio, 44100)

# Check execution log
for log_entry in result.execution_log:
    print(f"Step {log_entry.step}: {log_entry.operation} - {log_entry.status}")
```

## Tips for Custom Instructions

1. **Be Specific**: The more detailed your instructions, the better the result
2. **Mention Style**: Reference genres, artists, or moods
3. **Technical Details**: Specify BPM, effects, transitions if you have preferences
4. **Energy Level**: Describe the energy flow (building, constant, varying)
5. **Duration**: Mention target duration if you have one in mind
6. **Focus**: Specify which track should be featured or how to balance them

## Troubleshooting

### "Plan validation failed"
The LLM generated an invalid plan. Try:
- Simplifying your custom instructions
- Using a predefined style first
- Checking that your tracks have valid audio features

### "Execution failed"
An operation couldn't be executed. Check:
- Track audio quality (sample rate, channels)
- That tracks aren't too short (< 10 seconds)
- Result.execution_log for specific error

### Memory issues
For very long tracks:
- Use shorter segments
- Reduce sample rate (e.g., sr=22050)
- Process tracks in chunks

## Next Steps

- Experiment with different custom instructions
- Try different LLM providers for varied results
- Combine predefined styles with custom instructions
- Iterate on prompts based on results

For API integration, see the main API documentation.
