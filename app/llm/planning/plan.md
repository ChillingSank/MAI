# AI Mashup System - Implementation Plan

## Project Overview

**Goal**: Build an AI-powered mashup creation system using LLMs as decision engines

**Architecture Choice**: Option A - LLM as Decision Engine
- LLM analyzes track features and generates a complete mashup plan in JSON format
- System validates and executes the plan using 148 utility functions
- Structured output using Pydantic models ensures reliability

**Date Started**: October 11, 2025  
**Current Status**: Phases 1-3 Complete (Core Infrastructure Ready)

---

## Implementation Roadmap

### ✅ **Phase 1: Foundation** (COMPLETED)

**Objective**: Establish data models and prompt templates

#### Completed Tasks:
1. **Created `models.py`** (310 lines, 10 Pydantic models)
   - `CompatibilityAnalysis` - Track compatibility scoring (key, BPM, energy)
   - `PreprocessingStep` - Operations to apply before mixing
   - `MashupStructure` - Overall arrangement (sequential/parallel/sandwich)
   - `MixStep` - Individual mixing operations with timing
   - `CreativeEnhancement` - Optional effects to apply
   - `MashupPlan` - Complete top-level plan container
   - `ExecutionLog` - Log entries for executed operations
   - `MashupResult` - Final mashup output with status
   - `TrackFeatures` - Audio features for LLM analysis
   - `TwoTrackAnalysis` - Both tracks' features combined

2. **Created `prompts.py`** (421 lines, 5 prompt templates)
   - `DJ_MIX_PROMPT` - Seamless club-style mixing with Camelot wheel
   - `CREATIVE_MASHUP_PROMPT` - Experimental artistic mashups
   - `REMIX_PROMPT` - Feature Track A, enhance with Track B
   - `QUICK_MIX_PROMPT` - Simple fast mashups (max 5 steps)
   - `CUSTOM_MASHUP_PROMPT` - Free-form user instructions (NEW!)
   - `AVAILABLE_OPERATIONS` - Complete list of 90+ operations
   - `MUSIC_THEORY_GUIDELINES` - Camelot wheel, BPM compatibility, energy matching
   - Helper functions: `get_prompt_for_style()`, `build_user_prompt()`

3. **Created `__init__.py`** (27 lines)
   - Package initialization
   - Exports main models for external use

#### Key Features:
- ✅ Structured JSON output validation
- ✅ Music theory integration (Camelot wheel)
- ✅ Support for 4 predefined styles + custom instructions
- ✅ Comprehensive operation registry documentation

---

### ✅ **Phase 2: LLM Integration** (COMPLETED)

**Objective**: Build multi-provider LLM client with reliable JSON output

#### Completed Tasks:
1. **Created `client.py`** (290 lines, 3 LLM clients)
   - `LLMClient` - Abstract base class with `generate()` and `generate_json()`
   - `OpenAIClient` - GPT-4 integration with `response_format={"type": "json_object"}`
   - `AnthropicClient` - Claude 3 integration with JSON extraction from markdown
   - `LocalModelClient` - Ollama/LM Studio integration via HTTP
   - `create_llm_client()` - Factory function for provider selection
   - `generate_mashup_plan_json()` - Convenience function for one-shot generation

#### Key Features:
- ✅ Multi-provider support (OpenAI, Anthropic, local models)
- ✅ JSON-forced responses with retry logic (max 3 attempts)
- ✅ Environment variable configuration
- ✅ Temperature control for creativity tuning
- ✅ Graceful error handling and fallbacks

#### Configuration:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
# Local model at localhost:11434 (Ollama default)
```

---

### ✅ **Phase 3: Execution Engine** (COMPLETED)

**Objective**: Build robust execution, validation, and state management

#### Completed Tasks:
1. **Created `executor.py`** (280 lines)
   - `OPERATION_REGISTRY` - Maps 90+ operation names to utility functions
   - `OperationExecutor` class with `execute_operation()` method
   - Automatic parameter injection (sr, y)
   - Execution logging (success/failed/skipped)
   - Helper functions: `validate_operation()`, `get_available_operations()`, `get_operation_categories()`
   - Operations from 7 utility modules:
     * Key manipulations (transpose, pitch shift, key detection)
     * BPM manipulations (time stretch, beat detection, alignment)
     * Volume manipulations (normalize, gain, compression)
     * Mixing manipulations (crossfade, blend, layering)
     * Audio analysis (detect key, BPM, beats, energy)
     * Effects (reverb, delay, chorus, distortion, EQ)
     * Transitions (sweeps, risers, drops, buildups)

2. **Created `audio_state.py`** (250 lines)
   - `TrackMetadata` dataclass - Track info and operation history
   - `AudioState` class - Manages all audio buffers
     * `track_a_original`, `track_b_original` - Pristine copies for rollback
     * `track_a`, `track_b` - Working copies after operations
     * `sections` dict - Stored extracted sections by name
     * `mashup` - Final concatenated output
   - Methods: `set_track_a/b()`, `update_track_a/b()`, `add_section()`, `get_section()`, `reset_track_a/b()`
   - Memory usage tracking in MB
   - Helper functions: `concatenate_sections()`, `blend_sections()`

3. **Created `validators.py`** (360 lines)
   - `PARAMETER_LIMITS` - Safety ranges for 40+ parameters
     * Semitones: (-12, 12), Target BPM: (60, 200), Gain: (0, 2.0)
     * Target LUFS: (-20, -5), Ratio: (1, 20), Attack: (0.001, 1.0)
     * Wet/Dry: (0, 1), Feedback: (0, 0.95), Delay Time: (0.01, 2.0)
   - `PlanValidator` class
     * `validate_plan()` - Validates entire MashupPlan
     * `_validate_preprocessing_step()` - Check operation exists, validate params
     * `_validate_mix_step()` - Same for mix steps
     * `_validate_enhancement()` - Same for effects
     * `_validate_parameters()` - Clamp to safe ranges, check time ranges
   - Sanitization functions:
     * `sanitize_parameters()` - Clamp all params to PARAMETER_LIMITS
     * `validate_track_id()` - Normalize track identifiers
     * `validate_time_range()` - Ensure start < end, within duration
     * `validate_key()` - Check "C major", "F# minor" format
     * `validate_json_structure()` - Verify required keys present

4. **Created `producer.py`** (537 lines) - **MAIN ORCHESTRATOR**
   - `AIMashupProducer` class - Coordinates entire workflow
   - Main method: `create_mashup()`
     * Step 1: Load tracks into audio state
     * Step 2: Analyze both tracks (extract features)
     * Step 3: Generate mashup plan from LLM
     * Step 4: Validate plan with safety checks
     * Step 5: Execute plan (preprocessing → mix steps → enhancements)
     * Step 6: Finalize and return mashup
   - Private methods:
     * `_analyze_track_features()` - Extract audio features
     * `_generate_plan()` - Call LLM with prompts
     * `_execute_plan()` - Execute all steps sequentially
     * `_execute_preprocessing_step()` - Apply key/BPM/volume adjustments
     * `_execute_mix_step()` - Execute mixing operations
     * `_execute_enhancement()` - Apply creative effects
   - Convenience function: `create_mashup()` - One-line mashup creation
   - Progress callback support for real-time updates

#### Key Features:
- ✅ Complete operation registry (90+ functions)
- ✅ Robust state management with rollback capability
- ✅ Comprehensive parameter validation and sanitization
- ✅ Memory tracking and optimization
- ✅ Detailed execution logging
- ✅ Progress callbacks for UI integration
- ✅ Error recovery and graceful degradation

---

## Current System Capabilities

### What Works Now:

```python
from app.llm.producer import create_mashup
import librosa

# Load tracks
track_a, sr = librosa.load("song_a.mp3")
track_b, _ = librosa.load("song_b.mp3", sr=sr)

# Option 1: Use predefined style
mashup, result = create_mashup(
    track_a, track_b,
    mashup_style='dj_mix',  # or 'creative_mashup', 'remix', 'quick_mix'
    progress_callback=lambda msg, pct: print(f"[{pct}%] {msg}")
)

# Option 2: Use custom instructions (NEW!)
mashup, result = create_mashup(
    track_a, track_b,
    custom_instructions="""
    Create a high-energy festival mashup with heavy bass drops,
    dramatic buildups, and keep energy consistently high.
    Use Track A's vocals over Track B's instrumental.
    """
)

# Save result
if result.status == 'success':
    import soundfile as sf
    sf.write("mashup.wav", mashup, sr)
    print(f"Duration: {result.total_duration:.2f}s")
```

### Available Styles:
1. **DJ Mix** - Professional seamless transitions, Camelot wheel mixing
2. **Creative Mashup** - Experimental, artistic, risk-taking combinations
3. **Remix** - Feature one track, enhance with the other
4. **Quick Mix** - Simple, fast (max 5 steps, minimal processing)
5. **Custom Instructions** - Free-form text description of desired mashup

### LLM Providers Supported:
- ✅ OpenAI (GPT-4, GPT-3.5)
- ✅ Anthropic (Claude 3 Opus/Sonnet/Haiku)
- ✅ Local Models (Ollama, LM Studio)

---

## ⏳ **Phase 4: API Integration** (PENDING - NEXT)

**Objective**: Create FastAPI endpoints for web/app integration

### Planned Tasks:
1. **Create API endpoint** - `POST /api/ai-mashup`
   - Accept file uploads (track_a, track_b)
   - Accept mashup_style or custom_instructions
   - Return task ID for async processing

2. **Implement async processing**
   - Use FastAPI BackgroundTasks or Celery
   - Store mashup results in database/filesystem
   - Support progress tracking

3. **Add WebSocket endpoint** - `/ws/ai-mashup/{task_id}`
   - Real-time progress updates
   - Stream execution log
   - Notify on completion

4. **Create result endpoint** - `GET /api/ai-mashup/{task_id}`
   - Return mashup status
   - Provide download link when ready
   - Include execution log and metadata

5. **Add plan preview endpoint** - `POST /api/ai-mashup/preview`
   - Generate plan without execution
   - Allow user to review before committing
   - Support plan regeneration

### Expected API Structure:
```python
# Request
POST /api/ai-mashup
{
    "track_a": <file>,
    "track_b": <file>,
    "mashup_style": "dj_mix",  # or custom_instructions
    "llm_provider": "openai"
}

# Response
{
    "task_id": "abc123",
    "status": "processing",
    "websocket_url": "/ws/ai-mashup/abc123"
}

# WebSocket messages
{
    "type": "progress",
    "message": "Analyzing tracks...",
    "percentage": 25
}

# Result
GET /api/ai-mashup/abc123
{
    "status": "success",
    "mashup_url": "/downloads/mashup_abc123.wav",
    "duration": 180.5,
    "plan": {...},
    "execution_log": [...]
}
```

### Files to Create:
- `app/routers/ai_mashup.py` - API routes
- `app/services/ai_mashup_service.py` - Business logic
- `app/tasks/mashup_tasks.py` - Background task definitions (if using Celery)
- `app/schemas/ai_mashup.py` - Pydantic request/response models
- Update `app/main.py` - Register new router

---

## ⏳ **Phase 5: Validation & Safety** (PENDING)

**Objective**: Test and refine the system with real-world tracks

### Planned Tasks:
1. **Create test dataset**
   - Collect 20-30 diverse track pairs
   - Various genres, BPMs, keys
   - Different compatibility levels

2. **Quality metrics**
   - Define success criteria (harmonic compatibility, smoothness, creativity)
   - Create evaluation rubric
   - Measure output quality

3. **Prompt iteration**
   - Test each style with multiple track combinations
   - Refine prompts based on results
   - A/B test prompt variations
   - Document what works best

4. **Safety testing**
   - Test parameter edge cases
   - Verify all operations are safe
   - Check memory limits
   - Test with corrupted/unusual audio files

5. **LLM comparison**
   - Compare OpenAI vs Anthropic vs Local models
   - Measure success rates per provider
   - Document quality differences
   - Optimize model selection

### Success Metrics:
- [ ] 80%+ success rate on diverse track pairs
- [ ] Harmonic compatibility in 90%+ of outputs
- [ ] No crashes or undefined behavior
- [ ] Predictable results for similar inputs
- [ ] Clear error messages for failures

---

## ⏳ **Phase 6: Testing & Production Readiness** (PENDING)

**Objective**: Comprehensive testing and production hardening

### Planned Tasks:
1. **Unit tests**
   - Test each module independently
   - Mock LLM responses
   - Test edge cases
   - Target 80%+ code coverage

2. **Integration tests**
   - Full workflow tests (load → analyze → plan → execute → save)
   - Test different LLM providers
   - Test all mashup styles
   - Test custom instructions

3. **Performance optimization**
   - Profile memory usage
   - Optimize audio processing bottlenecks
   - Cache expensive operations (feature extraction)
   - Parallel processing where possible

4. **Error handling**
   - Graceful degradation
   - Retry logic for transient failures
   - Clear error messages
   - Rollback on partial failures

5. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - User guide
   - Developer guide
   - Troubleshooting guide

### Files to Create:
- `tests/unit/test_models.py`
- `tests/unit/test_executor.py`
- `tests/unit/test_validators.py`
- `tests/integration/test_producer.py`
- `tests/integration/test_api.py`
- `docs/API.md`
- `docs/TROUBLESHOOTING.md`

---

## ⏳ **Phase 7: Advanced Features** (OPTIONAL)

**Objective**: Enhance user experience and system intelligence

### Planned Features:
1. **User feedback loop**
   - Rate generated mashups (1-5 stars)
   - Report issues with specific plans
   - Request plan regeneration
   - Learn from user preferences

2. **Plan caching**
   - Cache plans for similar track combinations
   - Avoid redundant LLM calls
   - Faster processing for repeated requests
   - Reduce API costs

3. **Multiple plan generation**
   - Generate 2-3 different plans per request
   - Let user preview and choose
   - Compare different creative approaches
   - Increase user satisfaction

4. **Smart plan selection**
   - Analyze track features to recommend best style
   - Auto-select LLM provider based on task
   - Predict quality before execution
   - Suggest parameter adjustments

5. **Analytics & monitoring**
   - Track success rates
   - Monitor LLM usage and costs
   - Popular styles and features
   - Error patterns

6. **Advanced customization**
   - Style blending (e.g., 70% dj_mix + 30% creative)
   - Section-specific instructions
   - Multi-track mashups (3+ tracks)
   - Live parameter adjustment

---

## Architecture Overview

### System Flow:
```
1. User Input
   ↓
2. Load & Analyze Tracks (librosa + audio_analysis)
   ↓
3. Generate Plan (LLM via client.py)
   ↓
4. Validate Plan (validators.py)
   ↓
5. Execute Plan (executor.py + audio_state.py)
   ↓
6. Return Mashup (producer.py)
```

### Module Dependencies:
```
producer.py (orchestrator)
├── models.py (data structures)
├── prompts.py (LLM instructions)
├── client.py (LLM communication)
├── executor.py (operation execution)
│   └── utils/* (148 audio functions)
├── audio_state.py (buffer management)
└── validators.py (safety checks)
```

### Key Design Principles:
1. **Separation of Concerns** - Each module has a single responsibility
2. **Safety First** - All parameters validated and clamped before execution
3. **Stateless LLM** - LLM generates plan, system executes deterministically
4. **Fail Gracefully** - Errors don't crash the system
5. **Observable** - Detailed logging and execution tracking

---

## Technology Stack

### Core Dependencies:
- **FastAPI** - Web framework (for Phase 4+)
- **Pydantic 2.0+** - Data validation and serialization
- **librosa** - Audio loading and basic analysis
- **soundfile** - Audio file I/O
- **numpy** - Audio data manipulation
- **openai** - OpenAI API client
- **anthropic** - Anthropic API client
- **requests** - HTTP client for local models

### Utility Modules (already implemented):
- `app/utils/key_manipulations.py` - 15 functions
- `app/utils/bpm_manipulations.py` - 12 functions
- `app/utils/volume_manipulations.py` - 18 functions
- `app/utils/mixing_manipulations.py` - 24 functions
- `app/utils/audio_analysis.py` - 22 functions
- `app/utils/effects_manipulations.py` - 31 functions
- `app/utils/transition_manipulations.py` - 26 functions
- **Total: 148 utility functions**

---

## File Structure

```
app/
├── llm/
│   ├── __init__.py              ✅ Package initialization
│   ├── models.py                ✅ Pydantic schemas (10 models)
│   ├── prompts.py               ✅ Prompt templates (5 styles)
│   ├── client.py                ✅ LLM client (3 providers)
│   ├── executor.py              ✅ Operation executor
│   ├── audio_state.py           ✅ Audio buffer management
│   ├── validators.py            ✅ Parameter validation
│   ├── producer.py              ✅ Main orchestrator
│   ├── USAGE_EXAMPLES.md        ✅ Usage documentation
│   └── planning/
│       └── plan.md              ✅ This file
├── utils/                       ✅ 148 utility functions (7 modules)
├── routers/
│   └── ai_mashup.py             ⏳ API routes (Phase 4)
├── services/
│   └── ai_mashup_service.py     ⏳ Business logic (Phase 4)
├── schemas/
│   └── ai_mashup.py             ⏳ API models (Phase 4)
└── main.py                      ⏳ Update for new routes (Phase 4)

tests/                           ⏳ Test suite (Phase 6)
├── unit/
│   ├── test_models.py
│   ├── test_executor.py
│   ├── test_validators.py
│   └── test_audio_state.py
└── integration/
    ├── test_producer.py
    └── test_api.py

docs/                            ⏳ Documentation (Phase 6)
├── API.md
├── USER_GUIDE.md
└── TROUBLESHOOTING.md
```

---

## Success Criteria

### Phase 1-3 (Core) ✅ COMPLETED
- [x] All 8 core modules created and documented
- [x] Support for 4 predefined styles + custom instructions
- [x] Multi-provider LLM support (OpenAI, Anthropic, local)
- [x] Complete operation registry (90+ functions)
- [x] Comprehensive validation and safety system
- [x] Main orchestrator working end-to-end
- [x] Usage examples and documentation

### Phase 4 (API) ⏳ PENDING
- [ ] Working FastAPI endpoint
- [ ] Async processing with progress tracking
- [ ] WebSocket real-time updates
- [ ] File upload and download handling
- [ ] Plan preview without execution

### Phase 5 (Validation) ⏳ PENDING
- [ ] 80%+ success rate on diverse tracks
- [ ] Refined prompts for each style
- [ ] LLM provider comparison completed
- [ ] Safety testing passed

### Phase 6 (Testing) ⏳ PENDING
- [ ] 80%+ test coverage
- [ ] All integration tests passing
- [ ] Performance benchmarks documented
- [ ] Complete API documentation

### Phase 7 (Advanced) ⏳ OPTIONAL
- [ ] User feedback system
- [ ] Plan caching implemented
- [ ] Multiple plan generation
- [ ] Analytics dashboard

---

## Next Steps

### Immediate (Phase 4):
1. Create FastAPI endpoint for AI mashup creation
2. Implement async processing with background tasks
3. Add WebSocket for progress updates
4. Create file upload/download handlers
5. Add plan preview endpoint

### Short-term (Phase 5):
1. Collect test dataset (20-30 track pairs)
2. Run quality evaluation on all styles
3. Iterate on prompts based on results
4. Compare LLM providers
5. Document best practices

### Long-term (Phase 6-7):
1. Build comprehensive test suite
2. Optimize performance and memory usage
3. Create production deployment guide
4. Add advanced features (feedback, caching, etc.)
5. Build analytics and monitoring

---

## Resources

### Documentation:
- [USAGE_EXAMPLES.md](../USAGE_EXAMPLES.md) - How to use the system programmatically
- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [Anthropic API Docs](https://docs.anthropic.com/claude/reference)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

### Utility Modules:
- See `app/utils/` for all 148 audio processing functions
- Each module has comprehensive docstrings

### Related Files:
- `requirements.txt` - Python dependencies
- `app/main.py` - FastAPI application entry point
- `README.md` - Project overview (to be updated)

---

## Changelog

### 2025-10-11
- ✅ Completed Phase 1: Foundation (models, prompts, package structure)
- ✅ Completed Phase 2: LLM Integration (multi-provider client)
- ✅ Completed Phase 3: Execution Engine (executor, state, validators, producer)
- ✅ Added custom instructions support (free-form text descriptions)
- ✅ Created comprehensive usage examples
- ✅ Created implementation plan documentation

### Next Update:
- Phase 4: API Integration (target: TBD)

---

**Last Updated**: October 11, 2025  
**Status**: Ready for Phase 4 (API Integration)  
**Contributors**: MAI Team
