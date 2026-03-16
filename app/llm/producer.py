"""
AI Mashup Producer - Main Orchestrator

This is the main entry point for AI-guided mashup creation.
It coordinates all components: analysis, LLM planning, and execution.

Workflow:
1. Load and analyze both tracks (extract features)
2. Send features to LLM to generate mashup plan
3. Validate the plan
4. Execute the plan step-by-step
5. Finalize and export the mashup

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np

# Import utils
from app.utils.audio_analysis import analyze_track

# Import LLM modules
from app.llm.models import (
    MashupPlan,
    MashupResult,
    ExecutionLog,
    TrackFeatures,
    TwoTrackAnalysis,
    PreprocessingStep,
    MixStep
)
from app.llm.prompts import get_prompt_for_style, build_user_prompt
from app.llm.client import create_llm_client
from app.llm.executor import OperationExecutor, extract_section
from app.llm.audio_state import AudioState, concatenate_sections
from app.llm.validators import PlanValidator, sanitize_parameters, validate_track_id

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN PRODUCER CLASS
# =============================================================================

class AIMashupProducer:
    """
    AI-powered mashup producer.
    
    Uses LLM to analyze tracks and generate intelligent mashup plans,
    then executes the plans using professional audio utilities.
    """
    
    def __init__(
        self,
        sr: int = 44100,
        llm_provider: str = 'openai',
        llm_model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize AI Mashup Producer.
        
        Args:
            sr: Sample rate for audio processing
            llm_provider: LLM provider ('openai', 'anthropic', 'local')
            llm_model: LLM model name (uses default if None)
            temperature: LLM temperature (0-1, higher = more creative)
        """
        self.sr = sr
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize components
        self.audio_state = AudioState(sr=sr)
        self.executor = OperationExecutor(sr=sr)
        
        # Create LLM client
        self.llm_client = create_llm_client(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature
        )
        
        logger.info(f"AI Mashup Producer initialized ({llm_provider}, sr={sr}Hz)")
    
    def create_mashup(
        self,
        track_a_audio: np.ndarray,
        track_b_audio: np.ndarray,
        track_a_name: str = "Track A",
        track_b_name: str = "Track B",
        mashup_style: str = 'dj_mix',
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> MashupResult:
        """
        Create a mashup from two tracks.
        
        Args:
            track_a_audio: Audio data for Track A
            track_b_audio: Audio data for Track B
            track_a_name: Name of Track A
            track_b_name: Name of Track B
            mashup_style: Mashup style ('dj_mix', 'creative_mashup', 'remix', 'quick_mix')
                         Ignored if custom_instructions is provided
            custom_instructions: Free-form text description of desired mashup
                                (e.g., "Create a high-energy festival drop with heavy bass")
            progress_callback: Optional callback for progress updates (message, percentage)
        
        Returns:
            MashupResult with final mashup and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Load tracks
            self._update_progress(progress_callback, "🎵 Phase 1/5: Loading audio tracks...", 5)
            self.audio_state.set_track_a(track_a_audio, track_a_name)
            self.audio_state.set_track_b(track_b_audio, track_b_name)
            
            # Step 2: Analyze tracks
            self._update_progress(progress_callback, "🔍 Phase 2/5: Analyzing Track A (detecting BPM, key, structure)...", 10)
            track_a_features = self._analyze_track_features(track_a_audio, track_a_name)
            
            self._update_progress(progress_callback, "🔍 Phase 2/5: Analyzing Track B (detecting BPM, key, structure)...", 20)
            track_b_features = self._analyze_track_features(track_b_audio, track_b_name)
            
            # Step 3: Generate mashup plan from LLM
            self._update_progress(progress_callback, "🤖 Phase 3/5: Generating mashup plan with AI (waiting for LLM response)...", 30)
            plan = self._generate_plan(
                track_a_features,
                track_b_features,
                mashup_style,
                custom_instructions
            )
            
            # Step 4: Validate plan
            self._update_progress(progress_callback, "✅ Phase 4/5: Validating mashup plan...", 35)
            validator = PlanValidator(
                track_a_duration=track_a_features['duration'],
                track_b_duration=track_b_features['duration'],
                strict=False
            )
            
            if not validator.validate_plan(plan):
                logger.warning("Plan validation failed, attempting fallback template")
                warnings = validator.get_warnings()
                errors = validator.get_errors()
                
                # Try fallback template if validation failed
                logger.info("Generating fallback template-based mashup plan")
                plan = self._create_fallback_plan(track_a_features, track_b_features)
                
                # Validate fallback plan
                if not validator.validate_plan(plan):
                    logger.error("Even fallback plan validation failed")
                    return MashupResult(
                        status='failed',
                        plan_used=plan,
                        execution_log=[],
                        errors=errors + ["Fallback plan also failed validation"]
                    )
                else:
                    logger.info("Using fallback template plan")
            
            # Step 5: Execute plan
            self._update_progress(progress_callback, "⚙️ Phase 5/5: Processing audio (executing mashup plan)...", 40)
            success = self._execute_plan(plan, progress_callback)
            
            if not success:
                return MashupResult(
                    status='failed',
                    plan_used=plan,
                    execution_log=self._get_execution_log(),
                    errors=["Execution failed"]
                )
            
            # Step 6: Finalize
            self._update_progress(progress_callback, "🎉 Finalizing mashup (applying final touches)...", 95)
            mashup_audio = self.audio_state.mashup
            
            if mashup_audio is None:
                logger.error("No mashup audio generated")
                return MashupResult(
                    status='failed',
                    plan_used=plan,
                    execution_log=self._get_execution_log(),
                    errors=["No mashup audio generated"]
                )
            
            # Success!
            self._update_progress(progress_callback, "✨ Mashup ready! Download available.", 100)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Mashup created successfully in {elapsed_time:.2f}s")
            
            return MashupResult(
                status='success',
                mashup_file=None,  # Will be set by caller when saved
                plan_used=plan,
                execution_log=self._get_execution_log(),
                total_duration=len(mashup_audio) / self.sr,
                errors=[]
            )
        
        except Exception as e:
            logger.error(f"Mashup creation failed: {e}", exc_info=True)
            return MashupResult(
                status='failed',
                plan_used=None,
                execution_log=self._get_execution_log(),
                errors=[str(e)]
            )
    
    def _analyze_track_features(self, audio: np.ndarray, name: str) -> Dict[str, Any]:
        """Analyze track and extract features."""
        features = analyze_track(audio, self.sr)
        features['name'] = name
        return features
    
    def _generate_plan(
        self,
        track_a_features: Dict[str, Any],
        track_b_features: Dict[str, Any],
        mashup_style: str,
        custom_instructions: Optional[str] = None
    ) -> MashupPlan:
        """Generate mashup plan using LLM."""
        # Get appropriate prompt template
        system_prompt = get_prompt_for_style(mashup_style, custom_instructions)
        
        # Build user prompt with track features
        user_prompt = build_user_prompt(
            track_a_features,
            track_b_features,
            mashup_style,
            custom_instructions
        )
        
        # Generate plan from LLM
        if custom_instructions:
            logger.info(f"Requesting custom mashup plan from LLM: '{custom_instructions[:50]}...'")
        else:
            logger.info(f"Requesting mashup plan from LLM ({mashup_style} style)...")
        
        plan_json = self.llm_client.generate_json(system_prompt, user_prompt)
        
        # Parse into Pydantic model
        plan = MashupPlan(**plan_json)
        
        logger.info(f"LLM generated plan with {len(plan.preprocessing)} preprocessing steps, "
                   f"{len(plan.mix_plan)} mix steps, {len(plan.creative_enhancements)} enhancements")
        
        return plan
    
    def _create_fallback_plan(
        self,
        track_a_features: Dict[str, Any],
        track_b_features: Dict[str, Any]
    ) -> MashupPlan:
        """
        Create a simple, safe fallback mashup plan when LLM generation fails.
        Uses an ABABCB structure (intro-verse-chorus-verse-chorus-bridge-chorus-outro).
        """
        logger.info("Creating fallback template-based mashup plan")
        
        # Get sections from both tracks
        sections_a = track_a_features.get('sections', [])
        sections_b = track_b_features.get('sections', [])
        
        # Preprocessing: Match keys if needed (simple approach)
        preprocessing = []
        key_diff = track_b_features.get('key', 0) - track_a_features.get('key', 0)
        if abs(key_diff) > 2 and abs(key_diff) < 10:
            # Transpose track B to match track A (but not extreme)
            semitones = key_diff if abs(key_diff) <= 6 else (key_diff - 12 if key_diff > 0 else key_diff + 12)
            preprocessing.append(PreprocessingStep(
                track='b',
                operation='transpose',
                parameters={'semitones': int(semitones)},
                reason=f'Match key (transpose by {semitones:+d} semitones)'
            ))
        
        # Build mix plan with ABABCB structure
        mix_plan = []
        step_num = 1
        current_time = 0.0
        
        def get_safe_section(sections, section_type, max_duration=30.0):
            """Get a section of the specified type, limiting to max_duration."""
            for sec in sections:
                if sec['type'] == section_type:
                    duration = sec['end'] - sec['start']
                    if duration > max_duration:
                        # Trim to max duration
                        return sec['start'], sec['start'] + max_duration
                    return sec['start'], sec['end']
            # Fallback: use first available section
            if sections:
                sec = sections[0]
                duration = min(sec['end'] - sec['start'], max_duration)
                return sec['start'], sec['start'] + duration
            return 0.0, min(max_duration, track_a_features.get('duration', 30.0))
        
        # 1. Intro from Track A (intro or first section)
        start, end = get_safe_section(sections_a, 'intro', 20.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'a', 'start_time': start, 'end_time': end},
            reasoning='Intro from Track A'
        ))
        current_time += (end - start)
        step_num += 1
        
        # 2. Verse/Chorus from Track B
        start, end = get_safe_section(sections_b, 'verse', 25.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'b', 'start_time': start, 'end_time': end},
            reasoning='Verse from Track B'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 3. Chorus from Track A
        start, end = get_safe_section(sections_a, 'chorus', 25.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'a', 'start_time': start, 'end_time': end},
            reasoning='Chorus from Track A'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 4. Verse from Track B
        start, end = get_safe_section(sections_b, 'verse', 25.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'b', 'start_time': start, 'end_time': end},
            reasoning='Verse from Track B'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 5. Chorus from Track A
        start, end = get_safe_section(sections_a, 'chorus', 25.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'a', 'start_time': start, 'end_time': end},
            reasoning='Chorus from Track A (repeat)'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 6. Bridge from Track B (or outro)
        start, end = get_safe_section(sections_b, 'bridge', 20.0)
        if start == 0.0:  # No bridge found, use outro
            start, end = get_safe_section(sections_b, 'outro', 20.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'b', 'start_time': start, 'end_time': end},
            reasoning='Bridge/transition section'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 7. Final chorus from Track A
        start, end = get_safe_section(sections_a, 'chorus', 25.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'a', 'start_time': start, 'end_time': end},
            reasoning='Final chorus'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='crossfade',
            parameters={'duration': 2.0},
            reasoning='Smooth transition'
        ))
        current_time += (end - start)
        step_num += 2
        
        # 8. Outro from Track A
        start, end = get_safe_section(sections_a, 'outro', 15.0)
        mix_plan.append(MixStep(
            step=step_num,
            operation='extract',
            parameters={'track': 'a', 'start_time': start, 'end_time': end},
            reasoning='Outro'
        ))
        mix_plan.append(MixStep(
            step=step_num + 1,
            operation='fade_out',
            parameters={'duration': 3.0},
            reasoning='Smooth ending'
        ))
        
        # Minimal creative enhancements (safe approach)
        creative_enhancements = []
        
        plan = MashupPlan(
            preprocessing=preprocessing,
            mix_plan=mix_plan,
            creative_enhancements=creative_enhancements,
            reasoning='Fallback template: ABABCB structure with safe 20-30s sections and smooth crossfades'
        )
        
        logger.info(f"Fallback plan created with {len(mix_plan)} steps")
        return plan
    
    def _execute_plan(
        self,
        plan: MashupPlan,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> bool:
        """Execute the mashup plan."""
        try:
            # Execute preprocessing
            total_steps = len(plan.preprocessing) + len(plan.mix_plan) + len(plan.creative_enhancements)
            current_step = 0
            
            for prep_step in plan.preprocessing:
                current_step += 1
                progress = 40 + int((current_step / total_steps) * 50)
                self._update_progress(
                    progress_callback,
                    f"⚙️ Preprocessing Track {prep_step.track.upper()}: {prep_step.operation}",
                    progress
                )
                
                self._execute_preprocessing_step(prep_step)
            
            # Execute mix plan
            mashup_sections = []
            
            for mix_step in plan.mix_plan:
                current_step += 1
                progress = 40 + int((current_step / total_steps) * 50)
                self._update_progress(
                    progress_callback,
                    f"🎛️ Processing Step {mix_step.step}/{len(plan.mix_plan)}: {mix_step.action}",
                    progress
                )
                
                result = self._execute_mix_step(mix_step, mashup_sections)
                
                # Handle different operation types for section management
                if 'crossfade' in mix_step.operation or mix_step.operation == 'blend_tracks':
                    # Crossfade/blend operations consume the last 2 sections and produce 1 result
                    # Remove the last 2 sections and add the result
                    if len(mashup_sections) >= 2 and result is not None:
                        mashup_sections.pop()  # Remove last section
                        mashup_sections.pop()  # Remove second-to-last section
                        mashup_sections.append(result)  # Add blended result
                    elif result is not None:
                        # Shouldn't happen if validation is correct, but handle gracefully
                        mashup_sections.append(result)
                elif result is not None:
                    # Other operations (extract_section, effects, etc.) just append
                    mashup_sections.append(result)
            
            # Concatenate all sections to create final mashup
            if mashup_sections:
                final_mashup = concatenate_sections(mashup_sections)
                self.audio_state.set_mashup(final_mashup)
            
            # Apply creative enhancements
            for enhancement in plan.creative_enhancements:
                current_step += 1
                progress = 40 + int((current_step / total_steps) * 50)
                self._update_progress(
                    progress_callback,
                    f"✨ Applying creative enhancement: {enhancement.effect}",
                    progress
                )
                
                self._execute_enhancement(enhancement)
            
            return True
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            return False
    
    def _execute_preprocessing_step(self, step):
        """Execute a preprocessing step."""
        track_id = validate_track_id(step.track)
        audio = self.audio_state.get_track(track_id)
        
        if audio is None:
            logger.error(f"Track {track_id} not found")
            return
        
        # Sanitize parameters
        params = sanitize_parameters(step.parameters)
        
        # Execute operation
        result = self.executor.execute_operation(step.operation, params, audio)
        
        # Update track
        if track_id == 'a':
            self.audio_state.update_track_a(result, step.operation)
        else:
            self.audio_state.update_track_b(result, step.operation)
    
    def _execute_mix_step(self, step, current_sections):
        """Execute a mix plan step."""
        params = sanitize_parameters(step.parameters)
        
        # Handle different operation types
        if step.operation == 'extract_section':
            # Extract section from track
            track_id = validate_track_id(params.get('track', 'a'))
            audio = self.audio_state.get_track(track_id)
            
            if audio is None:
                logger.error(f"Track {track_id} not found")
                return None
            
            start = params.get('start', params.get('start_time', 0))
            end = params.get('end', params.get('end_time', len(audio) / self.sr))
            
            section = extract_section(audio, self.sr, start, end)
            
            # Store section
            section_name = f"section_{step.step}"
            self.audio_state.add_section(section_name, section)
            
            return section
        
        elif 'crossfade' in step.operation:
            # Crossfade between last two sections
            if len(current_sections) < 2:
                logger.warning(f"Need at least 2 sections to crossfade, have {len(current_sections)}")
                return current_sections[-1] if current_sections else None
            
            # Get last two sections
            track_a = current_sections[-2]
            track_b = current_sections[-1]
            
            # Add track_a and track_b to params
            params['track_a'] = track_a
            params['track_b'] = track_b
            
            # Get crossfade function from executor
            result = self.executor.execute_operation(step.operation, params, None)
            return result
        
        elif step.operation == 'blend_tracks':
            # Blend multiple tracks/sections
            # This is complex - for now, just mix last two sections
            if len(current_sections) >= 2:
                from app.llm.audio_state import blend_sections
                return blend_sections([current_sections[-2], current_sections[-1]])
            return current_sections[-1] if current_sections else None
        
        else:
            # Apply operation to last section
            if current_sections:
                audio = current_sections[-1]
                result = self.executor.execute_operation(step.operation, params, audio)
                return result
            return None
    
    def _execute_enhancement(self, enhancement):
        """Execute a creative enhancement."""
        params = sanitize_parameters(enhancement.parameters)
        
        # Determine target
        if enhancement.target == 'mix' and self.audio_state.mashup is not None:
            audio = self.audio_state.mashup
            result = self.executor.execute_operation(enhancement.effect, params, audio)
            self.audio_state.set_mashup(result)
        
        elif enhancement.target in ['track_a', 'track_b']:
            track_id = 'a' if enhancement.target == 'track_a' else 'b'
            audio = self.audio_state.get_track(track_id)
            
            if audio is not None:
                result = self.executor.execute_operation(enhancement.effect, params, audio)
                
                if track_id == 'a':
                    self.audio_state.update_track_a(result, enhancement.effect)
                else:
                    self.audio_state.update_track_b(result, enhancement.effect)
    
    def _get_execution_log(self) -> list:
        """Get execution log."""
        return [
            ExecutionLog(
                step=i + 1,
                operation=log['operation'],
                status=log['status'],
                message=log['message']
            )
            for i, log in enumerate(self.executor.get_log())
        ]
    
    def _update_progress(
        self,
        callback: Optional[Callable[[str, int], None]],
        message: str,
        percentage: int
    ):
        """Update progress."""
        logger.info(f"[{percentage}%] {message}")
        if callback:
            try:
                callback(message, percentage)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def get_mashup_audio(self) -> Optional[np.ndarray]:
        """Get the final mashup audio."""
        return self.audio_state.mashup
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get audio state summary."""
        return self.audio_state.get_summary()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mashup(
    track_a_audio: np.ndarray,
    track_b_audio: np.ndarray,
    track_a_name: str = "Track A",
    track_b_name: str = "Track B",
    mashup_style: str = 'dj_mix',
    custom_instructions: Optional[str] = None,
    sr: int = 44100,
    llm_provider: str = 'openai',
    llm_model: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Tuple[Optional[np.ndarray], MashupResult]:
    """
    Convenience function to create a mashup in one call.
    
    Args:
        track_a_audio: Audio data for Track A
        track_b_audio: Audio data for Track B
        track_a_name: Name of Track A
        track_b_name: Name of Track B
        mashup_style: Mashup style ('dj_mix', 'creative_mashup', 'remix', 'quick_mix')
                      Ignored if custom_instructions is provided
        custom_instructions: Free-form text description of desired mashup
                            (e.g., "Create a high-energy festival drop with heavy bass")
        sr: Sample rate
        llm_provider: LLM provider
        llm_model: LLM model name
        progress_callback: Progress callback
    
    Returns:
        Tuple of (mashup_audio, result_metadata)
    
    Example:
        >>> import librosa
        >>> track_a, sr = librosa.load("song_a.mp3")
        >>> track_b, _ = librosa.load("song_b.mp3")
        >>> 
        >>> def progress(msg, pct):
        >>>     print(f"[{pct}%] {msg}")
        >>> 
        >>> # Using predefined style
        >>> mashup, result = create_mashup(
        >>>     track_a, track_b,
        >>>     mashup_style='dj_mix',
        >>>     progress_callback=progress
        >>> )
        >>> 
        >>> # Using custom instructions
        >>> mashup, result = create_mashup(
        >>>     track_a, track_b,
        >>>     custom_instructions="Create a high-energy festival mashup with heavy bass drops and dramatic buildups",
        >>>     progress_callback=progress
        >>> )
        >>> 
        >>> if result.status == 'success':
        >>>     import soundfile as sf
        >>>     sf.write("mashup.wav", mashup, sr)
    """
    producer = AIMashupProducer(
        sr=sr,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    result = producer.create_mashup(
        track_a_audio=track_a_audio,
        track_b_audio=track_b_audio,
        track_a_name=track_a_name,
        track_b_name=track_b_name,
        mashup_style=mashup_style,
        custom_instructions=custom_instructions,
        progress_callback=progress_callback
    )
    
    mashup_audio = producer.get_mashup_audio()
    
    return mashup_audio, result


if __name__ == '__main__':
    print("AI Mashup Producer - Main Orchestrator")
    print("\nThis is the main entry point for AI-guided mashup creation.")
    print("\nUsage:")
    print("  from app.llm.producer import create_mashup")
    print("\n  # Predefined style:")
    print("  mashup, result = create_mashup(track_a, track_b, mashup_style='dj_mix')")
    print("\n  # Custom instructions:")
    print("  mashup, result = create_mashup(")
    print("      track_a, track_b,")
    print('      custom_instructions="Create a high-energy festival mashup with heavy bass drops"')
    print("  )")
    print("\nAvailable predefined styles:")
    print("  - dj_mix: Seamless DJ-style mix")
    print("  - creative_mashup: Experimental, artistic")
    print("  - remix: Feature Track A, enhance with Track B")
    print("  - quick_mix: Simple, fast")
    print("\nOr use custom_instructions for any custom description!")
