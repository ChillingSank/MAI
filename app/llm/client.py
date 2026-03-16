"""
LLM Client Abstraction Layer

Provides unified interface for different LLM providers (OpenAI, Anthropic, local models).
Includes retry logic, error handling, and response parsing.

Author: MAI Team
Date: 2025-10-11
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# BASE CLIENT
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response from LLM."""
        pass


# =============================================================================
# OPENAI CLIENT
# =============================================================================

class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4, GPT-4-turbo, GPT-4o-mini)."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",  # Default to GPT-4o-mini
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        super().__init__(model, temperature)
        self.api_key = "" or api_key or os.getenv('OPENAI_API_KEY')
        self.max_retries = max_retries
        self.token_limit = 8000  # Set token limit for gpt-4o-mini
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Write prompt to file before sending
        self._save_prompt_to_file(system_prompt, user_prompt)

        # Check token limits
        total_tokens = sum(len(m["content"].split()) for m in messages)
        if total_tokens > self.token_limit:
            raise ValueError(f"Request exceeds token limit ({self.token_limit}). Reduce input size.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **kwargs
            )
            content = response.choices[0].message.content
            
            # Save LLM output to file
            self._save_llm_output(content)
            
            return content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response with structured output."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Write prompt to file before sending
        self._save_prompt_to_file(system_prompt, user_prompt)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},  # Force JSON
                    **kwargs
                )
                
                content = response.choices[0].message.content
                
                # Save LLM output to file
                self._save_llm_output(content)
                
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

    def _save_prompt_to_file(self, system_prompt: str, user_prompt: str):
        """Save the prompt to a text file for debugging."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"OPENAI API REQUEST - {timestamp}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write("="*80 + "\n\n")
                
                f.write("="*80 + "\n")
                f.write("SYSTEM PROMPT\n")
                f.write("="*80 + "\n")
                f.write(system_prompt)
                f.write("\n\n")
                
                f.write("="*80 + "\n")
                f.write("USER PROMPT\n")
                f.write("="*80 + "\n")
                f.write(user_prompt)
                f.write("\n\n")
                
                # Token estimation
                total_chars = len(system_prompt) + len(user_prompt)
                estimated_tokens = total_chars / 4
                f.write("="*80 + "\n")
                f.write("STATISTICS\n")
                f.write("="*80 + "\n")
                f.write(f"System prompt length: {len(system_prompt):,} characters\n")
                f.write(f"User prompt length: {len(user_prompt):,} characters\n")
                f.write(f"Total length: {total_chars:,} characters\n")
                f.write(f"Estimated tokens: {estimated_tokens:,.0f}\n")
            
            logger.info(f"Prompt saved to: {filename} ({estimated_tokens:,.0f} estimated tokens)")
        except Exception as e:
            logger.warning(f"Failed to save prompt to file: {e}")

    def _save_llm_output(self, output: str):
        """Save the LLM output to a text file for debugging."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_output_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"LLM OUTPUT - {timestamp}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write("="*80 + "\n\n")
                f.write(output)
                f.write("\n\n")
                f.write("="*80 + "\n")
                f.write("STATISTICS\n")
                f.write("="*80 + "\n")
                f.write(f"Output length: {len(output):,} characters\n")
                f.write(f"Estimated tokens: {len(output) / 4:,.0f}\n")
            
            logger.info(f"LLM output saved to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save LLM output to file: {e}")


# =============================================================================
# ANTHROPIC CLIENT
# =============================================================================

class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        super().__init__(model, temperature)
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text response."""
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
                max_tokens=4096,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response."""
        # Add JSON instruction to prompts
        system_prompt = f"{system_prompt}\n\nIMPORTANT: Return ONLY valid JSON, no markdown or extra text."
        user_prompt = f"{user_prompt}\n\nReturn your response as valid JSON."
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.temperature,
                    max_tokens=4096,
                    **kwargs
                )
                
                content = response.content[0].text
                
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                if attempt == self.max_retries - 1:
                    raise


# =============================================================================
# LOCAL MODEL CLIENT (Ollama, LM Studio)
# =============================================================================

class LocalModelClient(LLMClient):
    """Client for local LLMs via Ollama or LM Studio."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        super().__init__(model, temperature)
        self.base_url = base_url
        self.max_retries = max_retries
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text response via local API."""
        import requests
        
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Local model error: {e}")
            raise
    
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response."""
        system_prompt = f"{system_prompt}\n\nIMPORTANT: Return ONLY valid JSON, no markdown or extra text."
        user_prompt = f"{user_prompt}\n\nReturn your response as valid JSON."
        
        for attempt in range(self.max_retries):
            try:
                content = self.generate(system_prompt, user_prompt, **kwargs)
                
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")
            except Exception as e:
                logger.error(f"Local model error: {e}")
                if attempt == self.max_retries - 1:
                    raise


# =============================================================================
# HUGGING FACE CLIENT (FREE!)
# =============================================================================

class HuggingFaceClient(LLMClient):
    """
    Hugging Face Inference API client - FREE to use!
    
    Get your FREE API token at: https://huggingface.co/settings/tokens
    
    Recommended free models:
    - mistralai/Mistral-7B-Instruct-v0.2 (good quality, fast)
    - meta-llama/Llama-2-7b-chat-hf (Facebook's Llama)
    - HuggingFaceH4/zephyr-7b-beta (excellent for instructions)
    - tiiuae/falcon-7b-instruct (fast and efficient)
    """
    
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_token: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        super().__init__(model, temperature)
        self.api_token = "" or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        self.max_retries = max_retries
        
        if not self.api_token:
            raise ValueError(
                "Hugging Face API token not provided. "
                "Get a FREE token at https://huggingface.co/settings/tokens and set it as: "
                "export HUGGINGFACE_TOKEN='hf_...' or pass api_token parameter"
            )
        
        logger.info(f"Using Hugging Face model: {model}")
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text response via Hugging Face Inference API."""
        import requests
        
        API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Format prompt for instruction-following models
        full_prompt = f"""<s>[INST] {system_prompt}

{user_prompt} [/INST]"""
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": 2048,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    import time
                    logger.info(f"Model loading, waiting 10 seconds... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(10)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
                else:
                    return str(result)
                    
            except Exception as e:
                logger.warning(f"Hugging Face API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                import time
                time.sleep(2)
        
        raise RuntimeError("Failed to generate response from Hugging Face")
    
    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response."""
        system_prompt = f"{system_prompt}\n\nCRITICAL: Return ONLY valid JSON in your response, no markdown, no explanations."
        user_prompt = f"{user_prompt}\n\nReturn your response as pure JSON (no ```json markers)."
        
        for attempt in range(self.max_retries):
            try:
                content = self.generate(system_prompt, user_prompt, **kwargs)
                
                # Clean up response
                content = content.strip()
                
                # Remove markdown code blocks if present
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # Try to parse JSON
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}")
                logger.debug(f"Response content: {content[:500]}")
                
                if attempt == self.max_retries - 1:
                    # Last attempt - try to extract JSON with regex
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            return json.loads(json_match.group(0))
                        except:
                            pass
                    raise ValueError(f"Failed to parse JSON after {self.max_retries} attempts")
        
        raise RuntimeError("Failed to generate valid JSON")


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def create_llm_client(
    provider: Literal['openai', 'anthropic', 'local', 'huggingface'] = 'openai',
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Create LLM client based on provider.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'local')
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments for client
    
    Returns:
        LLM client instance
    
    Example:
        >>> client = create_llm_client('openai', model='gpt-4o-mini')
        >>> response = client.generate_json(system_prompt, user_prompt)
    """
    if provider == 'openai':
        default_model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        return OpenAIClient(model=default_model, **kwargs)
    
    elif provider == 'anthropic':
        default_model = model or os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
        return AnthropicClient(model=default_model, **kwargs)
    
    elif provider == 'local':
        default_model = model or os.getenv('LOCAL_MODEL', 'llama2')
        base_url = os.getenv('LOCAL_MODEL_URL', 'http://localhost:11434')
        return LocalModelClient(model=default_model, base_url=base_url, **kwargs)
    
    elif provider == 'huggingface':
        default_model = model or os.getenv('HUGGINGFACE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
        return HuggingFaceClient(model=default_model, **kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: openai, anthropic, local, huggingface")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_mashup_plan_json(
    system_prompt: str,
    user_prompt: str,
    provider: str = 'openai',
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate mashup plan JSON from LLM.
    
    Args:
        system_prompt: System prompt with instructions
        user_prompt: User prompt with track features
        provider: LLM provider
        model: Model name
        temperature: Temperature (0-1, higher = more creative)
    
    Returns:
        JSON dictionary with mashup plan
    
    Example:
        >>> plan = generate_mashup_plan_json(
        >>>     system_prompt=DJ_MIX_PROMPT,
        >>>     user_prompt=user_prompt,
        >>>     provider='openai'
        >>> )
    """
    client = create_llm_client(provider=provider, model=model, temperature=temperature)
    return client.generate_json(system_prompt, user_prompt)


if __name__ == '__main__':
    print("LLM Client Module")
    print("Supported providers: openai, anthropic, local")
    print("\nEnvironment variables:")
    print(f"  OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Not set'}")
    print(f"  ANTHROPIC_API_KEY: {'✓ Set' if os.getenv('ANTHROPIC_API_KEY') else '✗ Not set'}")
    print(f"  LOCAL_MODEL_URL: {os.getenv('LOCAL_MODEL_URL', 'Not set (default: http://localhost:11434)')}")
