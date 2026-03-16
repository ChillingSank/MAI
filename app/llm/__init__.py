"""
AI-Powered Mashup Producer

This package provides LLM-guided mashup creation using the utility functions
from app.utils. The LLM analyzes tracks and generates execution plans.

Author: MAI Team
Date: 2025-10-11
"""

from .models import (
    CompatibilityAnalysis,
    PreprocessingStep,
    MashupStructure,
    MixStep,
    CreativeEnhancement,
    MashupPlan,
)

__all__ = [
    'CompatibilityAnalysis',
    'PreprocessingStep',
    'MashupStructure',
    'MixStep',
    'CreativeEnhancement',
    'MashupPlan',
]
