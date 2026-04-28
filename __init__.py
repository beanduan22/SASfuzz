# SASFuzz — Synthesizing Models with LLMs for Fuzzing Deep Learning Libraries
#
# Quick-start:
#   from sasfuzz.backends.llm_client import OllamaClient, OpenAIClient, AnthropicClient
#   from sasfuzz.core.synthesizer    import ModelSynthesizer
#   from sasfuzz.core.executor       import ModelExecutor
#   from sasfuzz.core.oracle         import DifferentialOracle
#   from sasfuzz.core.selector       import MultiRouletteSelector
#   from sasfuzz.core.api_loader     import load_and_classify

from .core.api_loader   import load_and_classify, group_summary
from .backends.llm_client import LLMBackend, OllamaClient, OpenAIClient, AnthropicClient
from .core.synthesizer  import ModelSynthesizer
from .core.executor     import ModelExecutor
from .core.oracle       import DifferentialOracle
from .core.selector     import MultiRouletteSelector

__all__ = [
    "LLMBackend",
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "ModelSynthesizer",
    "ModelExecutor",
    "DifferentialOracle",
    "MultiRouletteSelector",
    "load_and_classify",
    "group_summary",
]
