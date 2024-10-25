from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class LLMResponse:
    content: str
    token_count: int

class ModelConfigError(Exception):
    """Raised when there are issues with model configuration"""
    pass

class LLMClientError(Exception):
    """Raised when there are issues with LLM client operations"""
    pass

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

def get_config_path(config_file: str) -> Path:
    """Returns the full path to a config file."""
    return get_project_root() / "configs" / config_file