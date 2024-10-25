import yaml, os
from pathlib import Path
from typing import Optional
from .utils import ModelConfigError, LLMClientError, get_config_path
import warnings
from .clients import OpenAIBasedClient, AnthropicClient, GoogleClient, OllamaClient

class LLMPortal:
    """Main interface for interacting with various LLM providers"""

    def __init__(self, model_name: str,
                model_directory_file: str = 'model_directory.yaml'):
        self.model_name = model_name
        self.system_context = None
        self._load_configuration(get_config_path(model_directory_file))
        self._initialize_client()

    def _load_configuration(self, model_directory_file: Path) -> None:
        """Load and validate configuration from YAML"""
        try:
            with open(model_directory_file) as f:
                self.model_directory = yaml.safe_load(f)

            self.model_provider = self.model_directory['model_providers'].get(self.model_name)
            if not self.model_provider:
                raise ModelConfigError(f'Unknown model {self.model_name}')

            self.endpoint_url = self.model_directory['endpoints'].get(self.model_provider)
            if not self.endpoint_url:
                raise ModelConfigError(f'Failed to find endpoint for {self.model_provider}')

        except Exception as e:
            raise ModelConfigError(f'Configuration error: {str(e)}')

    def _initialize_client(self) -> None:
        """Initialize the appropriate LLM client"""
        if self.model_provider != "localhost":
            try:
                self.api_key = os.environ[f"{self.model_provider.upper()}_API_KEY"]
            except KeyError:
                raise ModelConfigError(f'Cannot find API key for {self.model_provider}')

        if self.model_provider in ['Open_AI', 'Together_AI']:
            self.client = OpenAIBasedClient(self.api_key, self.endpoint_url, self.model_name)
        elif self.model_provider == 'Anthropic':
            self.client = AnthropicClient(self.api_key, self.model_name)
        elif self.model_provider == "Google":
            self.client = GoogleClient(self.api_key, self.model_name)
        elif self.model_provider == "localhost":
            self.client = OllamaClient(self.model_name)
        else:
            raise ModelConfigError(f'Unknown model provider {self.model_provider}')

    def set_context(self, system_context: str, **kwargs) -> None:
        """Set system context and additional parameters"""
        self.system_context = system_context
        self.kwargs = kwargs

    def __call__(self, prompt: str, temp: float = 0.2,
                 max_tokens: int = 1000) -> str:
        """Generate response from the LLM"""
        if not self.system_context:
            raise ValueError("System context not set. Call set_context() first.")

        try:
            response = self.client.generate(
                prompt=prompt,
                system_context=self.system_context,
                temperature=temp,
                max_tokens=max_tokens
            )

            if response.token_count > max_tokens:
                warnings.warn(
                    f"Response length ({response.token_count} tokens) exceeded "
                    f"max_tokens ({max_tokens}). Truncating..."
                )
                return response.content[:max_tokens]

            return response.content

        except LLMClientError as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class TestPortal:
    """Test harness for LLM evaluation"""

    def __init__(self, model_name: str, system_message: str,
                 rubric: str, temp: float = 0.2, max_tokens: int = 1000):
        self.portal = LLMPortal(model_name)
        self.portal.set_context(system_context=system_message)
        self.rubric = rubric
        self.temp = temp
        self.max_tokens = max_tokens

    def __call__(self, question: str, answer_key: Optional[str] = None) -> str:
        prompt = f"{self.rubric}\n{question}\n"
        if answer_key:
            prompt += answer_key
        return self.portal(prompt=prompt, temp=self.temp, max_tokens=self.max_tokens)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass