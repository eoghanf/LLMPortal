from .types import LLMResponse
from .utils import ModelConfigError, LLMClientError
from abc import ABC, abstractmethod

import openai
import anthropic
import google.generativeai as genai
import ollama

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, system_context: str,
                 temperature: float, max_tokens: int) -> LLMResponse:
        """Generate response from LLM"""
        pass


class OpenAIBasedClient(BaseLLMClient):
    """Client for OpenAI API compatible providers (OpenAI, TogetherAI)"""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str, system_context: str,
                 temperature: float, max_tokens: int) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                token_count=response.usage.completion_tokens
            )
        except Exception as e:
            raise LLMClientError(f"OpenAI API error: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude"""

    def __init__(self, api_key: str, model_name: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_context: str,
                 temperature: float, max_tokens: int) -> LLMResponse:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_context,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return LLMResponse(
                content=response.content[0].text,
                token_count=response.usage.output_tokens
            )
        except Exception as e:
            raise LLMClientError(f"Anthropic API error: {str(e)}")


class GoogleClient(BaseLLMClient):
    """Client for Google's GenAI models"""

    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_context: str,
                 temperature: float, max_tokens: int) -> LLMResponse:
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            response = self.client.generate_content(
                contents=[system_context, prompt],
                generation_config=generation_config
            )

            if response.candidates[0].finish_reason.RECITATION is not None:
                raise LLMClientError("Response contained recitation")

            return LLMResponse(
                content=response.text,
                token_count=response.usage_metadata.candidates_token_count
            )
        except Exception as e:
            raise LLMClientError(f"Google API error: {str(e)}")


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, system_context: str,
                 temperature: float, max_tokens: int) -> LLMResponse:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_context},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': temperature, 'num_predict': max_tokens}
            )
            return LLMResponse(
                content=response['message']['content'],
                token_count=response['eval_count']
            )
        except Exception as e:
            raise LLMClientError(f"Ollama error: {str(e)}")
