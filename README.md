# LLM Portal

A provider-agnostic Python library for interfacing with various Large Language Model (LLM) providers. This library provides a unified interface for both cloud-based LLMs (OpenAI, Anthropic, Google) and local models (via Ollama), making it easy to switch between different providers while maintaining consistent code.

## Features

- **Unified Interface**: Single consistent API for multiple LLM providers
- **Provider Support**:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Together AI
  - Local models via Ollama
- **Built-in Testing Framework**: Includes a TestPortal class for systematic LLM evaluation
- **Type Safety**: Full type hints for better IDE support and code reliability
- **Error Handling**: Comprehensive error handling and informative error messages
- **Configuration Management**: YAML-based model configuration

## Installation

```bash
pip install -r requirements.txt
```

## Prerequisites

1. API keys for the cloud providers you plan to use
2. Ollama installation (if using local models)
3. Python 3.8+

## Configuration

1. Create a `configs` directory in your project root
2. Add a `model_directory.yaml` file with your model configurations:

```yaml
model_providers:
  gpt-4: "Open_AI"
  claude-3-opus-20240229: "Anthropic"
  gemini-pro: "Google"
  mistral-7b: "localhost"
  mixtral-8x7b: "Together_AI"

endpoints:
  Open_AI: "https://api.openai.com/v1"
  Anthropic: "https://api.anthropic.com"
  Google: ""
  Together_AI: "https://api.together.xyz/v1"
  localhost: "http://localhost:11434"
```

3. Set up environment variables for API keys:
```bash
export OPEN_AI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export TOGETHER_AI_API_KEY="your-together-key"
```

## Basic Usage

### Simple Generation

```python
from llm_portal import LLMPortal

# Initialize the portal with your chosen model
portal = LLMPortal("gpt-4")

# Set the system context
portal.set_context("You are a helpful assistant specializing in Python programming.")

# Generate a response
response = portal(
    prompt="What are the key differences between Python lists and tuples?",
    temp=0.2,  # Optional: Set temperature (default: 0.2)
    max_tokens=1000  # Optional: Set max tokens (default: 1000)
)

print(response)
```

### Using the Test Portal

The TestPortal class is designed for systematic evaluation of LLM responses:

```python
from llm_portal import TestPortal

# Initialize test portal
test_portal = TestPortal(
    model_name="claude-3-opus-20240229",
    system_message="You are an expert evaluator of mathematical solutions.",
    rubric="Grade the following answer on a scale of 1-10. Provide specific feedback.",
    temp=0.2,
    max_tokens=1000
)

# Evaluate an answer
question = "What is the derivative of x^2?"
answer_key = "2x"
evaluation = test_portal(question, answer_key)

print(evaluation)
```

### Using Context Manager

Both LLMPortal and TestPortal support the context manager protocol:

```python
with LLMPortal("gemini-pro") as portal:
    portal.set_context("You are a helpful assistant.")
    response = portal("Tell me about Python's context managers.")
```

## Error Handling

The library provides custom exceptions for better error handling:

```python
from llm_portal import LLMPortal, ModelConfigError, LLMClientError

try:
    portal = LLMPortal("unknown-model")
except ModelConfigError as e:
    print(f"Configuration error: {e}")
except LLMClientError as e:
    print(f"Client error: {e}")
```

## Extending the Library

To add support for a new LLM provider:

1. Create a new client class inheriting from `BaseLLMClient`
2. Implement the required `generate` method
3. Update the `_initialize_client` method in `LLMPortal`
4. Add the provider to your `model_directory.yaml`

Example:

```python
from llm_portal import BaseLLMClient, LLMResponse

class NewProviderClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str):
        # Initialize client
        pass

    def generate(self, prompt: str, system_context: str, 
                temperature: float, max_tokens: int) -> LLMResponse:
        # Implement generation logic
        pass
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details.

## Known Limitations

- Token counting may vary between providers
- Not all providers support all features (e.g., system messages)
- Local models via Ollama may have different performance characteristics

## Troubleshooting

Common issues and solutions:

1. **API Key Not Found**
   - Ensure environment variables are properly set
   - Check variable naming matches config file

2. **Model Not Found**
   - Verify model name in `model_directory.yaml`
   - Check provider endpoint is correct

3. **Response Truncation**
   - Increase `max_tokens` parameter
   - Check provider-specific token limits

