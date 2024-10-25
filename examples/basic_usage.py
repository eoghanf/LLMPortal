from src import LLMPortal
from dotenv import load_dotenv


def demonstrate_basic_usage():
    # Initialize the portal with different models
    print("Demonstrating LLM Portal with different providers...")

    # Example 1: Using OpenAI
    print("\n1. Using OpenAI GPT-4o-mini")
    openai_portal = LLMPortal("gpt-4o-mini")
    openai_portal.set_context("You are a Python programming expert.")
    response = openai_portal(
        prompt="Explain list comprehension and provide an example.",
        temp=0.2
    )
    print(f"OpenAI Response:\n{response}\n")

    # Example 2: Using Anthropic
    print("\n2. Using Anthropic Claude")
    claude_portal = LLMPortal("claude-3-opus-20240229")
    claude_portal.set_context("You are a creative writing assistant.")
    response = claude_portal(
        prompt="Write a short poem about coding.",
        temp=0.7  # Higher temperature for more creative responses
    )
    print(f"Anthropic Response:\n{response}\n")

    # Example 3: Using local model via Ollama
    print("\n3. Using Local Model (Ollama)")
    local_portal = LLMPortal("qwen2.5:latest")
    local_portal.set_context("You are a helpful math tutor.")
    response = local_portal(
        prompt="Solve this problem step by step: If x + 2 = 5, what is x?",
        temp=0.1  # Lower temperature for more deterministic responses
    )
    print(f"Local Model Response:\n{response}\n")

    # Example 4: Error handling
    print("\n4. Demonstrating Error Handling")
    try:
        invalid_portal = LLMPortal("nonexistent-model")
    except Exception as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    try:
        load_dotenv()
    except:
        print("Did not find a dotenv file.... skipping......")
    demonstrate_basic_usage()