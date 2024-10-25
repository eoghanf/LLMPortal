from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    token_count: int

