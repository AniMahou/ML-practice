import tiktoken
from typing import List, Dict

class TokenManager:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
    
    def count(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a chat messages array."""
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += 4  # Overhead per message (role tokens, formatting)
        total += 2  # Overall overhead
        return total
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost."""
        p = self.pricing.get(self.model, self.pricing["gpt-4o-mini"])
        input_cost = (input_tokens / 1_000_000) * p["input"]
        output_cost = (output_tokens / 1_000_000) * p["output"]
        return input_cost + output_cost
    
    def truncate_to_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

# Usage
tm = TokenManager("gpt-4o-mini")
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Tell me about AI."}
]
tokens = tm.count_messages(messages)
cost = tm.calculate_cost(tokens, 500) 
print(f"Input tokens: {tokens}, Estimated cost: ${cost:.6f}")