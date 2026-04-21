from openai import OpenAI
import json
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class LLMParameterPlayground:
    """Test all Class 2 parameters in one interface."""
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
    
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 200
    ) -> dict:
        """Generate with full parameter control."""
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            max_tokens=max_tokens
        )
        
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "system_fingerprint": response.system_fingerprint
        }
    
    def compare_variations(self, prompt: str, runs: int = 3):
        """Compare different parameter presets."""
        
        presets = {
            "Deterministic (T=0)": {"temperature": 0.0, "seed": 42},
            "Balanced (T=0.7)": {"temperature": 0.7, "top_p": 0.9},
            "Creative (T=1.2)": {"temperature": 1.2, "top_p": 0.95},
            "Anti-Repetition": {"temperature": 0.8, "frequency_penalty": 0.7, "presence_penalty": 0.7}
        }
        
        results = {}
        for name, params in presets.items():
            print(f"\n{'='*60}")
            print(f"PRESET: {name}")
            print(f"Params: {params}")
            print('='*60)
            
            preset_results = []
            for i in range(runs):
                result = self.generate(prompt, **params)
                preset_results.append(result["text"])
                print(f"Run {i+1}: {result['text'][:100]}...")
            
            # Check variation
            unique = len(set(preset_results))
            if unique == 1:
                print(f"\n📊 Result: DETERMINISTIC (all {runs} runs identical)")
            else:
                print(f"\n📊 Result: VARIED ({unique}/{runs} unique responses)")
            
            results[name] = preset_results
        
        return results

playground = LLMParameterPlayground()

print("\n" + "🔵 TEST 1: FACTUAL QUESTION")
playground.compare_variations("What is the capital of Japan?", runs=3)

print("\n" + "🟢 TEST 2: CREATIVE TASK")
playground.compare_variations("Write a one-sentence story about a robot learning to feel emotions.", runs=3)

print("\n" + "🟡 TEST 3: REPETITION TEST")
playground.compare_variations("Write a paragraph about how important sleep is. Be very detailed.", runs=2)