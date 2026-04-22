from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def test_penalties(prompt: str, freq_pen: float, pres_pen: float):
    """Test how penalties affect output."""
    print(f"\n{'='*60}")
    print(f"Frequency Penalty: {freq_pen}, Presence Penalty: {pres_pen}")
    print(f"{'='*60}")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        max_tokens=150
    )
    
    text = response.choices[0].message.content
    
    # Simple repetition analysis
    words = text.lower().split()
    unique_words = set(words)
    repetition_ratio = len(unique_words) / len(words) if words else 0
    
    print(f"Output: {text}")
    print(f"\n📊 Repetition Ratio: {repetition_ratio:.2%} unique words")
    print(f"   ({len(unique_words)} unique / {len(words)} total)")
    
    return text

# Test scenarios
prompt = "Write a paragraph about the importance of sleep. Be detailed."

# No penalties (baseline)
test_penalties(prompt, 0.0, 0.0)

# Aggressive anti-repetition
test_penalties(prompt, 0.8, 0.8)

# Encourage repetition (for demonstration)
test_penalties(prompt, -0.5, -0.5)