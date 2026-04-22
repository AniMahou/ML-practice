from openai import OpenAI
import json
from typing import Optional
from dotenv import load_dotenv
import os
from collections import Counter
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def optimized_self_consistency(question: str, max_samples: int = 7, 
                                confidence_threshold: float = 0.8) -> dict:
    """Early stopping when confidence threshold is reached."""
    
    prompt = f"""Q: {question}
A: Let's think step by step."""
    
    answers = []
    
    for i in range(max_samples):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        full_response = response.choices[0].message.content
        
        # Extract answer (simplified)
        import re
        numbers = re.findall(r'\d+', full_response)
        answer = numbers[-1] if numbers else "no_number"
        answers.append(answer)
        
        # Check if we have confidence
        if i >= 2:  # Need at least 3 samples
            counts = Counter(answers)
            most_common_count = counts.most_common(1)[0][1]
            confidence = most_common_count / len(answers)
            
            if confidence >= confidence_threshold:
                print(f"✅ Early stop at {i+1} samples with {confidence:.0%} confidence")
                break
    
    final_counts = Counter(answers)
    return {
        "answer": final_counts.most_common(1)[0][0],
        "samples_used": len(answers),
        "confidence": final_counts.most_common(1)[0][1] / len(answers)
    }

# Test
result = optimized_self_consistency("What is the integration of e^(lnx) with respect to x", max_samples=7)
print(f"Answer: {result['answer']}, Samples: {result['samples_used']}, Confidence: {result['confidence']:.0%}")