from openai import OpenAI
import json
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class MessageStructureLab:
    """Experiment with different message structures."""
    
    def __init__(self):
        self.conversations = {}
    
    def create_conversation(self, name: str, system_prompt: str):
        """Initialize a new conversation thread."""
        self.conversations[name] = {
            "system": system_prompt,
            "history": []
        }
        print(f"✅ Created conversation: '{name}'")
    
    def chat(self, conv_name: str, user_input: str) -> str:
        """Send a message in a specific conversation."""
        if conv_name not in self.conversations:
            raise ValueError(f"Conversation '{conv_name}' not found")
        
        conv = self.conversations[conv_name]
        
        # Build messages
        messages = [{"role": "system", "content": conv["system"]}]
        messages.extend(conv["history"])
        messages.append({"role": "user", "content": user_input})
        
        # Debug: Show what's being sent
        print(f"\n📤 SENDING TO API ({conv_name}):")
        print(f"   Messages count: {len(messages)}")
        print(f"   Last user input: {user_input[:50]}...")
        
        # Call API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        
        assistant_response = response.choices[0].message.content
        
        # Update history
        conv["history"].append({"role": "user", "content": user_input})
        conv["history"].append({"role": "assistant", "content": assistant_response})
        
        print(f"\n📥 RESPONSE:")
        print(f"   {assistant_response}")
        
        return assistant_response
    
    def compare_system_prompts(self, user_input: str, system_prompts: dict):
        """Compare how different system prompts affect the same user input."""
        print("\n" + "="*60)
        print(f"COMPARING SYSTEM PROMPTS")
        print(f"User Input: '{user_input}'")
        print("="*60)
        
        results = {}
        for name, sys_prompt in system_prompts.items():
            print(f"\n--- {name} ---")
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input}
            ]
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.0
            )
            result = response.choices[0].message.content
            results[name] = result
            print(f"Response: {result}")
        
        return results
    
    def show_history(self, conv_name: str):
        """Display the conversation history."""
        conv = self.conversations[conv_name]
        print(f"\n📜 HISTORY: {conv_name}")
        print(f"System: {conv['system'][:100]}...")
        for i, msg in enumerate(conv["history"]):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            print(f"{role_emoji} {msg['role']}: {msg['content'][:80]}...")

# Run the lab
lab = MessageStructureLab()

# Experiment 1: Different system prompts, same input
system_prompts = {
    "Pirate": "You are a pirate. Speak like one. Be enthusiastic about treasure.",
    "Professor": "You are a formal academic professor. Use scholarly language.",
    "Teenager": "You are a teenager. Use slang, be casual, use 'like' and 'literally'.",
    "Robot": "You are a robot. Be literal, precise, and unemotional."
}

lab.compare_system_prompts("Tell me about the ocean.", system_prompts)

# Experiment 2: Multi-turn conversation with memory
lab.create_conversation("memory_test", "You are a helpful assistant. Remember user's name.")
lab.chat("memory_test", "My name is Alex.")
lab.chat("memory_test", "What's my favorite color? I like blue.")
lab.chat("memory_test", "What is my name and favorite color?")
lab.show_history("memory_test")

template = "Explain {concept} to a {audience} in {word_count} words or less."

variables = [
    {"concept": "quantum entanglement", "audience": "5-year-old", "word_count": "50"},
    {"concept": "quantum entanglement", "audience": "physics PhD", "word_count": "50"},
]

print("\n" + "="*60)
print("VARIABLE INJECTION TEST")
print("="*60)

for varset in variables:
    prompt = template.format(**varset)
    print(f"\nPrompt: {prompt}")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    print(f"Response: {response.choices[0].message.content}")