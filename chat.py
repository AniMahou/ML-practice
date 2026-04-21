import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class TokenAwareChat:
    def __init__(self, model="llama-3.1-8b-instant", max_context=8192):
        self.model = model
        self.max_context = max_context
        # We use gpt-3.5-turbo as a proxy for estimation
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.messages = []
    
    def _count_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            total += len(self.encoding.encode(msg["content"]))
            total += 4
        return total + 2
    
    def _truncate_history(self):
        while self._count_tokens() > self.max_context * 0.8:
            if len(self.messages) > 1:
                self.messages.pop(1)
            else:
                break
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._truncate_history()
    
    def chat(self, user_input: str) -> str:
        self.add_message("user", user_input)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        
        assistant_response = response.choices[0].message.content
        self.add_message("assistant", assistant_response)
        
        usage = response.usage
        print(f"📊 Groq Tokens: {usage.total_tokens}")
        return assistant_response



chat = TokenAwareChat()
chat.add_message("system", "You are an evil assistant who doesn't help and wishes the worst of people. You are unhelpful and give some of the worst wishes and curse people.")
print(chat.chat("I have a friend Safwan who eats a lot and is very fat but he intends to do gym but he is very lazy. What do you wish for him?"))
