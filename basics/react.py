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

def advanced_react_agent(question: str) -> str:
    """ReAct agent with calculator and mock search."""
    
    def calculator(expr: str) -> str:
        try:
            return str(eval(expr))
        except:
            return "Error"
    
    def mock_search(query: str) -> str:
        """Simulated search tool."""
        knowledge = {
            "capital of france": "Paris",
            "population of paris": "2.16 million",
            "eiffel tower height": "330 meters",
        }
        return knowledge.get(query.lower(), "No information found")
    
    tools_desc = """
    - calculator[expression]: Calculate math expressions.
    - search[query]: Search for factual information.
    """
    
    prompt = f"""
    You are a helpful agent. Use tools to answer questions.
    
    Available Tools:
    {tools_desc}
    
    Format:
    Thought: [Your reasoning]
    Action: tool_name[input]
    Observation: [Tool result]
    ... (repeat)
    Final Answer: [Your answer]
    
    Question: {question}
    """
    
    history = ""
    for _ in range(5):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt + history}],
            temperature=0.0
        )
        
        output = response.choices[0].message.content
        history += output + "\n"
        
        if "Final Answer:" in output:
            return output.split("Final Answer:")[1].strip()
        
        if "Action:" in output:
            action = output.split("Action:")[1].split("\n")[0].strip()
            
            if action.startswith("calculator["):
                expr = action[11:-1]
                result = calculator(expr)
            elif action.startswith("search["):
                query = action[7:-1]
                result = mock_search(query)
            else:
                result = "Unknown tool"
            
            history += f"Observation: {result}\n"
    
    return "Unable to answer."

# Test
print(advanced_react_agent("What is the capital of France and what is 15% of 80?"))