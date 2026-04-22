class FewShotCoTLab:
    """Compare zero-shot, few-shot, and CoT on the same tasks."""
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    load_dotenv()

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.results = {}
    
    def zero_shot(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Q: {question}\nA:"}],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def few_shot(self, question: str) -> str:
        prompt = """
        Q: If a train travels 60 miles in 2 hours, what is its speed?
        A: 30 miles per hour
        
        Q: A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?
        A: 2 slices
        
        Q: {question}
        A:"""
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt.format(question=question)}],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def chain_of_thought(self, question: str) -> str:
        prompt = """
        Q: If a train travels 60 miles in 2 hours, what is its speed?
        A: Let me think step by step.
        Distance = 60 miles
        Time = 2 hours
        Speed = Distance ÷ Time = 60 ÷ 2 = 30
        Final answer: 30 miles per hour
        
        Q: A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?
        A: Step by step:
        Total slices = 8
        Each person eats 2 slices, 3 people = 3 × 2 = 6 slices eaten
        Slices left = 8 - 6 = 2
        Final answer: 2 slices
        
        Q: {question}
        A: Let me think step by step."""
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt.format(question=question)}],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def compare_methods(self, question: str, correct_answer: str):
        """Compare all three methods."""
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"CORRECT ANSWER: {correct_answer}")
        print('='*60)
        
        methods = {
            "Zero-Shot": self.zero_shot,
            "Few-Shot": self.few_shot,
            "Chain-of-Thought": self.chain_of_thought
        }
        
        for name, method in methods.items():
            result = method(question)
            is_correct = correct_answer.lower() in result.lower()
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"\n{name}:")
            print(f"  Result: {result}")
            print(f"  Status: {status}")

# Test with challenging problems
lab = FewShotCoTLab()

# Problem 1: Multi-step math
lab.compare_methods(
    question="John has 3 boxes. Each box contains 4 bags. Each bag has 5 marbles. How many marbles does John have?",
    correct_answer="60"
)

# Problem 2: Trick question
lab.compare_methods(
    question="If you have 10 apples and you give away half, then eat 2, then buy 5 more, how many do you have?",
    correct_answer="8"
)

# Problem 3: Logical reasoning
lab.compare_methods(
    question="All dogs are animals. All animals need water. Does a dog need water?",
    correct_answer="yes"
)