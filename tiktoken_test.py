import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    tokens = encoding.encode(text)
    return len(tokens)

test_strings = [
    "Hello world",
    "Hello world" * 10, 
    "Supercalifragilisticexpialidocious",
    "🍕 I love AI!",
    "The quick brown fox jumps over the lazy dog."
]

for text in test_strings:
    token_count = count_tokens(text)
    char_count = len(text)
    print(f"Text: {text[:50]}...")
    print(f"Characters: {char_count} | Tokens: {token_count}")
    print(f"Ratio (chars/token): {char_count/token_count:.2f}")
    print("-" * 50)