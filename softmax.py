import numpy as np
import matplotlib.pyplot as plt

def softmax(logits, temperature=1.0):
    """Apply softmax with temperature scaling."""
    scaled = np.array(logits) / temperature
    exp_logits = np.exp(scaled - np.max(scaled))  # Subtract max for numerical stability
    return exp_logits / exp_logits.sum()

# Example vocabulary and logits
tokens = ["blue", "clear", "cloudy", "falling", "purple", "stormy"]
logits = [5.0, 3.0, 2.5, 1.0, 0.5, 0.2]

# Compare temperatures
temperatures = [0.2, 0.5, 1.0, 1.5]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i, T in enumerate(temperatures):
    probs = softmax(logits, T)
    axes[i].bar(tokens, probs)
    axes[i].set_title(f"T = {T}")
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()