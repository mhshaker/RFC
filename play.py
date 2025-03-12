import re

def extract_numbers(text):
    return re.findall(r'\d+', text)

# Example usage:
text = "synthetic_nn"
numbers = extract_numbers(text)
print(numbers)  # Output: ['123', '456', '78']
