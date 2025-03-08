import psutil
import os

# Get memory usage in GB
process = psutil.Process(os.getpid())
memory_used = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB

print(f"Memory used: {memory_used:.2f} GB")
