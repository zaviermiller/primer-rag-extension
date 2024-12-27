import os
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens_in_files(directory: str, encoding_name: str = "o200k_base"):
    """Counts and prints the number of tokens in each file in the given directory and its subdirectories."""
    encoding = tiktoken.get_encoding(encoding_name)
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                # skip images
                if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                    continue
                try:
                    content = f.read()
                    num_tokens = len(encoding.encode(content))
                    if num_tokens > 8192:
                        print(f"{file_path} has too many tokens: {num_tokens} tokens")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Example usage:
# count_tokens_in_files('data')

count_tokens_in_files('server/data')