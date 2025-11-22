import ollama

response = ollama.chat(model='llama3', messages=[{"role": "user", "content": "Summarize this JD..."}])
print(response['message']['content']) 