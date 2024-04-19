import json

def load_config(path: str) -> dict:
  with open(path, 'r') as file:
    content = file.read()
    return json.loads(content)