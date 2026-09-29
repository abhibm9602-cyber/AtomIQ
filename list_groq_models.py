import os
from dotenv import load_dotenv
import requests

load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("No GROQ_API_KEY found.")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("Available Groq Models:")
        for model in data.get('data', []):
            print(f"- {model['id']}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
