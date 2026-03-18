import requests
import json

def test_proxy():
    url = "http://localhost:8000/api/v1/proxy/generate"
    payload = {
        "prompt": "Say hello in JSON format",
        "system_instruction": "You are a helpful assistant. You MUST return JSON."
    }
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_proxy()
