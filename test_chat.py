import requests
import json

base_url = "http://localhost:8000"

print("1. Logging in...")
login_response = requests.post(
    f"{base_url}/login",
    data={"username": "host_admin", "password": "admin123"}
)
print("Login Status Code:", login_response.status_code)
if login_response.status_code != 200:
    print("Login failed:", login_response.text)
    exit(1)

token_data = login_response.json()
token = token_data["access_token"]
print("Login successful. Token acquired.")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("\n2. Sending chat message to LLM (qwen2.5:14b)...")
chat_payload = {
    "message": "Xin chào! Bạn là ai và có thể giúp tôi việc gì?",
    "session_id": "test_session_123"
}

chat_response = requests.post(
    f"{base_url}/chat",
    headers=headers,
    json=chat_payload
)

print("Chat Response Status Code:", chat_response.status_code)
if chat_response.status_code == 200:
    res_data = chat_response.json()
    print("\nAssistant Answer:\n", res_data.get("answer"))
else:
    print("Chat failed:", chat_response.text)
