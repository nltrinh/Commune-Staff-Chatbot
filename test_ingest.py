import requests
import json

base_url = "http://localhost:8000"

print("1. Logging in...")
login_response = requests.post(
    f"{base_url}/login",
    data={"username": "host_admin", "password": "admin123"}
)
token_data = login_response.json()
token = token_data["access_token"]
headers = {"Authorization": f"Bearer {token}"}

print("\n2. Ingesting test_document.txt...")
files = {
    "file": ("test_document.txt", open("test_document.txt", "rb"), "text/plain")
}
data = {
    "department": "tu_phap"
}

upload_response = requests.post(
    f"{base_url}/admin/upload",
    headers=headers,
    files=files,
    data=data
)
print("Upload Response:", upload_response.json())
file_id = upload_response.json().get("file_id")

# Wait for background task to ingest document and generate vector embeddings
import time
print("Waiting 10 seconds for document processing...")
time.sleep(10)

print("\n3. Verifying ingestion status...")
files_response = requests.get(
    f"{base_url}/admin/files",
    headers=headers
)
print("Files Status:")
for file_info in files_response.json().get("files", []):
    if file_info.get("file_id") == file_id:
        print(f"File: {file_info['file_name']}, Status: {file_info['status']}, Chunks: {file_info.get('chunks_count')}")

print("\n4. Sending chat query related to the document...")
chat_payload = {
    "message": "Quy trình hành chính xã xử lý trong vòng mấy ngày làm việc?",
    "session_id": "test_session_rag"
}
headers["Content-Type"] = "application/json"
chat_response = requests.post(
    f"{base_url}/chat",
    headers=headers,
    json=chat_payload
)

print("Chat Response Status Code:", chat_response.status_code)
if chat_response.status_code == 200:
    res_data = chat_response.json()
    print("\nAssistant Answer:\n", res_data.get("answer"))
    print("\nSources:\n", json.dumps(res_data.get("sources"), indent=2, ensure_ascii=False))
else:
    print("Chat failed:", chat_response.text)
