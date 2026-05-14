#!/bin/bash

# --- Commune Staff Chatbot - Auto Setup Script ---
# Dành cho máy chủ GPU (Vast.ai, RunPod, Bare Metal)

echo "🚀 Bắt đầu thiết lập hệ thống Trợ lý Hành chính Xã..."

# 1. Cập nhật hệ thống & Cài đặt dependencies
echo "📦 Đang cài đặt thư viện hệ thống..."
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip curl libmagic1

# 2. Thiết lập môi trường Python
echo "🐍 Đang thiết lập môi trường ảo..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Cấu hình MongoDB 8.0+
echo "🍃 Đang khởi động MongoDB với Replica Set..."
mkdir -p ~/data/db
sudo pkill -9 mongod || true
sleep 2
nohup mongod --port 27017 --dbpath ~/data/db --replSet rs0 --fork --logpath ~/data/mongod.log > /dev/null 2>&1 &

# Chờ MongoDB khởi động và khởi tạo Replica Set (bắt buộc cho Vector Search)
sleep 5
mongosh --eval 'rs.initiate()' || echo "Replica Set đã được khởi tạo."

# --- BƯỚC QUAN TRỌNG: Khởi tạo Vector Search Index ---
echo "🔍 Đang tạo Vector Search Index cho MongoDB..."
sleep 2
mongosh commune_staff_bot --eval '
db.documents.createSearchIndex({
  name: "vector_index",
  type: "vector",
  definition: {
    "fields": [
      { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" },
      { "type": "filter", "path": "metadata.departments" }
    ]
  }
})'

# 4. Cấu hình Ollama & Tải Models
echo "🤖 Đang thiết lập Ollama AI..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

nohup ollama serve > ollama.log 2>&1 &
sleep 5

echo "📥 Đang tải mô hình Qwen2.5:14b..."
ollama pull qwen2.5:14b
echo "📥 Đang tải mô hình Embedding BGE-M3..."
ollama pull bge-m3

# 5. Cấu hình file .env
echo "📝 Đang thiết lập file cấu hình .env..."
cat <<EOF > .env
APP_TITLE=Trợ lý Hành chính Xã
USE_MOCK_AI=False

SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
ADMIN_USERNAME=host_admin
ADMIN_PASSWORD=admin123

MONGO_URI=mongodb://localhost:27017/?replicaSet=rs0
MONGO_DB_NAME=commune_staff_bot

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_EMBED_MODEL=bge-m3

VECTOR_INDEX_NAME=vector_index
MONGODB_VECTOR_MODE=native

TOP_K_RESULTS=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EOF

# 6. Khởi động Server
echo "🔥 Đang khởi động Backend Server..."
pkill -9 -f uvicorn || true
sleep 1
nohup venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 > server.log 2>&1 &

echo "----------------------------------------------------"
echo "✅ HOÀN TẤT THIẾT LẬP!"
echo "📍 Dashboard: http://localhost:8000/dashboard"
echo "🔑 Admin: host_admin / admin123"
echo "📜 Xem log: tail -f server.log"
echo "----------------------------------------------------"
