# 🚀 Hướng Dẫn Tự Động Hóa Triển Khai (Vast.ai / GPU Server)

Tài liệu này giúp AI Agent tự động thiết lập toàn bộ hệ thống Trợ lý Hành chính Xã từ trạng thái máy trắng lên trạng thái sẵn sàng vận hành.

## 🗣 Lệnh Kích Hoạt cho AI Agent
*Copy dòng dưới đây dán vào cửa sổ Chat của Agent:*

> "Chào Agent, bạn hãy đóng vai là một kỹ sư Backend & AI Expert. Tôi vừa mới kết nối vào một máy chủ mới. Nhiệm vụ của bạn là đọc hiểu file AGENT_AUTO_DEPLOY.md này để nắm bắt kiến trúc hiện tại ĐÃ ĐƯỢC CỐ ĐỊNH (Qwen2.5 14B, MongoDB 8, BGE-M3) và thực thi tất cả các bước để đưa hệ thống lên trạng thái chạy ổn định (Ready to Demo). Hãy cài đặt dependencies, chạy hệ thống và test kết quả phản hồi,cấu hình ENV và khởi động server với 4 workers. Lưu ý: Luôn sử dụng mô hình Qwen2.5:14b bất kể cấu hình máy như thế nào."

---

## 🏗 Thông Số Kỹ Thuật (Production)
- **Model**: `qwen2.5:14b` & `bge-m3`.
- **Database**: MongoDB 8.2 (Replica Set `rs0` bắt buộc).
- **Storage**: `data/uploads/` cho tài liệu gốc.

---

## 🛠 Quy Trình Triển Khai Chi Tiết

### BƯỚC 1: Cấu hình MongoDB 8 Replica Set
```bash
# Tạo thư mục dữ liệu
mkdir -p ~/data/db

# Khởi chạy MongoDB với Replica Set
mongod --port 27017 --dbpath ~/data/db --replSet rs0 --fork --logpath ~/data/mongod.log

# Chờ 5s và khởi tạo
sleep 5
mongosh --eval 'rs.initiate()'
```

### BƯỚC 1.5: Khởi tạo Vector Search Index (Quan trọng)
*Lệnh này giúp MongoDB 8.2 hiểu cách tìm kiếm vector 1024 chiều:*
```bash
mongosh commune_staff_bot --eval '
db.documents.createSearchIndex({
  name: "vector_index",
  type: "vector",
  definition: {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1024,
        "similarity": "cosine"
      },
      {
         "type": "filter",
         "path": "metadata.departments"
      }
    ]
  }
})'
```

### BƯỚC 2: Cài đặt & Tải Model Ollama
```bash
# Cài đặt Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Tải Models (Sẽ mất thời gian tùy tốc độ mạng)
ollama pull bge-m3
ollama pull qwen2.5:14b
```

### BƯỚC 3: Thiết lập Môi trường Python
```bash
# Cài đặt venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### BƯỚC 4: Cấu hình Biến Môi trường (.env)
Tạo file `.env` với nội dung:
```env
# Server
APP_TITLE=Trợ lý Hành chính Xã
USE_MOCK_AI=False

# Auth
SECRET_KEY=commune_secret_key_change_me_in_production
ADMIN_USERNAME=host_admin
ADMIN_PASSWORD=admin123

# MongoDB (Local Replica Set)
MONGO_URI=mongodb://localhost:27017/?replicaSet=rs0
MONGO_DB_NAME=commune_staff_bot

# AI & Vector Store
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=bge-m3
EMBEDDING_DIM=1024
LLM_MODEL=qwen2.5:14b

# MongoDB Vector Search Index name
VECTOR_INDEX_NAME=vector_index
MONGODB_VECTOR_MODE=native

# RAG Params
TOP_K_RESULTS=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### BƯỚC 5: Khởi chạy Hệ thống
```bash
# Khởi động server với 4 workers để tối ưu hiệu năng
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 > server.log 2>&1 &
```

### BƯỚC 6: Kiểm tra (Health Check) & Nạp dữ liệu thực tế
1. **Kiểm tra trạng thái hệ thống**:
```bash
curl -s http://localhost:8000/health
```

2. **Nạp dữ liệu**:
- **Cách 1 (Khuyên dùng)**: Truy cập giao diện Dashboard tại `http://<IP_VAST_AI>:8000/dashboard` -> Vào mục **Quản lý Tài liệu** -> Chọn tài liệu thực tế từ máy bạn để nạp.
- **Cách 2 (CLI)**: Sử dụng lệnh curl với tệp tin của bạn:
```bash
curl -X POST -F "file=@/đường/dẫn/tài/liệu/của/bạn.pdf" \
     "http://localhost:8000/admin/upload?department=tat_ca" \
     -H "Authorization: Bearer <TOKEN_YÊU_CẦU_ĐĂNG_NHẬP>"
```

---
⚠️ **Lưu ý cho Agent:** Luôn giữ `USE_MOCK_AI=False` khi ở trên GPU Server. Tuyệt đối không thay đổi model LLM đã chỉ định.
