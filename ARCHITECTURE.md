# 🏛️ Kiến trúc Hệ thống - Trợ lý Hành chính Xã (Commune Staff Chatbot)

Hệ thống được thiết kế theo kiến trúc **SOA (Service-Oriented Architecture)** kết hợp với **Provider Pattern**, giúp dễ dàng hoán đổi giữa môi trường Mock (Phát triển) và Vast.ai (Production).

## 1. Các Lớp Thành Phần (Layers)

### 1.1 Core Layer (`app/core/`)
- **`config.py`**: Quản lý toàn bộ cấu hình hệ thống qua biến môi trường (.env).
- **`database.py`**: Singleton kết nối MongoDB và các helper lấy collection.
- **`factory.py`**: Trái tim của hệ thống. Quyết định khởi tạo Provider thật (Ollama, Unstructured) hay Provider giả (Mock) dựa trên biến `USE_MOCK_AI`.
- **`interfaces.py`**: Định nghĩa các bản thiết kế (Abstract Classes) cho AI, VectorStore và DocumentProcessor.

### 1.2 Provider Layer (`app/providers/`)
- **AI Provider**: 
  - `ollama.py`: Kết nối Qwen2.5 qua LangChain.
  - `mock.py`: Trả về phản hồi giả lập để test giao diện.
- **VectorStore Provider**:
  - `mongodb.py`: Sử dụng native Vector Search của MongoDB 8.2+.
- **Processor Provider**:
  - `production.py`: Dùng thư viện Unstructured để parse PDF/Docx chuyên sâu.

### 1.3 Service Layer (`app/services/`)
- **`rag_service.py`**: Điều phối luồng RAG (Tìm kiếm -> Prompt -> AI).
- **`document_service.py`**: Quản lý Metadata, Trạng thái tài liệu và logic lưu trữ.
- **`auth_service.py`**: Xử lý JWT, Password Hashing.
- **`admin_service.py`**: Quản lý Người dùng, Phòng ban và Thống kê.

## 2. Luồng Xử lý Dữ liệu (Pipeline)

### 2.1 Luồng Nạp Tài liệu (Ingestion)
1. **Upload**: API nhận file và lưu vào `data/uploads/`.
2. **Parsing**: `DocumentProcessor` trích xuất text thô.
3. **Chunking**: Chia nhỏ văn bản theo kích thước cấu hình (512-1024 tokens).
4. **Embedding**: Chuyển text thành vector (1024D).
5. **Storage**: Lưu metadata vào `uploaded_files` và vector vào `documents`.

### 2.2 Luồng Truy vấn (RAG Chat)
1. **Filter**: Xác định phòng ban của User để giới hạn phạm vi tìm kiếm.
2. **Search**: Tìm Top-K đoạn văn bản liên quan nhất.
3. **Augment**: Ghép ngữ cảnh vào Prompt Template tiếng Việt.
4. **Inference**: Gọi LLM (Qwen2.5) để sinh câu trả lời.
5. **Streaming**: Trả kết quả về Frontend theo thời gian thực (SSE).

## 3. Công nghệ Sử dụng (Tech Stack)
- **Backend**: FastAPI (Python 3.10+).
- **AI**: Ollama (Qwen2.5:14b, BGE-M3).
- **Database**: MongoDB 8.2 (Replica Set).
- **Frontend**: Vanilla JS, HTML5, CSS3 (Modern UI).