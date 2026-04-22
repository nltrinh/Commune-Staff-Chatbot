# 🏛️ Trợ lý Số Cán bộ Xã (Commune Staff Chatbot)

<div align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/4712/4712038.png" alt="Logo" width="120">
  <h1 align="center">🏛️ Trợ lý Số Cán bộ Xã RAG</h1>
  <p align="center">
    <strong>Hệ thống Trí tuệ Nhân tạo thông minh, hỗ trợ tra cứu nghiệp vụ và cẩm nang chính quyền địa phương cấp cơ sở.</strong>
    <br />
    <br />
    <a href="#-tính-năng-cốt-lõi">Tính năng</a>
    ·
    <a href="#-kiến-trúc-hệ-thống">Kiến trúc</a>
    ·
    <a href="#-triển-khai-nhanh-quick-start">Triển Khai Mẫu</a>
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
  [![MongoDB](https://img.shields.io/badge/MongoDB-8.0-47A248.svg?logo=mongodb&logoColor=white)](https://mongodb.com)
  [![Ollama](https://img.shields.io/badge/AI_Engine-Ollama-white.svg?logo=ollama&logoColor=black)](https://ollama.com)
  [![LangChain](https://img.shields.io/badge/Framework-LangChain_0.3-orange.svg)](https://langchain.com)
</div>

---

## 🌟 Tính Năng Cốt Lõi

- **100% Khép Kín (Local-First):** Hệ thống được xây dựng để hoạt động hoàn toàn trên máy chủ nội bộ. Đảm bảo bảo mật dữ liệu hành chính nhà nước (`Privacy-first`) và không phát sinh chi phí vận hành API ngoài.
- **RAG Nâng Cao (Hybrid Search):** Tìm kiếm thông tin chính xác từ các văn bản pháp luật, cẩm nang nghiệp vụ bằng cơ chế lai ghép **Reciprocal Rank Fusion (RRF)**. Kết hợp Vector Search (Semantic) và Keyword Search (BM25).
- **LLM Chuyên Trị Tiếng Việt:** Sử dụng mô hình **Qwen-2.5 14B** tối ưu nhất cho văn bản hành chính Tiếng Việt, giúp phản hồi tự nhiên, chuẩn xác.
- **Quản lý Tài liệu Thông minh:** Giao diện quản trị cho phép upload và index các tệp PDF, DOCX, TXT về quy định pháp luật nhanh chóng.

## 🏗 Kiến Trúc Hệ Thống

Dự án sử dụng bộ Stack công nghệ hiện đại nhất cho bài toán RAG:
- **LLM Engine:** Ollama chạy cục bộ.
- **Embedding:** `BGE-M3` (Đa ngôn ngữ, hỗ trợ Tiếng Việt mạnh nhất).
- **Database:** MongoDB 8.0 với tính năng Vector Search Native.
- **Orchestration:** LangChain v0.3.
- **Backend:** FastAPI với cơ chế Streaming Response (SSE).

## 🚀 Triển Khai Nhanh (Quick Start)

### 1. Cấu hình môi trường
Sao chép file `.env` và điền các thông số kết nối:
```bash
cp env.example .env
```

### 2. Cài đặt Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Khởi chạy Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📂 Cấu trúc dự án

```text
├── app/
│   ├── api/          # Các endpoint API (Admin, Chat)
│   ├── core/         # Cấu hình hệ thống & DB
│   ├── rag/          # Logic xử lý RAG, Vector Search
│   └── main.py       # Entry point FastAPI
├── static/           # Giao diện Web (HTML, CSS, JS)
├── sample_data/      # Tài liệu mẫu về hành chính
└── AGENT_AUTO_DEPLOY.md # Hướng dẫn cho AI Agent
```

---
© 2026 - Dự án hỗ trợ Chuyển đổi số cấp cơ sở.
