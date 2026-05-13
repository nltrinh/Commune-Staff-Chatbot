# 🧠 VIBECODE - Ngữ cảnh Dự án & Quy tắc Bất biến

Tài liệu này dành cho các AI Agent tiếp nhận dự án. Đọc kỹ để tránh làm sai lệch kiến trúc và "mất trí nhớ" về các quyết định quan trọng.

## 🌟 Linh hồn của Dự án
Dự án không chỉ là một Chatbot, mà là một **Hệ thống Quản trị Tri thức Hành chính**. Mục tiêu cuối cùng là triển khai trên **Vast.ai (GPU)** với mô hình nội bộ, không phụ thuộc API bên ngoài.

## 🛠 Tech Stack Cố định
- **LLM Core**: `qwen2.5:14b` (Tuyệt đối không dùng model khác cho tiếng Việt).
- **Embedding**: `bge-m3` (Chiều vector 1024).
- **Orchestration**: LangChain 0.3 + LCEL.
- **Frontend**: Vanilla HTML/JS (Aesthetics cao, không dùng React/Next để tối ưu tốc độ load trên server yếu).

## 📐 Quy tắc Kiến trúc (Bất biến)
1. **Provider Pattern**: Mọi logic AI/Vector phải thông qua Interface. Không import trực tiếp class cụ thể của Ollama hay MongoDB vào Service.
2. **Scoping**: Tài liệu luôn có tag phòng ban (`departments`). Phải luôn kiểm tra quyền truy cập của User trước khi Retrieval.
3. **Dual-Mode**: Hệ thống phải chạy được ở cả `USE_MOCK_AI=True` (Local/CPU) và `USE_MOCK_AI=False` (Production/GPU).
4. **Data Integrity**: Khi xóa tài liệu, phải dọn dẹp cả 3 nơi: MongoDB Metadata, MongoDB Vector, và File vật lý trên đĩa.

## 📂 Cấu trúc Dữ liệu Quan trọng
- `tat_ca`: Mã phòng ban đặc biệt cho tài liệu dùng chung toàn cơ quan.
- `host_admin`: Tài khoản quản trị cao nhất, cấu hình qua `.env`.

## ⚠️ Lưu ý cho Agent
- **Đừng Rewrite**: Nếu cần sửa, hãy dùng `replace_file_content` hoặc `multi_replace`. Đừng viết lại toàn bộ file nếu không yêu cầu.
- **Đừng Over-engineering**: Không thêm các framework phức tạp như LangGraph nếu không thực sự cần thiết cho tính năng hiện tại.
- **UI First**: Giao diện phải đẹp và premium. Dùng CSS hiện đại (Glassmorphism, Gradients).

## 🚀 Trạng thái Hiện tại (Checkpoint)
- DMS (Quản lý tài liệu) đã hoàn thiện 100%.
- Phân quyền theo phòng ban đã chạy đúng.
- Sẵn sàng để Deploy lên Vast.ai.
