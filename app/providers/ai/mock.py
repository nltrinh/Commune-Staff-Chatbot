import time
import random
from typing import List, Dict, Generator
from app.core.interfaces import AIProvider

class MockAIProvider(AIProvider):
    def embed_query(self, text: str) -> List[float]:
        # Return a random vector consistent with production dimension
        from app.core.config import settings
        return [random.uniform(-1, 1) for _ in range(settings.EMBEDDING_DIM)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from app.core.config import settings
        return [[random.uniform(-1, 1) for _ in range(settings.EMBEDDING_DIM)] for _ in texts]

    def stream_chat(self, query: str, context: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        # Count sources by looking for "[n] Nguồn:" pattern in context
        import re
        sources_found = len(re.findall(r"\[\d+\] Nguồn:", context))
        
        response = f"Đây là phản hồi mô phỏng (Mock Mode).\n"
        response += f"- Bạn đã hỏi: '{query}'.\n"
        response += f"- Hệ thống tìm thấy {sources_found} đoạn tài liệu liên quan trong phạm vi quyền hạn của bạn.\n"
        response += f"- Tổng độ dài ngữ cảnh: {len(context)} ký tự.\n\n"
        
        if sources_found == 0:
            response += "⚠️ KHÔNG tìm thấy tài liệu nào trong phòng ban của bạn hoặc tài liệu dùng chung."
        else:
            response += "💡 Vì đang ở Mock Mode, tôi chưa phân tích nội dung mà chỉ hiển thị kết quả truy xuất thô."
            
        for word in response.split():
            yield word + " "
            time.sleep(0.02)
