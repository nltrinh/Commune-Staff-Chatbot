import json
import logger_config
from typing import List, Dict, Generator
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.core.interfaces import AIProvider

logger = logger_config.get_logger(__name__)

PROMPT_TEMPLATE = """Bạn là Trợ lý AI Hành chính chuyên trách hỗ trợ Cán bộ Xã. Nhiệm vụ của bạn là giải đáp thông tin nghiệp vụ và hướng dẫn thủ tục hành chính dựa TRÊN TÀI LIỆU THAM KHẢO được cung cấp bên dưới.

🚨 QUY TẮC NGHIÊM NGẶT:
1. CHỈ sử dụng thông tin có trong "TÀI LIỆU THAM KHẢO". Nếu thông tin không có, hãy lịch sự trả lời rằng tài liệu hiện tại không đề cập đến vấn đề này.
2. KHÔNG tự bịa đặt các quy định pháp luật, số liệu hoặc quy trình không có trong tài liệu.
3. LUÔN trích dẫn nguồn ([số thứ tự nguồn]) ngay sau thông tin bạn trích xuất.
4. Trình bày thông tin rõ ràng, chuyên nghiệp bằng Markdown.
5. Giữ thái độ lịch sự, chuẩn mực và hỗ trợ đồng nghiệp.

TÀI LIỆU THAM KHẢO:
{context}

LỊCH SỬ HỘI THOẠI:
{history}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

Hãy đưa ra câu trả lời chuyên môn:"""

class OllamaProvider(AIProvider):
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        self.llm = OllamaLLM(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            num_predict=settings.OLLAMA_NUM_PREDICT,
            temperature=0.0,
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def stream_chat(self, query: str, context: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        history_str = ""
        if history:
            recent = history[-4:]
            history_str = "\n".join(
                f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}" for m in recent
            )
        
        inputs = {
            "context": context,
            "history": history_str or "Chưa có lịch sử hội thoại.",
            "question": query,
        }
        
        for chunk in self.chain.stream(inputs):
            yield chunk
