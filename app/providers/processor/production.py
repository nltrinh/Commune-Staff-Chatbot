import os
import io
import tempfile
import logger_config
from typing import List, Dict, Any
from app.core.interfaces import DocumentProcessor

logger = logger_config.get_logger(__name__)

class ProductionProcessor(DocumentProcessor):
    """
    Bộ xử lý tài liệu tối ưu cho môi trường Production (Vast.ai / Linux).
    Sử dụng PyPDFLoader và Docx2txtLoader để trích xuất văn bản ổn định và nhẹ nhàng.
    """
    def extract_text(self, content: bytes, file_type: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            # Lưu tạm file để Loader của LangChain có thể đọc được
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                if file_type == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for i, d in enumerate(docs, 1):
                        if d.page_content.strip():
                            pages.append({"text": d.page_content, "page_num": i})
                            
                elif file_type in [".docx", ".doc"]:
                    # Lưu ý: Docx2txtLoader xử lý tốt cả .docx và một số tệp .doc (thực chất là docx)
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(tmp_path)
                    docs = loader.load()
                    full_text = "\n\n".join(d.page_content for d in docs)
                    if full_text.strip():
                        pages.append({"text": full_text, "page_num": 1})
                
                else:
                    # Fallback cho text thô
                    text = content.decode("utf-8", errors="ignore")
                    if text.strip():
                        pages.append({"text": text, "page_num": 1})
                        
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"ProductionProcessor Error ({file_type}): {e}")
            pages.append({"text": f"[LỖI PRODUCTION]: {str(e)}", "page_num": 1})
            
        return pages
