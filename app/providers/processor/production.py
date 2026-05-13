import os
import io
import tempfile
import logger_config
from typing import List, Dict, Any
from app.core.interfaces import DocumentProcessor

logger = logger_config.get_logger(__name__)

class ProductionProcessor(DocumentProcessor):
    """
    Bộ xử lý tài liệu mạnh mẽ sử dụng thư viện unstructured.
    Yêu cầu môi trường có cài đặt soffice (LibreOffice), pandoc, tesseract.
    Phù hợp cho GPU Server (Vast.ai) / Linux Production.
    """
    def extract_text(self, content: bytes, file_type: str) -> List[Dict[str, Any]]:
        # Đối với .txt, .pdf, .docx đơn giản, ta vẫn có thể dùng Basic logic để nhanh hơn
        # Nhưng ở đây ta minh họa việc dùng Unstructured hoàn toàn.
        pages = []
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                if file_type in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(tmp_path)
                elif file_type == ".pdf":
                    # Unstructured có thể xử lý PDF rất tốt (bao gồm cả bảng biểu)
                    from langchain_community.document_loaders import UnstructuredPDFLoader
                    loader = UnstructuredPDFLoader(tmp_path)
                else:
                    # Mặc định cho các loại khác
                    from langchain_community.document_loaders import UnstructuredFileLoader
                    loader = UnstructuredFileLoader(tmp_path)
                
                docs = loader.load()
                full_text = "\n\n".join(d.page_content for d in docs)
                if full_text.strip():
                    pages.append({"text": full_text, "page_num": 1})
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"ProductionProcessor Error ({file_type}): {e}")
            # Fallback to basic error message if unstructured fails
            pages.append({"text": f"[LỖI PRODUCTION]: Không thể trích xuất văn bản. Kiểm tra dependencies (soffice, pandoc...). Chi tiết: {e}", "page_num": 1})
            
        return pages
