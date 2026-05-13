import json
import hashlib
import logger_config
from datetime import datetime, timezone
from typing import List, Dict, Generator, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.interfaces import AIProvider, VectorStoreProvider, DocumentProcessor

logger = logger_config.get_logger(__name__)

class RAGService:
    def __init__(self, ai_provider: AIProvider, vector_provider: VectorStoreProvider, processor: DocumentProcessor):
        self.ai = ai_provider
        self.vector_store = vector_provider
        self.processor = processor
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def ingest_file(self, content: bytes, file_name: str, file_type: str, file_id: str, department: Any = "tat_ca") -> Dict[str, Any]:
        file_hash = hashlib.sha256(content).hexdigest()
        pages = self.processor.extract_text(content, file_type)
        
        if not pages:
            raise ValueError("File không có nội dung văn bản.")

        raw_docs = []
        for page in pages:
            doc = Document(
                page_content=page["text"],
                metadata={
                    "file_id": file_id,
                    "source": file_name,
                    "file_name": file_name,
                    "file_type": file_type.lstrip("."),
                    "file_hash": file_hash,
                    "page_num": page["page_num"],
                    "departments": department if isinstance(department, list) else [department],
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raw_docs.append(doc)

        chunks = self.splitter.split_documents(raw_docs)
        
        saved = 0
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_id}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total
            chunk.metadata["doc_id"] = doc_id
            
            # Create embedding
            vector = self.ai.embed_query(chunk.page_content)
            
            # Save to vector store
            success = self.vector_store.save_document(
                doc_id=doc_id,
                content=chunk.page_content,
                embedding=vector,
                metadata=chunk.metadata
            )
            if success:
                saved += 1
                
        return {"chunks_total": total, "chunks_saved": saved}

    def search_context(self, query: str, department: List[str]) -> List[Dict[str, Any]]:
        query_vector = self.ai.embed_query(query)
        
        # Normalize search departments
        if "tat_ca" in department:
            search_filter = {} # Admin/Unrestricted see everything
        else:
            # Restricted: must match user's dept OR be shared
            search_depts = list(set(department + ["tat_ca"]))
            search_filter = {"metadata.departments": {"$in": search_depts}}
            
        return self.vector_store.search(query_vector, settings.TOP_K_RESULTS, search_filter)

    def chat_stream(self, query: str, department: List[str], history: List[Dict[str, str]]) -> Generator[str, None, None]:
        # 1. Search context
        results = self.search_context(query, department)
        
        # 2. Build context string
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('source', 'Unknown')
            context_parts.append(f"[{i}] Nguồn: {source}\n{res['content']}")
        context = "\n\n".join(context_parts)
        
        # 3. Stream response
        for chunk in self.ai.stream_chat(query, context, history):
            yield chunk

    def chat(self, query: str, department: List[str], history: List[Dict[str, str]]) -> Dict[str, Any]:
        # 1. Search context
        results = self.search_context(query, department)
        
        # 2. Build context string
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('source', 'Unknown')
            context_parts.append(f"[{i}] Nguồn: {source}\n{res['content']}")
        context = "\n\n".join(context_parts)
        
        # 4. Generate response (non-streaming)
        full_answer = ""
        for chunk in self.ai.stream_chat(query, context, history):
            full_answer += chunk
            
        return {
            "answer": full_answer,
            "sources": results
        }
