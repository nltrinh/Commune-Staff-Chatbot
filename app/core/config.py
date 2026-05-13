"""
Cấu hình toàn bộ ứng dụng từ file .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MongoDB
    MONGO_URI: str
    MONGO_DB_NAME: str = "commune_staff_bot"
    COLLECTION_DOCUMENTS: str = "documents"
    COLLECTION_CHAT_HISTORY: str = "chat_history"
    COLLECTION_UPLOADED_FILES: str = "uploaded_files"
    COLLECTION_VECTOR_CACHE: str = "vector_search_cache"
    COLLECTION_USERS: str = "users"
    COLLECTION_DEPARTMENTS: str = "departments"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen2.5:14b"
    OLLAMA_EMBED_MODEL: str = "bge-m3"

    # MongoDB Vector Search Index name
    VECTOR_INDEX_NAME: str = "vector_index"

    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIM: int = 1024  # bge-m3 standard
    MONGODB_VECTOR_MODE: str = "atlas"  # options: "atlas", "native"
    OLLAMA_NUM_PREDICT: int = 1000 

    # API
    APP_TITLE: str = "Trợ lý Hành chính Xã"
    APP_VERSION: str = "2.0.0"
    CORS_ORIGINS: list[str] = ["*"]

    # Security — KHÔNG có giá trị mặc định để bắt buộc đặt trong .env
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours

    # Host Admin
    ADMIN_USERNAME: str = "host_admin"
    ADMIN_PASSWORD: str = "admin123"

    # Môi trường Dev: True = bỏ qua Ollama/VectorSearch, trả về Mock data
    USE_MOCK_AI: bool = False


settings = Settings()
