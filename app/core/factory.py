import logger_config
from app.core.config import settings
from app.core.interfaces import AIProvider, VectorStoreProvider
from app.providers.ai.ollama import OllamaProvider
from app.providers.ai.mock import MockAIProvider
from app.providers.vectorstore.mongodb import MongoDBVectorProvider
from app.providers.processor.basic import BasicProcessor
from app.providers.processor.production import ProductionProcessor

logger = logger_config.get_logger(__name__)

def get_ai_provider() -> AIProvider:
    if settings.USE_MOCK_AI:
        logger.info("[AI] Using MockAIProvider")
        return MockAIProvider()
    
    logger.info("[AI] Using OllamaProvider")
    return OllamaProvider()

def get_vector_provider() -> VectorStoreProvider:
    # We can add more logic here if we support other vector DBs later
    logger.info("[VECTOR] Using MongoDBVectorProvider")
    return MongoDBVectorProvider()

def get_document_processor():
    if settings.USE_MOCK_AI:
        logger.info("[PROCESSOR] Using BasicProcessor (Local/Dev)")
        return BasicProcessor()
    
    logger.info("[PROCESSOR] Using ProductionProcessor (Unstructured)")
    return ProductionProcessor()
