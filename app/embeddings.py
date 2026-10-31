from sentence_transformers import SentenceTransformer
from app.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self):
        logger.info("Loading embedding model")
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text: str):
        logger.debug("Creating embedding for %d characters", len(text))
        embedding = self.model.encode(text).tolist()
        logger.debug("Created embedding with %d dimensions", len(embedding))
        return embedding
