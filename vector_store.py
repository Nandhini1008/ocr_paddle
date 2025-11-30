import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import config
import uuid

class VectorDB:
    def __init__(self):
        # Use Qdrant Cloud if URL is provided, otherwise use local
        if config.QDRANT_URL and config.QDRANT_API_KEY:
            print(f"Connecting to Qdrant Cloud: {config.QDRANT_URL}")
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
        else:
            print(f"Connecting to local Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.collection_name = config.COLLECTION_NAME
        
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

    def embed_text(self, text):
        return self.encoder.encode(text).tolist()

    def add_document(self, text, metadata=None):
        if metadata is None:
            metadata = {}
            
        # Simple chunking strategy: split by paragraphs or fixed size
        # For now, let's split by paragraphs (double newline)
        chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        points = []
        for chunk in chunks:
            vector = self.embed_text(chunk)
            payload = metadata.copy()
            payload['text'] = chunk
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            ))
            
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        return len(points)

    def search(self, query, k=5):
        query_vector = self.embed_text(query)
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        
        results = []
        for hit in search_result:
            results.append({
                'text': hit.payload.get('text', ''),
                'score': hit.score,
                'metadata': {k:v for k,v in hit.payload.items() if k != 'text'}
            })
            
        return results
