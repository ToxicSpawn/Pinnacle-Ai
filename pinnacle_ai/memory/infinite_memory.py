import numpy as np
from typing import List, Dict, Optional
from loguru import logger
import json
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Memory will use simple embeddings.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Memory retrieval will be slower.")

from collections import deque


class InfiniteMemory:
    """
    Infinite Memory System with semantic retrieval
    
    Features:
    - Semantic search using embeddings
    - Fast retrieval with FAISS
    - Memory consolidation
    - Importance scoring
    """
    
    def __init__(
        self,
        dimension: int = 384,
        max_size: int = 100000,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.dimension = dimension
        self.max_size = max_size
        
        # Embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {model_name}")
            self.encoder = SentenceTransformer(model_name)
        else:
            logger.warning("Using simple embedding fallback")
            self.encoder = None
        
        # FAISS index for fast retrieval
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
        
        # Memory storage
        self.memories: List[Dict] = []
        
        # Short-term buffer
        self.short_term = deque(maxlen=1000)
        
        # Memory statistics
        self.total_stored = 0
        self.total_retrieved = 0
        
        logger.info(f"Infinite Memory initialized (capacity: {max_size:,})")
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        if self.encoder:
            embedding = self.encoder.encode(text, normalize_embeddings=True)
            if len(embedding) != self.dimension:
                # Resize if needed
                if len(embedding) > self.dimension:
                    embedding = embedding[:self.dimension]
                else:
                    padded = np.zeros(self.dimension)
                    padded[:len(embedding)] = embedding
                    embedding = padded
            return embedding
        else:
            # Simple fallback
            return np.random.randn(self.dimension).astype(np.float32)
    
    def store(self, text: str, context: str = "", importance: float = 0.5) -> Dict:
        """
        Store a memory
        
        Args:
            text: Text to store
            context: Additional context
            importance: Importance score (0-1)
        
        Returns:
            Storage confirmation
        """
        # Create embedding
        embedding = self._encode(text)
        
        # Create memory object
        memory = {
            "id": len(self.memories),
            "text": text,
            "context": context,
            "importance": importance,
            "access_count": 0,
            "embedding": embedding
        }
        
        # Add to index
        if self.index is not None:
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        # Store memory
        self.memories.append(memory)
        self.short_term.append(memory)
        self.total_stored += 1
        
        # Consolidate if needed
        if len(self.memories) > self.max_size:
            self._consolidate()
        
        return {"status": "stored", "id": memory["id"]}
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories
        
        Args:
            query: Query text
            top_k: Number of memories to retrieve
        
        Returns:
            List of relevant memories
        """
        if len(self.memories) == 0:
            return []
        
        # Create query embedding
        query_embedding = self._encode(query)
        
        # Search
        if self.index is not None and self.index.ntotal > 0:
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            
            # Get memories
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.memories):
                    memory = self.memories[idx]
                    memory["access_count"] += 1
                    results.append({
                        "text": memory["text"],
                        "context": memory["context"],
                        "relevance": float(dist),
                        "importance": memory["importance"]
                    })
        else:
            # Fallback: simple text matching
            results = []
            query_lower = query.lower()
            for memory in self.memories:
                if query_lower in memory["text"].lower():
                    results.append({
                        "text": memory["text"],
                        "context": memory["context"],
                        "relevance": 0.5,
                        "importance": memory["importance"]
                    })
            results = results[:top_k]
        
        self.total_retrieved += len(results)
        return results
    
    def _consolidate(self):
        """Consolidate memories by removing less important ones"""
        logger.info("Consolidating memories...")
        
        # Sort by importance and access count
        scored = [(i, m["importance"] + m["access_count"] * 0.01)
                  for i, m in enumerate(self.memories)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top memories
        keep_count = int(self.max_size * 0.8)
        keep_indices = set([s[0] for s in scored[:keep_count]])
        
        # Rebuild
        new_memories = []
        new_embeddings = []
        
        for i, memory in enumerate(self.memories):
            if i in keep_indices:
                memory["id"] = len(new_memories)
                new_memories.append(memory)
                new_embeddings.append(memory["embedding"])
        
        # Rebuild index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            if new_embeddings:
                self.index.add(np.array(new_embeddings).astype(np.float32))
        
        self.memories = new_memories
        logger.info(f"Consolidated to {len(self.memories):,} memories")
    
    def size(self) -> int:
        """Get number of stored memories"""
        return len(self.memories)
    
    def clear(self):
        """Clear all memories"""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.memories = []
        self.short_term.clear()
    
    def dream(self, duration: int = 50) -> List[str]:
        """
        Dream mode - generate novel combinations
        
        Args:
            duration: Number of dream iterations
        
        Returns:
            List of dream insights
        """
        if len(self.memories) < 2:
            return []
        
        dreams = []
        for _ in range(duration):
            # Random memory combination
            idx1, idx2 = np.random.choice(len(self.memories), 2, replace=False)
            m1, m2 = self.memories[idx1], self.memories[idx2]
            
            # Create dream insight
            dream = f"Connection: '{m1['text'][:50]}...' relates to '{m2['text'][:50]}...'"
            dreams.append(dream)
        
        return dreams
    
    def save(self, path: str):
        """Save memories to disk"""
        data = {
            "memories": [
                {k: v for k, v in m.items() if k != "embedding"}
                for m in self.memories
            ],
            "stats": {
                "total_stored": self.total_stored,
                "total_retrieved": self.total_retrieved
            }
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings separately
        if self.memories:
            embeddings = np.array([m["embedding"] for m in self.memories])
            np.save(path.replace(".json", "_embeddings.npy"), embeddings)
    
    def load(self, path: str):
        """Load memories from disk"""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Load embeddings
        embedding_path = path.replace(".json", "_embeddings.npy")
        if os.path.exists(embedding_path):
            embeddings = np.load(embedding_path)
        else:
            embeddings = None
        
        # Rebuild
        self.memories = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        for i, m in enumerate(data["memories"]):
            if embeddings is not None and i < len(embeddings):
                m["embedding"] = embeddings[i]
            else:
                m["embedding"] = self._encode(m["text"])
            self.memories.append(m)
        
        if FAISS_AVAILABLE and len(self.memories) > 0:
            embeddings_array = np.array([m["embedding"] for m in self.memories])
            self.index.add(embeddings_array.astype(np.float32))
        
        self.total_stored = data["stats"]["total_stored"]
        self.total_retrieved = data["stats"]["total_retrieved"]
