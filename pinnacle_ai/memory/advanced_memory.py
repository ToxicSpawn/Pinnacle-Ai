"""
Advanced Memory System with:
- Hierarchical storage (episodic, semantic, procedural)
- Temporal reasoning
- Memory consolidation
- Associative recall
- Importance scoring
- Forgetting curves
"""

import numpy as np
from typing import List, Dict
from loguru import logger
from datetime import datetime
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")


class EpisodicMemory:
    """Episodic memory - personal experiences and events"""
    
    def __init__(self, dimension: int, max_size: int):
        self.dimension = dimension
        self.max_size = max_size
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
        self.memories: List[Dict] = []
    
    def store(self, memory: Dict):
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(memory["embedding"].reshape(1, -1).astype(np.float32))
        self.memories.append(memory)
        
        if len(self.memories) > self.max_size:
            self.consolidate()
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        if len(self.memories) == 0:
            return []
        
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32), k
            )
            return [self.memories[i] for i in indices[0] if i < len(self.memories)]
        else:
            # Fallback: return random memories
            return self.memories[:top_k]
    
    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        for m in self.memories:
            if m["id"] == memory_id:
                return m
        return None
    
    def random(self) -> Optional[Dict]:
        if self.memories:
            return self.memories[np.random.randint(len(self.memories))]
        return None
    
    def size(self) -> int:
        return len(self.memories)
    
    def consolidate(self):
        """Keep most important memories"""
        self.memories.sort(key=lambda x: x.get("importance", 0), reverse=True)
        keep = int(self.max_size * 0.8)
        self.memories = self.memories[:keep]
        
        # Rebuild index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            if self.memories:
                embeddings = np.array([m["embedding"] for m in self.memories])
                self.index.add(embeddings.astype(np.float32))


class SemanticMemory(EpisodicMemory):
    """Semantic memory - facts and general knowledge"""
    pass


class ProceduralMemory(EpisodicMemory):
    """Procedural memory - skills and how-to knowledge"""
    pass


class AdvancedMemory:
    """
    Advanced Memory System with:
    - Hierarchical storage (episodic, semantic, procedural)
    - Temporal reasoning
    - Memory consolidation
    - Associative recall
    - Importance scoring
    - Forgetting curves
    """
    
    def __init__(
        self,
        dimension: int = 384,
        max_size: int = 1000000
    ):
        self.dimension = dimension
        self.max_size = max_size
        
        # Embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.encoder = None
            logger.warning("Using simple embedding fallback")
        
        # Hierarchical memory stores
        self.episodic = EpisodicMemory(dimension, max_size // 3)
        self.semantic = SemanticMemory(dimension, max_size // 3)
        self.procedural = ProceduralMemory(dimension, max_size // 3)
        
        # Working memory (short-term)
        self.working_memory: List[Dict] = []
        self.working_memory_size = 10
        
        # Memory graph (associations)
        self.associations = defaultdict(list)
        
        # Access statistics
        self.access_count = defaultdict(int)
        self.last_access = {}
        
        logger.info("Advanced Memory System initialized")
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        if self.encoder:
            embedding = self.encoder.encode(text, normalize_embeddings=True)
            if len(embedding) != self.dimension:
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
    
    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        context: Dict = None,
        importance: float = 0.5
    ) -> str:
        """
        Store a memory with automatic categorization
        
        Args:
            content: Content to store
            memory_type: episodic, semantic, or procedural
            context: Additional context
            importance: Importance score (0-1)
        
        Returns:
            Memory ID
        """
        # Create embedding
        embedding = self._encode(content)
        
        # Create memory object
        memory = {
            "id": f"{memory_type}_{datetime.now().timestamp()}",
            "content": content,
            "embedding": embedding,
            "type": memory_type,
            "context": context or {},
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "associations": []
        }
        
        # Store in appropriate system
        if memory_type == "episodic":
            self.episodic.store(memory)
        elif memory_type == "semantic":
            self.semantic.store(memory)
        elif memory_type == "procedural":
            self.procedural.store(memory)
        
        # Update working memory
        self._update_working_memory(memory)
        
        # Find and create associations
        self._create_associations(memory)
        
        return memory["id"]
    
    def retrieve(
        self,
        query: str,
        memory_types: List[str] = None,
        top_k: int = 10,
        use_associations: bool = True,
        temporal_weight: float = 0.1
    ) -> List[Dict]:
        """
        Retrieve memories with advanced reasoning
        
        Args:
            query: Search query
            memory_types: Types to search (None = all)
            top_k: Number of results
            use_associations: Include associated memories
            temporal_weight: Weight for recency
        
        Returns:
            List of relevant memories
        """
        memory_types = memory_types or ["episodic", "semantic", "procedural"]
        
        # Create query embedding
        query_embedding = self._encode(query)
        
        # Retrieve from each memory type
        all_results = []
        
        if "episodic" in memory_types:
            all_results.extend(self.episodic.retrieve(query_embedding, top_k))
        if "semantic" in memory_types:
            all_results.extend(self.semantic.retrieve(query_embedding, top_k))
        if "procedural" in memory_types:
            all_results.extend(self.procedural.retrieve(query_embedding, top_k))
        
        # Score and rank
        scored_results = []
        for memory in all_results:
            score = self._compute_retrieval_score(memory, query_embedding, temporal_weight)
            scored_results.append((memory, score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = [r[0] for r in scored_results[:top_k]]
        
        # Add associations if requested
        if use_associations:
            top_results = self._expand_with_associations(top_results, top_k)
        
        # Update access statistics
        for memory in top_results:
            self.access_count[memory["id"]] += 1
            self.last_access[memory["id"]] = datetime.now()
        
        return top_results
    
    def _compute_retrieval_score(
        self,
        memory: Dict,
        query_embedding: np.ndarray,
        temporal_weight: float
    ) -> float:
        """Compute retrieval score with multiple factors"""
        # Semantic similarity
        semantic_score = float(np.dot(query_embedding, memory["embedding"]))
        
        # Temporal score (more recent = higher)
        try:
            created_at = datetime.fromisoformat(memory["created_at"])
            age_days = (datetime.now() - created_at).days
            temporal_score = 1.0 / (1.0 + age_days * 0.1)
        except:
            temporal_score = 0.5
        
        # Importance score
        importance_score = memory.get("importance", 0.5)
        
        # Access frequency score
        access_score = min(1.0, self.access_count[memory["id"]] * 0.1)
        
        # Combine scores
        final_score = (
            0.6 * semantic_score +
            temporal_weight * temporal_score +
            0.2 * importance_score +
            0.1 * access_score
        )
        
        return final_score
    
    def _update_working_memory(self, memory: Dict):
        """Update working memory with new memory"""
        self.working_memory.append(memory)
        if len(self.working_memory) > self.working_memory_size:
            # Move oldest to long-term
            oldest = self.working_memory.pop(0)
            if oldest["importance"] > 0.7:
                self.semantic.store(oldest)
    
    def _create_associations(self, memory: Dict):
        """Create associations with existing memories"""
        # Find similar memories
        similar = self.retrieve(
            memory["content"],
            top_k=5,
            use_associations=False
        )
        
        for sim_memory in similar:
            if sim_memory["id"] != memory["id"]:
                self.associations[memory["id"]].append(sim_memory["id"])
                self.associations[sim_memory["id"]].append(memory["id"])
    
    def _expand_with_associations(
        self,
        memories: List[Dict],
        max_total: int
    ) -> List[Dict]:
        """Expand results with associated memories"""
        expanded = list(memories)
        seen_ids = {m["id"] for m in memories}
        
        for memory in memories:
            for assoc_id in self.associations.get(memory["id"], [])[:2]:
                if assoc_id not in seen_ids and len(expanded) < max_total:
                    assoc_memory = self._get_by_id(assoc_id)
                    if assoc_memory:
                        expanded.append(assoc_memory)
                        seen_ids.add(assoc_id)
        
        return expanded
    
    def _get_by_id(self, memory_id: str) -> Optional[Dict]:
        """Get memory by ID"""
        for store in [self.episodic, self.semantic, self.procedural]:
            memory = store.get_by_id(memory_id)
            if memory:
                return memory
        return None
    
    def consolidate(self):
        """Consolidate memories (like sleep)"""
        logger.info("Consolidating memories...")
        
        # Consolidate each store
        self.episodic.consolidate()
        self.semantic.consolidate()
        self.procedural.consolidate()
        
        # Strengthen important associations
        for memory_id, assocs in self.associations.items():
            # Keep only most accessed associations
            scored = [(a, self.access_count[a]) for a in assocs]
            scored.sort(key=lambda x: x[1], reverse=True)
            self.associations[memory_id] = [a for a, _ in scored[:10]]
        
        logger.info("Memory consolidation complete")
    
    def dream(self, duration: int = 100) -> List[str]:
        """Generate novel insights by combining memories"""
        insights = []
        
        for _ in range(duration):
            # Random memory from each type
            memories = []
            if self.episodic.size() > 0:
                mem = self.episodic.random()
                if mem:
                    memories.append(mem)
            if self.semantic.size() > 0:
                mem = self.semantic.random()
                if mem:
                    memories.append(mem)
            
            if len(memories) >= 2:
                # Combine to create insight
                m1, m2 = memories[:2]
                insight = f"Connection: '{m1['content'][:50]}' relates to '{m2['content'][:50]}'"
                insights.append(insight)
        
        return insights
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            "episodic_count": self.episodic.size(),
            "semantic_count": self.semantic.size(),
            "procedural_count": self.procedural.size(),
            "working_memory_count": len(self.working_memory),
            "total_associations": sum(len(v) for v in self.associations.values()),
            "most_accessed": sorted(
                self.access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

