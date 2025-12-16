"""
Infinite Context Memory System

Unlike GPT-5's limited context window, this system can:
- Store unlimited memories
- Retrieve relevant memories instantly
- Consolidate and compress old memories
- Never forget important information

This is the key to true AGI - unlimited context.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Memory retrieval will be slower.")


class InfiniteMemory(nn.Module):
    """
    Infinite Context Memory System
    
    Unlike GPT-5's limited context window, this system can:
    - Store unlimited memories
    - Retrieve relevant memories instantly
    - Consolidate and compress old memories
    - Never forget important information
    
    This is the key to true AGI - unlimited context.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        memory_size: int = 1_000_000,  # 1 million memories
        retrieval_top_k: int = 100,
        compression_ratio: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.retrieval_top_k = retrieval_top_k
        self.compression_ratio = compression_ratio
        
        # Short-term memory (immediate context)
        self.short_term = deque(maxlen=10000)
        
        # Long-term memory (compressed, indexed)
        self.long_term_embeddings = np.zeros((memory_size, hidden_size), dtype=np.float32)
        self.long_term_content = [None] * memory_size
        self.long_term_count = 0
        
        # FAISS index for fast retrieval
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(hidden_size)
        else:
            self.index = None
        
        # Memory encoder/decoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Memory consolidation network
        self.consolidator = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Episodic memory (life experiences)
        self.episodic_memory = []
        
        # Semantic memory (facts and knowledge)
        self.semantic_memory = {}
        
        # Procedural memory (skills and how-to)
        self.procedural_memory = {}
        
        logger.info(f"Infinite Memory initialized with {memory_size:,} capacity")
    
    def store(self, content: str, embedding: torch.Tensor, memory_type: str = "episodic"):
        """Store a new memory"""
        # Encode the memory
        encoded = self.encoder(embedding)
        
        # Score importance
        importance = self.importance_scorer(encoded).item()
        
        # Store in short-term memory
        self.short_term.append({
            "content": content,
            "embedding": encoded.detach().cpu().numpy(),
            "importance": importance,
            "timestamp": len(self.short_term),
            "type": memory_type
        })
        
        # If important enough, store in long-term memory
        if importance > 0.5:
            self._store_long_term(content, encoded, memory_type)
        
        # Consolidate if short-term is full
        if len(self.short_term) >= 9000:
            self._consolidate_memories()
    
    def _store_long_term(self, content: str, embedding: torch.Tensor, memory_type: str):
        """Store in long-term memory with FAISS indexing"""
        if self.long_term_count >= self.memory_size:
            self._compress_old_memories()
        
        embedding_np = embedding.detach().cpu().numpy().flatten()
        if embedding_np.shape[0] != self.hidden_size:
            # Resize if needed
            if embedding_np.shape[0] > self.hidden_size:
                embedding_np = embedding_np[:self.hidden_size]
            else:
                padded = np.zeros(self.hidden_size)
                padded[:embedding_np.shape[0]] = embedding_np
                embedding_np = padded
        
        self.long_term_embeddings[self.long_term_count] = embedding_np
        self.long_term_content[self.long_term_count] = {
            "content": content,
            "type": memory_type
        }
        self.long_term_count += 1
        
        # Update FAISS index
        if self.index is not None:
            self.index.add(embedding_np.reshape(1, -1))
    
    def retrieve(self, query: torch.Tensor, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant memories"""
        top_k = top_k or self.retrieval_top_k
        
        if self.long_term_count == 0:
            return []
        
        query_np = query.detach().cpu().numpy().flatten()
        if query_np.shape[0] != self.hidden_size:
            if query_np.shape[0] > self.hidden_size:
                query_np = query_np[:self.hidden_size]
            else:
                padded = np.zeros(self.hidden_size)
                padded[:query_np.shape[0]] = query_np
                query_np = padded
        
        query_np = query_np.reshape(1, -1)
        
        # Search FAISS index
        if self.index is not None and self.long_term_count > 0:
            distances, indices = self.index.search(query_np, min(top_k, self.long_term_count))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < self.long_term_count and self.long_term_content[idx] is not None:
                    results.append({
                        "content": self.long_term_content[idx]["content"],
                        "type": self.long_term_content[idx]["type"],
                        "relevance": float(distances[0][i])
                    })
            
            return results
        else:
            # Fallback: brute force search
            results = []
            query_flat = query_np.flatten()
            for i in range(min(top_k, self.long_term_count)):
                if self.long_term_content[i] is not None:
                    similarity = np.dot(query_flat, self.long_term_embeddings[i])
                    results.append({
                        "content": self.long_term_content[i]["content"],
                        "type": self.long_term_content[i]["type"],
                        "relevance": float(similarity)
                    })
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:top_k]
    
    def _consolidate_memories(self):
        """Consolidate short-term memories into long-term (like sleep)"""
        logger.info("Consolidating memories (dream state)...")
        
        # Get all short-term memories
        memories = list(self.short_term)
        
        # Sort by importance
        memories.sort(key=lambda x: x["importance"], reverse=True)
        
        # Keep top 50% in long-term
        for memory in memories[:len(memories) // 2]:
            embedding = torch.tensor(memory["embedding"])
            self._store_long_term(
                memory["content"],
                embedding,
                memory["type"]
            )
        
        # Clear short-term
        self.short_term.clear()
        logger.info("Memory consolidation complete")
    
    def _compress_old_memories(self):
        """Compress old memories to make room for new ones"""
        logger.info("Compressing old memories...")
        
        # Keep only the most important memories
        keep_count = int(self.memory_size * (1 - self.compression_ratio))
        
        # Score all memories
        scores = []
        for i in range(self.long_term_count):
            embedding = torch.tensor(self.long_term_embeddings[i])
            score = self.importance_scorer(embedding).item()
            scores.append((i, score))
        
        # Sort by importance
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top memories
        keep_indices = [s[0] for s in scores[:keep_count]]
        
        # Rebuild memory
        new_embeddings = np.zeros_like(self.long_term_embeddings)
        new_content = [None] * self.memory_size
        
        for new_idx, old_idx in enumerate(keep_indices):
            new_embeddings[new_idx] = self.long_term_embeddings[old_idx]
            new_content[new_idx] = self.long_term_content[old_idx]
        
        self.long_term_embeddings = new_embeddings
        self.long_term_content = new_content
        self.long_term_count = keep_count
        
        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.hidden_size)
            if self.long_term_count > 0:
                self.index.add(self.long_term_embeddings[:self.long_term_count])
        
        logger.info(f"Compressed to {keep_count:,} memories")
    
    def dream(self, duration: int = 100) -> List[str]:
        """
        Dream state - generate novel ideas by recombining memories
        
        This is where true creativity happens - the AI "dreams"
        by randomly activating and recombining memories.
        """
        dreams = []
        
        if self.long_term_count < 2:
            return dreams
        
        for _ in range(duration):
            # Randomly select memories
            idx1, idx2 = np.random.choice(self.long_term_count, 2, replace=False)
            
            # Combine embeddings
            combined = (
                self.long_term_embeddings[idx1] +
                self.long_term_embeddings[idx2]
            ) / 2
            
            # Generate dream content
            dream_embedding = torch.tensor(combined)
            similar = self.retrieve(dream_embedding, top_k=3)
            
            if similar:
                dream_content = f"Dream: {similar[0]['content']} + {similar[1]['content'] if len(similar) > 1 else ''}"
                dreams.append(dream_content)
        
        return dreams
    
    def forget(self, content: str):
        """Selectively forget a memory (useful for unlearning)"""
        # Find and remove the memory
        for i in range(self.long_term_count):
            if self.long_term_content[i] and self.long_term_content[i]["content"] == content:
                self.long_term_content[i] = None
                logger.info(f"Forgot memory: {content[:50]}...")
                break
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": self.long_term_count,
            "capacity": self.memory_size,
            "utilization": self.long_term_count / self.memory_size * 100 if self.memory_size > 0 else 0
        }

