#!/usr/bin/env python3
"""
kb_index.py

PostgreSQL Knowledge Base Vector Index

Provides similarity-based retrieval for KB entries using:
- Sentence transformers for embeddings
- FAISS or scikit-learn for efficient similarity search
- Hybrid retrieval combining keyword and semantic search

Usage:
    from kb_index import KBVectorIndex
    
    # Create index
    index = KBVectorIndex()
    index.build(kb_entries)
    
    # Search by similarity
    results = index.search("slow query performance", top_k=5)
    
    # Hybrid search (keyword + semantic)
    results = index.hybrid_search(
        query="slow query with high latency",
        keywords=["slow", "latency", "query"],
        top_k=5
    )
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from .kb_schema import KBEntry, KBVersion

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a similarity search."""
    entry: KBEntry
    score: float
    match_type: str  # 'semantic', 'keyword', 'hybrid'
    
    def to_dict(self) -> Dict:
        return {
            "kb_id": self.entry.metadata.kb_id,
            "category": self.entry.metadata.category,
            "severity": self.entry.metadata.severity,
            "issue_type": self.entry.problem_identity.issue_type,
            "description": self.entry.problem_identity.short_description,
            "score": self.score,
            "match_type": self.match_type,
            "recommendations": self.entry.get_actionable_recommendations()[:3],
        }


class KBVectorIndex:
    """
    Vector index for KB entries.
    
    Uses sentence-transformers for embedding generation and
    provides efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_faiss: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize the vector index.
        
        Args:
            embedding_model: Name of the sentence-transformers model
            use_faiss: Whether to use FAISS for faster search (requires faiss)
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.embedding_model_name = embedding_model
        self.use_faiss = use_faiss
        self.device = device
        self._model = None
        self._index: Any = None
        self._entries: List[KBEntry] = []
        self._embeddings: Any = None
        self._keyword_index: Dict[str, List[int]] = defaultdict(list)
        self._vocab: Dict[str, int] = {}
        
        # Try to import optional dependencies
        self._faiss_available = False
        self._transformers_available = False
        self._sklearn_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check which optional dependencies are available."""
        try:
            import faiss  # type: ignore
            self._faiss_available = True
            logger.info("FAISS is available for fast similarity search")
        except ImportError:
            logger.debug("FAISS not available, will use sklearn")
        
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._transformers_available = True
            logger.info("sentence-transformers is available")
        except ImportError:
            logger.debug("sentence-transformers not available")
        
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            self._sklearn_available = True
            logger.info("scikit-learn is available for fallback search")
        except ImportError:
            logger.debug("scikit-learn not available")
    
    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        if not self._transformers_available:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self._use_fallback = True
            return
        
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            self._use_fallback = False
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self._use_fallback = True
    
    def build(self, entries: List[KBEntry]) -> None:
        """
        Build the vector index from KB entries.
        
        Args:
            entries: List of KB entries to index
        """
        if not entries:
            logger.warning("No entries to index")
            return
        
        logger.info(f"Building index from {len(entries)} entries")
        
        self._entries = entries
        
        # Generate embeddings
        texts = [entry.get_text_content() for entry in entries]
        self._build_embeddings(texts)
        
        # Build keyword index
        self._build_keyword_index(entries)
        
        logger.info("Index built successfully")
    
    def _build_embeddings(self, texts: List[str]) -> None:
        """Build embedding vectors for texts."""
        import numpy as np
        
        self._load_model()
        
        if self._use_fallback:
            # Use TF-IDF like fallback
            self._embeddings = self._build_tfidf_embeddings(texts)
        else:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embeddings = self._model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            # Normalize embeddings
            self._embeddings = self._embeddings / np.linalg.norm(
                self._embeddings,
                axis=1,
                keepdims=True
            )
        
        # Build search index
        if self._faiss_available and self._embeddings is not None:
            import faiss  # type: ignore
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(self._embeddings)
            logger.info(f"FAISS index created with {self._index.ntotal} vectors")
        elif self._sklearn_available:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            self._index = NearestNeighbors(
                n_neighbors=min(10, len(self._entries)),
                metric='cosine',
                algorithm='brute'
            )
            self._index.fit(self._embeddings)
            logger.info("sklearn NearestNeighbors index created")
    
    def _build_tfidf_embeddings(self, texts: List[str]) -> Any:
        """Build TF-IDF like embeddings for fallback."""
        import numpy as np
        from collections import Counter
        
        # Build vocabulary from all texts
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        # Store vocabulary for query vectorization
        self._vocab = {word: i for i, word in enumerate(sorted(all_words))}
        
        # Build vectors
        vectors = []
        for text in texts:
            words = text.lower().split()
            word_counts = Counter(words)
            
            # TF
            tf = np.zeros(len(self._vocab))
            for word, count in word_counts.items():
                if word in self._vocab:
                    tf[self._vocab[word]] = count
            
            vectors.append(tf)
        
        # Normalize
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.shape[0] > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        
        return vectors
    
    def _query_to_vector(self, query: str) -> Any:
        """Convert query to vector using stored vocabulary."""
        import numpy as np
        from collections import Counter
        
        if not self._vocab:
            logger.warning("Vocabulary not built. Call build() first.")
            return np.array([])
        
        words = query.lower().split()
        word_counts = Counter(words)
        
        vec = np.zeros(len(self._vocab))
        for word, count in word_counts.items():
            if word in self._vocab:
                vec[self._vocab[word]] = count
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def _build_keyword_index(self, entries: List[KBEntry]) -> None:
        """Build inverted index for keyword search."""
        self._keyword_index = defaultdict(list)
        
        for i, entry in enumerate(entries):
            text = entry.get_searchable_text()
            words = set(text.lower().split())
            
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self._keyword_index[word].append(i)
        
        logger.info(f"Keyword index built with {len(self._keyword_index)} terms")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search KB entries by semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter
            severity: Optional severity filter
            
        Returns:
            List of SearchResult objects sorted by score
        """
        import numpy as np
        
        if not self._entries or self._embeddings is None:
            logger.warning("Index not built. Call build() first.")
            return []
        
        # Generate query embedding
        self._load_model()
        
        if self._use_fallback:
            query_vec = self._query_to_vector(query)
        else:
            query_vec = self._model.encode([query], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Search
        if self._faiss_available and self._index is not None:
            import faiss  # type: ignore
            scores, indices = self._index.search(query_vec.reshape(1, -1), top_k)
            scores = scores[0]
            indices = indices[0]
        elif self._sklearn_available and self._index is not None:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            scores, indices = self._index.kneighbors(query_vec.reshape(1, -1))
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback: compute similarities manually
            similarities = np.dot(self._embeddings, query_vec)
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]
        
        # Build results
        results = []
        for i, idx in enumerate(indices):
            if idx >= len(self._entries):
                continue
            
            entry = self._entries[idx]
            
            # Apply filters
            if category and not entry.matches_category(category):
                continue
            if severity and not entry.matches_severity(severity):
                continue
            
            results.append(SearchResult(
                entry=entry,
                score=float(scores[i]),
                match_type="semantic"
            ))
        
        logger.info(f"Semantic search returned {len(results)} results")
        return results
    
    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search KB entries by keyword matching.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not self._entries:
            return []
        
        # Find entries containing keywords
        keyword_scores: Dict[int, float] = defaultdict(float)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for word, entries in self._keyword_index.items():
                word_lower = word.lower()
                if len(keyword_lower) > 0 and len(word_lower) > 0:
                    if keyword_lower in word_lower or word_lower in keyword_lower:
                        for idx in entries:
                            score = 1.0 / (1 + abs(len(word_lower) - len(keyword_lower)))
                            keyword_scores[idx] += score
        
        # Sort by score
        sorted_scores = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for idx, score in sorted_scores:
            entry = self._entries[idx]
            results.append(SearchResult(
                entry=entry,
                score=score,
                match_type="keyword"
            ))
        
        logger.info(f"Keyword search returned {len(results)} results")
        return results
    
    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        category: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query for semantic search
            keywords: Keywords for keyword search
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            category: Optional category filter
            severity: Optional severity filter
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Get semantic results
        semantic_results = self.search(query, top_k=top_k * 2, category=category, severity=severity)
        
        # Get keyword results
        keyword_results = self.keyword_search(keywords, top_k=top_k * 2)
        
        # Normalize scores
        def normalize_scores(results: List[SearchResult]) -> Dict[int, float]:
            if not results:
                return {}
            max_score = max(r.score for r in results)
            if max_score == 0:
                return {id(r.entry): 0 for r in results}
            return {id(r.entry): r.score / max_score for r in results}
        
        semantic_scores = normalize_scores(semantic_results)
        keyword_scores = normalize_scores(keyword_results)
        
        # Combine scores
        combined_scores: Dict[int, Tuple[SearchResult, float]] = {}
        
        for result in semantic_results:
            entry_id = id(result.entry)
            score = semantic_weight * semantic_scores.get(entry_id, 0)
            combined_scores[entry_id] = [result, score]
        
        for result in keyword_results:
            entry_id = id(result.entry)
            keyword_score = keyword_weight * keyword_scores.get(entry_id, 0)
            
            if entry_id in combined_scores:
                combined_scores[entry_id][1] += keyword_score
                if combined_scores[entry_id][0].match_type == "semantic":
                    combined_scores[entry_id][0].match_type = "hybrid"
            else:
                combined_scores[entry_id] = [result, keyword_score]
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Update match type for combined results
        results = []
        for result, score in sorted_results:
            results.append(SearchResult(
                entry=result.entry,
                score=score,
                match_type=result.match_type
            ))
        
        logger.info(f"Hybrid search returned {len(results)} results")
        return results
    
    def get_similar_entries(
        self,
        entry: KBEntry,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find entries similar to a given entry.
        
        Args:
            entry: Reference entry
            top_k: Number of similar entries to return
            
        Returns:
            List of similar entries
        """
        text = entry.get_text_content()
        return self.search(text, top_k=top_k + 1)  # +1 to exclude self
    
    def filter_by_category(self, category: str) -> List[KBEntry]:
        """Get all entries in a category."""
        return [e for e in self._entries if e.matches_category(category)]
    
    def filter_by_severity(self, severity: str) -> List[KBEntry]:
        """Get all entries with a severity level."""
        return [e for e in self._entries if e.matches_severity(severity)]
    
    def filter_by_table(self, table: str) -> List[KBEntry]:
        """Get all entries involving a specific table."""
        return [e for e in self._entries if e.matches_table(table)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        stats = {
            "total_entries": len(self._entries),
            "embedding_dimension": self._embeddings.shape[1] if self._embeddings is not None else 0,
            "embedding_model": self.embedding_model_name if self._transformers_available else "tfidf_fallback",
            "keyword_index_terms": len(self._keyword_index),
            "index_type": "faiss" if self._faiss_available else ("sklearn" if self._sklearn_available else "numpy"),
        }
        
        # Entry counts by category
        category_counts: Dict[str, int] = defaultdict(int)
        for entry in self._entries:
            category_counts[entry.metadata.category] += 1
        stats["category_counts"] = dict(category_counts)
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save the index to disk."""
        import pickle
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entries": self._entries,
            "embeddings": self._embeddings,
            "keyword_index": dict(self._keyword_index),
            "model_name": self.embedding_model_name,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the index from disk."""
        import pickle
        
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self._entries = data["entries"]
        self._embeddings = data["embeddings"]
        self._keyword_index = defaultdict(list, data["keyword_index"])
        self.embedding_model_name = data["model_name"]
        
        # Rebuild search index
        if self._faiss_available and self._embeddings is not None:
            import faiss  # type: ignore
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(self._embeddings)
        
        logger.info(f"Index loaded from {filepath}")


class KBRuleMatcher:
    """
    Rule-based matcher for KB entries.
    
    Provides deterministic matching based on:
    - Category matching
    - Severity matching
    - Symptom matching
    - Cause matching
    - Table matching
    """
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(
        self,
        name: str,
        condition: str,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        priority: int = 0
    ) -> None:
        """
        Add a matching rule.
        
        Args:
            name: Rule name
            condition: Condition type (symptom, cause, category, severity, table)
            category: Category to match
            severity: Severity to match
            priority: Rule priority (higher = more important)
        """
        self.rules.append({
            "name": name,
            "condition": condition,
            "category": category,
            "severity": severity,
            "priority": priority
        })
    
    def match(
        self,
        kb: KBVersion,
        symptoms: Optional[List[str]] = None,
        causes: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Match KB entries based on criteria.
        
        Args:
            kb: Knowledge base to search
            symptoms: Symptoms to match
            causes: Causes to match
            categories: Categories to match
            severities: Severity levels to match
            tables: Tables to match
            top_k: Maximum results
            
        Returns:
            Matched entries with scores
        """
        scores: Dict[int, Tuple[KBEntry, float, List[str]]] = {}
        
        for entry in kb.entries:
            match_reasons = []
            score = 0.0
            
            # Category match
            if categories:
                if entry.metadata.category in categories:
                    score += 2.0
                    match_reasons.append("category")
            
            # Severity match
            if severities:
                if entry.metadata.severity in severities:
                    score += 1.5
                    match_reasons.append("severity")
            
            # Symptom match
            if symptoms:
                for symptom in symptoms:
                    if entry.matches_symptom(symptom):
                        score += 1.0
                        match_reasons.append("symptom")
                        break
            
            # Cause match
            if causes:
                for cause in causes:
                    if entry.matches_cause(cause):
                        score += 1.0
                        match_reasons.append("cause")
                        break
            
            # Table match
            if tables:
                for table in tables:
                    if entry.matches_table(table):
                        score += 0.5
                        match_reasons.append("table")
                        break
            
            if score > 0:
                scores[id(entry)] = (entry, score, match_reasons)
        
        # Sort by score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            SearchResult(
                entry=entry,
                score=score,
                match_type=" | ".join(reasons) if reasons else "general"
            )
            for entry, score, reasons in sorted_results
        ]
    
    def get_diagnosis_recommendations(
        self,
        kb: KBVersion,
        signals: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Get recommendations based on diagnostic signals.
        
        Args:
            kb: Knowledge base
            signals: Dictionary of detected signals
            
        Returns:
            Matching KB entries
        """
        symptoms = signals.get("symptoms", [])
        causes = signals.get("causes", [])
        category = signals.get("category")
        severity = signals.get("severity")
        
        return self.match(
            kb=kb,
            symptoms=symptoms,
            causes=causes,
            categories=[category] if category else None,
            severities=[severity] if severity else None,
            top_k=10
        )

