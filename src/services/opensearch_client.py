import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from opensearchpy import AsyncOpenSearch
from sentence_transformers import CrossEncoder
from src.config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    OPENSEARCH_USE_SSL,
    OPENSEARCH_VERIFY_CERTS,
    OPENSEARCH_PASSWORD,
    SEARCH_LIMIT_FINAL_K,
    SEARCH_KNN_SIZE,
    SEARCH_BM25_SIZE,
    SEARCH_RERANK_LIMIT,
    SEARCH_RRF_K,
    EMBEDDING_DIMENSION
)
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class QueryOpenSearchClient:
    """
    OpenSearch client for hybrid search with RRF (Reciprocal Rank Fusion).
    Combines vector (knn), full-text (BM25), and CrossEncoder reranking.
    """
    
    def __init__(self, host: str, port: int, index_name: str, use_ssl: bool = False, verify_certs: bool = False):
        self.__index_name = index_name
        
        http_auth = ("admin", OPENSEARCH_PASSWORD) if OPENSEARCH_PASSWORD else None

        # Initialize AsyncOpenSearch client
        self.__client = AsyncOpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
            timeout=60
        )
        
        # Initialize CrossEncoder for reranking
        if not hasattr(self, '_QueryOpenSearchClient__cross_encoder'):
            self.__cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        logger.info(f"OpenSearch client initialized: {host}:{port}, index: {index_name}")
    
    @staticmethod
    def _reciprocal_rank_fusion(
        knn_results: List[Dict], 
        bm25_results: List[Dict], 
        k: int = 60
    ) -> List[Dict]:
        """
        Implements Reciprocal Rank Fusion (RRF) algorithm.
        
        RRF Score = sum(1 / (k + rank_i)) for each ranking list
        
        Args:
            knn_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Constant (typically 60)
        
        Returns:
            Merged and re-ranked results
        """
        rrf_scores = {}
        
        # Process kNN results
        for rank, result in enumerate(knn_results, start=1):
            doc_id = result['_id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['_id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Create unified result list
        all_docs = {}
        for result in knn_results + bm25_results:
            doc_id = result['_id']
            if doc_id not in all_docs:
                all_docs[doc_id] = result
        
        # Sort by RRF score
        ranked_results = [
            {**all_docs[doc_id], 'rrf_score': score}
            for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return ranked_results
    
    async def create_index(self):
        """
        Creates the OpenSearch index with proper mappings for hybrid search.
        Includes knn_vector for embeddings and text field for full-text search.
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 200,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "standard",
                            "stopwords": "_russian_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 200,
                                "m": 16
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "document_id": {
                        "type": "integer"
                    },
                    "document_title": {
                        "type": "keyword"
                    },
                    "chunk_id": {
                        "type": "keyword"
                    },
                    "chunk_index": {
                        "type": "integer"
                    },
                    "page_number": {
                        "type": "integer"
                    },
                    "content_type": {
                        "type": "keyword"
                    },
                    "sheet_name": {
                        "type": "keyword"
                    },
                    "qdrant_id": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        try:
            exists = await self.__client.indices.exists(index=self.__index_name)
            if exists:
                logger.info(f"Index '{self.__index_name}' already exists.")
            else:
                await self.__client.indices.create(index=self.__index_name, body=index_body)
                logger.info(f"Index '{self.__index_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def semantic_search(
        self, 
        query_vector: List[float], 
        user_query: str, 
        limit_final_k: int = SEARCH_LIMIT_FINAL_K, 
        knn_size: int = SEARCH_KNN_SIZE,       
        bm25_size: int = SEARCH_BM25_SIZE,      
        rerank_limit: int = SEARCH_RERANK_LIMIT,
        use_rrf: bool = True
    ) -> Dict[str, Any]:
        """
        Performs hybrid search with RRF and CrossEncoder reranking.
        Supports multi-document context retrieval.
        
        Args:
            query_vector: Embedding vector for the query
            user_query: Text query for full-text search
            limit_k: Number of results to return
            use_rrf: Whether to use RRF (True) or native hybrid query (False)
            
        Returns:
            Dict with:
                - chunks: List of top chunks with metadata
                - document_ids: List of unique document IDs
                - max_score: Highest relevance score
        """
        logger.info(f"Performing hybrid search (RRF={use_rrf}) in OpenSearch...")
        SOURCE_FIELD = ["content", "document_id", "document_title", "chunk_id", "chunk_index", "page_number", "content_type", "sheet_name"]
        try:
            if use_rrf:
                # Separate kNN and BM25 queries for manual RRF
                knn_body = {
                    "size": knn_size,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": knn_size
                            }
                        }
                    },
                    "_source": SOURCE_FIELD }
                
                bm25_body = {
                    "size": bm25_size,
                    "query": {
                        "match": {
                            "content": {
                                "query": user_query,
                                "operator": "or"
                            }
                        }
                    },
                    "_source": SOURCE_FIELD
                }
                
                # Execute both searches in parallel
                knn_response, bm25_response = await asyncio.gather(
                    self.__client.search(index=self.__index_name, body=knn_body),
                    self.__client.search(index=self.__index_name, body=bm25_body)
                )
                
                knn_hits = knn_response['hits']['hits']
                bm25_hits = bm25_response['hits']['hits']
                
                logger.info(f"kNN: {len(knn_hits)} results, BM25: {len(bm25_hits)} results")
                
                # Apply RRF
                merged_results = self._reciprocal_rank_fusion(knn_hits, bm25_hits, k=SEARCH_RRF_K)
                hits = merged_results[:rerank_limit]

            else:
                # Use native OpenSearch hybrid query
                search_body = {
                    "size": limit_final_k,
                    "query": {
                        "hybrid": {
                            "queries": [
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": query_vector,
                                            "k": knn_size
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "content": {
                                            "query": user_query,
                                            "boost": 1.0
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "_source": SOURCE_FIELD}
                
                response = await self.__client.search(index=self.__index_name, body=search_body)
                hits = response['hits']['hits']
            
            if not hits:
                logger.warning("No candidates found from hybrid search.")
                return {"chunks": [], "document_ids": [], "max_score": 0}
            
            # Extract candidates
            candidates = []
            for hit in hits:
                candidate = {
                    'id': hit['_id'],
                    'score': hit.get('rrf_score', hit.get('_score', 0)),
                    'payload': hit['_source']
                }
                candidates.append(candidate)
            
            # Filter valid candidates
            valid_candidates = [
                c for c in candidates 
                if c['payload'] and 'content' in c['payload']
            ]
            
            if not valid_candidates:
                logger.warning("No valid candidates after filtering.")
                return {"chunks": [], "document_ids": [], "max_score": 0}
            
            # Rerank with CrossEncoder
            logger.info(f"Reranking {len(valid_candidates)} candidates with CrossEncoder...")
            rerank_pairs = [[user_query, c['payload']['content']] for c in valid_candidates]
            reranked_scores = await asyncio.to_thread(self.__cross_encoder.predict, rerank_pairs)
            
            # Update scores (CrossEncoder returns logits, no need for sigmoid)
            for candidate, score in zip(valid_candidates, reranked_scores):
                candidate['rerank_score'] = float(score)
            
            # Sort by CrossEncoder score
            valid_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Select top chunks (multi-document support)
            top_chunks = valid_candidates[:limit_final_k]
            
            logger.info(f"Top {min(limit_final_k, len(top_chunks))} chunks after reranking:")
            for i, candidate in enumerate(top_chunks[:limit_final_k]):
                logger.info(
                    f"{i+1}. Score: {candidate['rerank_score']:.4f}, "
                    f"Doc: {candidate['payload'].get('document_title', 'N/A')}, "
                    f"Page: {candidate['payload'].get('page_number', 'N/A')}"
                )
            
            # Extract unique document IDs
            document_ids = list(set(
                c['payload']['document_id'] 
                for c in top_chunks 
                if c['payload'] and 'document_id' in c['payload']
            ))
            
            logger.info(f"Context spans {len(document_ids)} document(s): {document_ids}")
            
            # Prepare result
            result_chunks = []
            for candidate in top_chunks:
                result_chunks.append({
                    'chunk_id': candidate['payload'].get('chunk_id'),
                    'document_id': candidate['payload'].get('document_id'),
                    'document_title': candidate['payload'].get('document_title'),
                    'content': candidate['payload'].get('content'),
                    'page_number': candidate['payload'].get('page_number'),
                    'content_type': candidate['payload'].get('content_type'),
                    'score': candidate['rerank_score']
                })
            
            return {
                "chunks": result_chunks,
                "document_ids": document_ids,
                "max_score": top_chunks[0]['rerank_score'] if top_chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            raise
    
    async def close(self):
        """Close the OpenSearch client connection."""
        await self.__client.close()
        logger.info("OpenSearch client connection closed.")


# Create singleton instance
opensearch_client = QueryOpenSearchClient(
    host=OPENSEARCH_HOST,
    port=OPENSEARCH_PORT,
    index_name=OPENSEARCH_INDEX,
    use_ssl=OPENSEARCH_USE_SSL,
    verify_certs=OPENSEARCH_VERIFY_CERTS
)
