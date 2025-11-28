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
    EMBEDDING_DIMENSION
)
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class QueryOpenSearchClient:
    """
    OpenSearch client for hybrid search combining vector (knn) and full-text (match) search.
    Uses AsyncOpenSearch for async operations and CrossEncoder for reranking.
    """
    
    def __init__(self, host: str, port: int, index_name: str, use_ssl: bool = False, verify_certs: bool = False):
        self.__index_name = index_name
        
        # Initialize AsyncOpenSearch client
        self.__client = AsyncOpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=None,  # No auth for development (DISABLE_SECURITY_PLUGIN=true)
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False
        )
        
        # Initialize CrossEncoder for reranking
        if not hasattr(self, '_QueryOpenSearchClient__cross_encoder'):
            self.__cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        logger.info(f"OpenSearch client initialized: {host}:{port}, index: {index_name}")
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid function for score normalization."""
        return 1 / (1 + np.exp(-x))
    
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
                    "chunk_id": {
                        "type": "integer"
                    },
                    "page_number": {
                        "type": "integer"
                    },
                    "type": {
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
        limit_k: int = 20
    ) -> tuple[List[int], Optional[int], float]:
        """
        Performs hybrid search combining knn (vector) and match (full-text) queries.
        
        Args:
            query_vector: Embedding vector for the query
            user_query: Text query for full-text search
            limit_k: Number of results to return
            
        Returns:
            Tuple of (chunk_ids, document_id, max_score)
        """
        logger.info("Performing hybrid search in OpenSearch...")
        
        # Hybrid search query combining knn and match
        search_body = {
            "size": limit_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": limit_k
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
            "_source": ["content", "document_id", "chunk_id", "page_number", "type", "sheet_name"]
        }
        
        try:
            response = await self.__client.search(
                index=self.__index_name,
                body=search_body
            )
            
            hits = response['hits']['hits']
            logger.info(f"Hybrid search found {len(hits)} candidates.")
            
            if not hits:
                logger.warning("No candidates found from hybrid search.")
                return [], None, 0
            
            max_score = hits[0]['_score'] if hits else 0
            
            # Extract candidates
            candidates = []
            for hit in hits:
                candidate = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'payload': hit['_source']
                }
                candidates.append(candidate)
            
            # Filter valid candidates
            valid_candidates = [
                c for c in candidates 
                if c['payload'] and 'content' in c['payload']
            ]
            
            if not valid_candidates:
                logger.warning("No valid candidates left after filtering for content.")
                return [], None, 0
            
            # Rerank with CrossEncoder
            logger.info(f"Reranking {len(valid_candidates)} candidates...")
            rerank_pairs = [[user_query, c['payload']['content']] for c in valid_candidates]
            reranked_scores = await asyncio.to_thread(self.__cross_encoder.predict, rerank_pairs)
            
            # Normalize scores and update candidates
            normalized_scores = self._sigmoid(reranked_scores)
            for candidate, score in zip(valid_candidates, normalized_scores):
                candidate['score'] = score
            
            # Sort by reranked score
            valid_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            logger.debug(f"Top 8 candidates after reranking:")
            for i, candidate in enumerate(valid_candidates[:8]):
                logger.debug(
                    f"{i+1}. ID: {candidate['id']}, Score: {candidate['score']:.4f}, "
                    f"DocID: {candidate['payload'].get('document_id')}"
                )
            
            # Get most relevant document_id
            first_candidate = valid_candidates[0]
            if not first_candidate['payload'] or 'document_id' not in first_candidate['payload']:
                logger.error("The most relevant candidate is missing 'document_id'.")
                return [], None, 0
            
            target_document_id = first_candidate['payload']['document_id']
            logger.info(f"Most relevant document_id: {target_document_id}")
            
            # Filter candidates by target document
            filtered_candidates = [
                c for c in valid_candidates
                if c['payload'] and c['payload'].get('document_id') == target_document_id
            ]
            
            top_relevant_chunks = filtered_candidates[:8]
            
            if not top_relevant_chunks:
                logger.warning(f"No relevant chunks found for document_id: {target_document_id}")
                return [], None, 0
            
            logger.info(f"Selected {len(top_relevant_chunks)} top relevant chunks from target document.")
            
            # Collect chunk_ids
            target_chunk_ids = {
                c['payload']['chunk_id'] for c in top_relevant_chunks
                if c['payload'] and 'chunk_id' in c['payload']
            }
            
            if not target_chunk_ids:
                logger.error("Selected chunks are missing 'chunk_id'.")
                return [], None, 0
            
            return sorted(list(target_chunk_ids)), target_document_id, first_candidate['score']
            
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
