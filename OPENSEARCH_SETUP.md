# OpenSearch Migration - Quick Start

## Prerequisites
- Docker and Docker Compose installed
- Python 3.12+

## Steps

### 1. Install Dependencies
```bash
poetry install
```

### 2. Start OpenSearch
```bash
docker-compose -f docker-compose.opensearch.yml up -d
```

Wait ~30 seconds for OpenSearch to start.

### 3. Verify OpenSearch is Running
```bash
curl http://localhost:9200
```

You should see cluster information in JSON format.

### 4. Test Connection and Create Index
```bash
poetry run python test_opensearch_connection.py
```

This will:
- Connect to OpenSearch
- Create the `rag_chunks` index with proper mappings
- Verify the index was created successfully

### 5. Stop OpenSearch (when done)
```bash
docker-compose -f docker-compose.opensearch.yml down
```

To remove data volume:
```bash
docker-compose -f docker-compose.opensearch.yml down -v
```

## Next Steps

After successful testing:
1. Update `rag_service.py` to use OpenSearch instead of Qdrant
2. Update document ingestion pipeline
3. Re-index documents
4. Test full RAG pipeline

## Troubleshooting

**OpenSearch won't start:**
- Check if port 9200 is already in use: `lsof -i :9200`
- Check Docker logs: `docker logs opensearch`

**Connection refused:**
- Wait longer for OpenSearch to fully start
- Check container status: `docker ps`

**Index creation fails:**
- Verify OpenSearch version supports knn plugin
- Check OpenSearch logs for errors
