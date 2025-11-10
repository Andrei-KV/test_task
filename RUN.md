# How to Run the Application

This guide provides step-by-step instructions to set up and run the RAG chatbot application locally for development and testing.

## Prerequisites

- **Docker and Docker Compose:** Required to run the backend services (PostgreSQL, Qdrant, Redis).
- **Python 3.12+:** The application is built on Python 3.12.
- **Poetry:** Used for managing Python dependencies.
- **pyenv:** Recommended for managing Python versions.

## 1. Environment Setup

### a. Clone the Repository

Clone the project to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### b. Configure Environment Variables

Create a `.env` file in the project root. This file is critical for storing database credentials, API keys, and other configuration.

```bash
cp .env.example .env
```

Now, open the `.env` file and fill in the required values:

- `DB_PASSWORD`: A secure password for the PostgreSQL database.
- `DEEPSEEK_API_KEY`: Your API key for the DeepSeek LLM.
- Other variables can typically be left with their default values for local development.

### c. Install Python Dependencies

Using Poetry, install all the required Python packages:

```bash
poetry install
```

## 2. Start Backend Services

All required backend services are managed by Docker Compose. To start them, run the following command from the project root:

```bash
docker-compose up -d
```

This will start three containers in detached mode:
- `legal_rag_postgres`: The PostgreSQL database.
- `legal_rag_qdrant`: The Qdrant vector database.
- `legal_rag_redis`: The Redis server for caching and session storage.

You can check the status of the containers with `docker-compose ps`.

## 3. Run the FastAPI Application

Once the backend services are running, you can start the main FastAPI application. The server should be run from the project root using Poetry to ensure it uses the correct virtual environment.

```bash
poetry run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--reload`: This flag enables hot-reloading, which is useful for development as the server will automatically restart after code changes.

The application will now be running and accessible at **http://localhost:8000**.

## 4. (Optional) Data Ingestion

To process and load new documents into the vector database, you need to run the ingestion script manually.

```bash
poetry run python src/process_new_documents_test.py
```

This script will scan for new documents, process them, generate embeddings, and store them in PostgreSQL and Qdrant.
