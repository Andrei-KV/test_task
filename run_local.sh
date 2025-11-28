#!/bin/bash
# Script to run the application locally with correct environment variables

export REDIS_HOST=localhost
export DB_HOST=localhost
export DB_PORT=7350
export OPENSEARCH_HOST=localhost

# Keep other variables from .env
source .env

poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
