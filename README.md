# RAG-приложение с веб-интерфейсом

Это RAG-приложение, которое использует FastAPI для предоставления веб-интерфейса и Docker для легкого развертывания.

## Запуск и тестирование

### 1. Настройка окружения

Перед запуском приложения необходимо настроить переменные окружения. Создайте файл `.env` в корне проекта и добавьте в него следующие переменные:

```
DB_URI=postgresql://user:password@db:5432/mydatabase
QDRANT_HOST=qdrant
REDIS_HOST=redis
DEEPSEEK_API_KEY=your_deepseek_api_key
SERVICE_ACCOUNT_FILE=path/to/your/service_account.json
TARGET_FOLDER_ID=your_google_drive_folder_id
COLLECTION_NAME=my_collection
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
```

### 2. Сборка и запуск

Для сборки и запуска приложения используйте скрипт `run_local.sh`:

```bash
./run_local.sh
```

Эта команда запустит все необходимые сервисы (приложение, Qdrant, PostgreSQL, Redis) в Docker-контейнерах.

### 3. Тестирование

После запуска откройте браузер и перейдите по адресу `http://localhost:8000`. Вы увидите веб-интерфейс чат-бота.

Вы можете отправлять запросы и получать ответы от RAG-модели. История диалога будет сохраняться, а ответы с низкой уверенностью будут вызывать уточняющие вопросы.

### 4. Обработка документов

Для обработки новых документов из Google Drive запустите скрипт `process_new_documents_test.py` в отдельном терминале:

```bash
docker-compose exec app python process_new_documents_test.py
```

Этот скрипт загрузит новые документы, обработает их и добавит в базу данных и Qdrant.
