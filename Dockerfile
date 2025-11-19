# Этап 1: Сборка зависимостей
FROM python:3.12-slim as builder

# Установка Poetry
RUN pip install poetry

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Установка зависимостей с помощью Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --with pytorch-cpu

# Этап 2: Создание конечного образа
FROM python:3.12-slim

# Установка системных зависимостей, необходимых для работы приложения
RUN apt-get update && apt-get install -y \
    tesseract-ocr-rus \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Копирование установленных зависимостей из этапа сборки
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование исходного кода приложения
COPY src ./src
COPY main.py .

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
