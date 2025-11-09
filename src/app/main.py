
# import schedule
import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(title="RAG Chatbot")

# Настройка CORS Cross-Origin Resource Sharing
origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Обслуживание статических файлов
app.mount("/static", StaticFiles(directory="src/static"), name="static")

if __name__ == "__main__":
    logger.info("Initializing application processes...")

    while True:
        try:
            print("🚀 Telegram bot is starting...")
            # 1. Запуск блокирующего процесса
            bot.polling(none_stop=True, timeout=80) 
            
        # ⚠️ ПЕРВОЕ ИСКЛЮЧЕНИЕ: Ловим KeyboardInterrupt (Ctrl+C)
        except KeyboardInterrupt:
            logger.info("✅ Завершение работы по команде пользователя (Ctrl+C)...")
            break  # Выход из цикла while True для корректного завершения программы
            
        # ⚠️ ВТОРОЕ ИСКЛЮЧЕНИЕ: Ловим ВСЕ остальные ошибки, требующие перезапуска
        except Exception as e:
            logger.error(f"❌ Критическая ошибка Polling: {e}. Перезапуск через 10 секунд...")
            
            # Ждем перед попыткой перезапуска
            time.sleep(10) 
            
    logger.info("Приложение Telegram бота успешно остановлено.")