
import schedule
import time
import logging
from telegram_bot.bot import bot  # Явный импорт
from google_drive_listener import process_new_documents

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing application processes...")

    # process_new_documents()
 
    while True:
        try:
            print("🚀 Telegram bot is starting...")
            bot.polling(none_stop=True, timeout=80) 
            
        except Exception as e:
            # Ловит общие ошибки, включая ReadTimeoutError, TimeoutError, 
            # ConnectionResetError, и даже ошибки авторизации/сети.
            logger.error(f"❌ Критическая ошибка Polling: {e}. Перезапуск через 10 секунд...")
            
            # Ждем перед попыткой перезапуска, чтобы избежать DDOS на Telegram
            time.sleep(10) 
            # Цикл while True гарантирует, что код вернется к try и попробует polling снова.