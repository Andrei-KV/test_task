import multiprocessing
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

def run_bot():
    """Запускает Telegram-бота с обработкой ошибок."""
    logger.info("Starting the Telegram bot process...")
    try:
        print("🚀 Telegram bot is starting...")
        bot.polling(none_stop=True)
    except Exception as e:
        logger.critical(f"Bot process failed with a critical error: {e}", exc_info=True)
        print(f"❌ Critical error in bot process: {e}")

def run_listener():
    """Запускает слушателя Google Drive с обработкой ошибок."""
    logger.info("Starting the Google Drive listener process...")
    try:
        print("🔄 Starting the Google Drive listener...")
        process_new_documents()
        schedule.every(1).minutes.do(process_new_documents)
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.critical(f"Listener process failed with a critical error: {e}", exc_info=True)
        print(f"❌ Critical error in listener process: {e}")

if __name__ == "__main__":
    logger.info("Initializing application processes...")
    
    # Создаем процессы
    bot_process = multiprocessing.Process(target=run_bot)
    listener_process = multiprocessing.Process(target=run_listener)

    # Запускаем процессы
    bot_process.start()
    listener_process.start()

    logger.info("Both processes have been started.")

    # Ожидаем завершения процессов
    bot_process.join()
    listener_process.join()

    logger.info("Application has shut down.")
