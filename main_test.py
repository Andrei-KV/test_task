
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
    print("🚀 Telegram bot is starting...")
    bot.polling(none_stop=True)

    logger.info("Both processes have been started.")

