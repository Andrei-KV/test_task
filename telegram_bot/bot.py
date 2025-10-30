import telebot
import logging
from threading import Thread
from config import TELEGRAM_TOKEN

from services import rag_pipeline

# Variables check
if TELEGRAM_TOKEN is None:
    raise ValueError("Переменные не найдены. Проверьте файл .env.")
# --- Инициализация бота ---

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- ОБРАБОТЧИК КОМАНДЫ /start ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Отправляет приветственное сообщение и предлагает задать вопрос."""
    logger.info(f"Received /start command from user {message.from_user.id}")
    bot.reply_to(
        message, 
        f"👋 Привет, {message.from_user.first_name}!\n"
        "Я — ваш RAG-ассистент. Задайте мне любой вопрос по документам, "
        "и я отвечу и дам ссылку на источник."
    )

# --- ОСНОВНОЙ ОБРАБОТЧИК ТЕКСТОВЫХ СООБЩЕНИЙ ---

@bot.message_handler(func=lambda message: True)
def handle_user_query_thread(message):
    """Запускает RAG-конвейер в отдельном потоке для предотвращения зависания бота."""
    logger.info(f"Received a text message from user {message.from_user.id}")
    # Запускаем сложную логику в отдельном потоке
    Thread(target=process_rag_request, args=(message,)).start()

def process_rag_request(message):
    """Основная логика RAG-конвейера."""
    user_query = message.text
    chat_id = message.chat.id

    # 1. Отправка уведомления о начале обработки
    logger.info(f"Processing query for user {message.from_user.id}: '{user_query}'")
    processing_message = bot.send_message(chat_id, "⏳ Ваш вопрос в обработке...")
    message_id = processing_message.message_id
    
    try:
        final_answer, web_link = rag_pipeline(user_query)
        response_text = f"""
{final_answer}

---
🔗 **Исходный документ:** {web_link if web_link else 'Ссылка на документ не найдена.'}
"""
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=response_text
        )
        logger.info(f"Successfully sent response to user {message.from_user.id}")

    except Exception as e:
        logger.error(f"Произошла ошибка RAG-конвейера: {e}")
        # Отправка сообщения об ошибке пользователю
        bot.edit_message_text(
            chat_id=chat_id, 
            message_id=message_id, 
            text=f"Произошла непредвиденная ошибка при обработке запроса: {e}"
        )


if __name__ == '__main__':
    logger.info("Bot is starting up...")
    print("🚀 Бот запущен. Нажмите Ctrl+C для остановки.")
    try:
        # Бот начинает опрос сервера Telegram
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
