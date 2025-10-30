import multiprocessing
from telegram_bot import bot
from google_drive_listener import process_new_documents
import schedule
import time

def run_bot():
    """Runs the Telegram bot."""
    print("🚀 Starting the Telegram bot...")
    bot.polling(none_stop=True)


def run_listener():
    """Runs the Google Drive listener."""
    print("🚀 Starting the Google Drive listener...")
    process_new_documents()
    schedule.every(1).minutes.do(process_new_documents)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_bot)
    
    p2 = multiprocessing.Process(target=run_listener)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
