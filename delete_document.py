import asyncio
import argparse
import logging
from src.services.document_deleter import delete_document

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    Парсит аргументы командной строки и запускает процесс удаления документа.
    """
    parser = argparse.ArgumentParser(description="Удалить документ и все связанные с ним данные.")
    parser.add_argument(
        "document_id",
        type=int,
        help="ID документа, который необходимо удалить."
    )
    
    args = parser.parse_args()
    
    document_id_to_delete = args.document_id
    
    logger.info(f"Запрос на удаление документа с ID: {document_id_to_delete}")
    
    # Запрос подтверждения от пользователя
    confirm = input(
        f"Вы уверены, что хотите навсегда удалить документ с ID {document_id_to_delete} "
        f"и все связанные с ним данные из PostgreSQL и Qdrant? (yes/no): "
    )
    if confirm.lower() != 'yes':
        logger.info("Операция отменена пользователем.")
        return
        
    await delete_document(document_id_to_delete)
    logger.info("Процесс удаления завершен.")

if __name__ == "__main__":
    asyncio.run(main())

'''
Чтобы запустить скрипт delete_document.py, вам нужно выполнить команду в терминале, указав ID документа, который вы хотите удалить.

Шаг 1: Как найти ID нужного документа
Сначала вам нужно узнать document_id документа, который вы хотите удалить. Самый простой способ — подключиться к вашей базе данных PostgreSQL и посмотреть список всех документов.

Подключитесь к контейнеру PostgreSQL:

docker exec -it legal_rag_postgres psql -U your_username -d your_database_name
Замените your_username и your_database_name на ваши реальные значения из .env файла (например, DB_USER и DB_NAME).
Выполните SQL-запрос для просмотра документов: Внутри psql введите следующую команду, чтобы увидеть список всех документов и их ID:

SELECT document_id, title FROM documents;
Вывод будет выглядеть примерно так:

 document_id |           title
-------------+---------------------------
           1 | Инструкция по безопасности.docx
           2 | Технический регламент.pdf
           3 | Стандарт предприятия.docx
Найдите в этом списке нужный документ и запомните его document_id.

Шаг 2: Как запустить скрипт удаления
Теперь, когда у вас есть ID, вы можете запустить скрипт.

Откройте терминал в корневой папке вашего проекта.

Выполните команду, заменив <ID_ДОКУМЕНТА> на реальный номер из предыдущего шага:

python delete_document.py <ID_ДОКУМЕНТА>
Например, чтобы удалить документ с ID 2:

python delete_document.py 2
Подтвердите удаление: Скрипт попросит вас подтвердить действие, чтобы избежать случайного удаления. Вам нужно будет напечатать yes и нажать Enter.

Вы уверены, что хотите навсегда удалить документ с ID 2 и все связанные с ним данные из PostgreSQL и Qdrant? (yes/no): yes
После этого скрипт выполнит полное удаление документа из обеих баз данных.
'''