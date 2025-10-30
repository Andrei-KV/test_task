from sqlalchemy import create_engine, select, Integer, String, Text, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from datetime import datetime
from uuid import UUID, uuid4
import re
from dotenv import load_dotenv 
import os

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchParams
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
# Загрузка переменных окружения из файла .env
load_dotenv()
# -----------
# Подключение к БД и создание таблиц
# --------------------

DB_URI = os.getenv('DB_URI')
engine = create_engine(DB_URI)

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint('title', 'drive_file_id', name='uq_title_id'),
    )

    document_id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))

    # ID файла в Google Drive (уникальный ключ)
    drive_file_id: Mapped[str] = mapped_column(String(255), unique=True)
    # Публичная web-ссылка, сгенерированная Drive API
    web_link: Mapped[str] = mapped_column(String)

    load_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"))
    content: Mapped[str] = mapped_column(Text)
    qdrant_id: Mapped[str] = mapped_column(String(255))
    # Обратная связь
    document: Mapped["Document"] = relationship(back_populates="chunks")
    

# Применение изменений к базе данных
Base.metadata.create_all(engine)

# # ----------------------
# # Чтение файла
# # -------------------

# try:
#     with open('codex22.md', 'r', encoding='utf-8') as f:
#         raw_content = f.read()
# except FileNotFoundError:
#     print("Ошибка: Файл codex22.md не найден.")
#     raw_content = ""


# --------
# Инициализация и подключение к Google Disk
# ------


# --- НАСТРОЙКИ ---
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')# JSON-файл для доступа по API
# ID папки на диске
TARGET_FOLDER_ID = os.getenv('TARGET_FOLDER_ID')
# Scope (область) доступа: только чтение
SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] 
# -----------------

def init_drive_service(service_account_file: str):
    """Инициализирует сервис Google Drive с помощью JSON-ключа."""
    print("1. Инициализация сервисного аккаунта...")
    try:
        # Авторизация с помощью учетных данных сервисного аккаунта
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=SCOPES
        )
        # Создание объекта-клиента для взаимодействия с Drive API v3
        service = build('drive', 'v3', credentials=creds)
        print("✅ Инициализация успешна.")
        return service
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл ключа не найден по пути: {service_account_file}")
        return None
    except Exception as e:
        print(f"❌ Ошибка авторизации: Проверьте ключ и SCOPES. {e}")
        return None

def list_files_in_folder(service, folder_id: str):
    """Получает и выводит список файлов (не папок) в указанной папке."""
    
    print(f"\n2. Попытка чтения файлов в папке ID: {folder_id}...")
    
    # 1. Запрос (Query) для поиска:
    #   'ID_папки' in parents -> Ищем файлы, родитель которых — наша папка.
    #   and mimeType != 'application/vnd.google-apps.folder' -> Исключаем вложенные папки.
    #   and trashed = false -> Исключаем удаленные файлы.
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"

    try:
        # Выполняем запрос к API
        response = service.files().list(
            q=query,
            fields='files(id, name, mimeType)', # Выбираем только нужные поля
            pageSize=10 # Для теста достаточно первых 10 файлов
        ).execute()

        files = response.get('files', [])
        
        if files:
            print(f"✅ Успех! Найдено {len(files)} файлов (или больше).")
            print("\n--- Список первых 10 файлов ---")
            
            # Вывод данных в таблице для удобства
            file_data_list = []
            for f in files:
                file_data_list.append({
                    "Имя Файла": f.get('name', 'N/A'),
                    "ID Файла": f.get('id', 'N/A'),
                    "MIME Тип": f.get('mimeType', 'N/A')
                })
            print('Полученный список документов: ', file_data_list)   

            return file_data_list
            
        else:
            print("⚠️ Успешное подключение, но файлы не найдены. Проверьте:")
            
    except Exception as e:
        print(f"Ошибка при запросе списка файлов: {e}")
  
# Инициализация
drive_service = init_drive_service(SERVICE_ACCOUNT_FILE)
    
if drive_service:
    files_in_folder = list_files_in_folder(drive_service, TARGET_FOLDER_ID)
    
    #TEST
    file_info = files_in_folder[0] # type: ignore
else:
    print('FLAG: FAIL')
    file_info = []

#TEST
# Генерируем публичную ссылку для сохранения (предполагая, что она уже доступна)
# В реальном коде эту ссылку лучше генерировать в отдельном шаге,
# но для ETL-сохранения мы ее временно сгенерируем здесь.
# Здесь мы используем стандартный шаблон ссылки, но в продакшене лучше
# использовать API, как описано ранее (service.files().get()).
file_id = file_info["ID Файла"]
file_name = file_info["Имя Файла"] 
file_mime_type = file_info["MIME Тип"] 

# ----------------------
# Чтение файла (Скачивание из Google Drive)
# ----------------------

def download_drive_file_content(drive_service, file_id: str, file_name: str) -> str | None:
    try:
        request = drive_service.files().get_media(fileId=file_id)

        # Выполнение запроса и получение контента
        content = request.execute()
        
        # Если скачиваем PDF/DOCX, content будет в виде байтов.
        # сохраним байты во временный файл для локального парсера.
        
        return content.decode('utf-8')

    except Exception as e:
        print(f"❌ Ошибка при скачивании файла {file_name} ({file_id}): {e}")
        return None

raw_content_bytes = download_drive_file_content(
    drive_service, file_id, file_name
)

if raw_content_bytes is None:
    print(f"Ошибка: Не удалось получить содержимое файла {file_name}.")
    exit() # Прерываем процесс для этого файла

# Предполагая, что download_drive_file_content вернул байты (например, PDF/DOCX):
# Далее необходимо вызвать ваш парсер, который преобразует байты в чистый текст.
# Например, для PDF: text_content = pdf_parser(raw_content_bytes)

# Для упрощения примера, допустим, мы получили готовый текст:
text_content = f"Содержимое файла {file_name} с ID {file_id}. [Это начало текста для чанков.]"

# ----------------------
# Получение Корректной Web-ссылки на документ
# ----------------------
def get_drive_web_link(service, file_id: str) -> str | None:
    """Возвращает webViewLink (ссылку для просмотра) на файл Google Drive по его ID."""
    try:
        # Запрос только поля webViewLink
        file_metadata = service.files().get(
            fileId=file_id, 
            fields='webViewLink'
        ).execute()
        
        # Drive API возвращает 'webViewLink'
        return file_metadata.get('webViewLink')
        
    except Exception as e:
        print(f"❌ Ошибка получения ссылки для файла ID {file_id}: {e}")
        return None
web_link = get_drive_web_link(drive_service, file_id)

if not web_link:
    print(f"⚠️ Не удалось получить web_link для {file_name}. Пропускаем документ.")

# ---------------------
# Сохранение данных файла в таблицу
# ------------------------

with Session(engine) as session:

    # Проверяем, существует ли уже документ с таким drive_file_id
    existing_doc = session.execute(
        select(Document).where(Document.drive_file_id == file_id)
    ).scalar_one_or_none()

    if existing_doc:
        print(f"Документ уже существует с ID = {existing_doc.document_id}")
        document_id = existing_doc.document_id
    else:
        # Если нет — добавляем новый
        new_document = Document(
            #НОВЫЕ ПОЛЯ
            title=file_name,
            drive_file_id=file_id,
            web_link=web_link,
        )
        session.add(new_document)
        session.commit()
        session.refresh(new_document) # получаем document_id после commit
        document_id = new_document.document_id
        print(f"✅ Документ {file_name} сохранён с ID = {document_id}")

#---------------------------------
# Очистка (!!Очистку для разных типов файлов)
# -----------------------------------
# 1. Удаление HTML/Markdown артефактов
# Паттерн 1: Удаляет HTML-теги (<...> и </...>)
cleaned_content = re.sub(r'<[^>]+>', ' ', raw_content_bytes)

# Паттерн 2: Удаляет HTML-сущности (например, &nbsp;, &lt;, &#x27;)
cleaned_content = re.sub(r'&[a-z]+;|&#x?[0-9a-f]+;', ' ', cleaned_content)

# 2. Нормализация пробелов
# Паттерн: Заменяет любые последовательности пробельных символов (пробел, \n, \t) на один пробел.
# .strip() удаляет пробел с начала и конца
cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()

print(f"Исходная длина: {len(raw_content_bytes)} символов")
print(f"Очищенная длина: {len(cleaned_content)} символов")
print(cleaned_content[:500]) # Вывод первых 500 символов для проверки


# -----------------------------
# Разбиение на chunks
# -----------------------------

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # приоритет разбиения
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=False
    )

    chunks = splitter.split_text(text)
    return chunks


chunks = split_text_into_chunks(cleaned_content, chunk_size=500, overlap=50)

# ------------------------------
# Добавление чанков в БД
# --------------------------

with Session(engine) as session:
    # Проверяем, существуют ли чанки документа
    existing_chunks_count = session.query(DocumentChunk).filter_by(document_id=document_id).count()
    
    if existing_chunks_count > 0:
        print(f"⚠️ Чанки для документа {document_id} уже существуют. Пропускаю добавление.")
    
    else:
        for chunk in chunks:
            # Генерация UUID для Qdrant
            qdrant_uuid = str(uuid4())
            new_document_chunk = DocumentChunk(
                document_id=document_id,
                content=chunk,
                qdrant_id=qdrant_uuid,
            )
            session.add(new_document_chunk)
        session.commit()

# ---------------------------
# Embedding
# ----------------

# Настройка Qdrant
QDRANT_HOST = os.getenv('QDRANT_HOST')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
qdrant_client = QdrantClient(url=QDRANT_HOST)


EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# Получаем размерность вектора (например, 768)
# VECTOR_DIMENSION = embedding_model.get_sentence_embedding_dimension()
VECTOR_DIMENSION = 768
print(f"Используемая модель: {EMBEDDING_MODEL_NAME}")
print(f"Размерность векторов: {VECTOR_DIMENSION}")

# Создание Коллекции в Qdrant
try:
    check_collection = qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
    if not check_collection:
        qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE), # type: ignore
    )
        print(f"✅ Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"✅ Коллекция '{COLLECTION_NAME}' уже существует.")
except Exception as e:
    print(f"❌ Ошибка Qdrant: {e}. Убедитесь, что Qdrant запущен.")


# ----------------------------------
# Извлечение Чанков для Векторизации и для создания поинтов
# ----------------------------------

with Session(engine) as session:
    # Используем document_id, полученный на этапе сохранения Document
    chunk_objects_to_process = session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    ).scalars().all() 

if not chunk_objects_to_process:
    print("❌ В PostgreSQL не найдены чанки для обработки. Проверьте, что document_id верен.")
    # Здесь можно добавить логику завершения, если данных нет
    # exit()

# Подготовка данных для пакетной обработки
texts_to_embed = [c.content for c in chunk_objects_to_process]
qdrant_ids = [c.qdrant_id for c in chunk_objects_to_process]

print(f"✅ Извлечено {len(texts_to_embed)} чанков для векторизации.")

# Теперь texts_to_embed содержит тексты, а qdrant_ids — соответствующие им ID.


# ----------------------------------
# Пакетная Векторизация
# ----------------------------------

# texts_to_embed уже содержит список строк: [chunk1_text, chunk2_text, ...]

print(f"Запуск пакетной векторизации {len(texts_to_embed)} чанков...")
# TEST tut
embeddings = embedding_model.encode(
    texts_to_embed,
    # show_progress_bar отображает индикатор выполнения в консоли
    show_progress_bar=True,
    # для нормализации векторов
    normalize_embeddings=True,
    # tolist() преобразует выходной массив NumPy в стандартный список Python,
    # который удобнее для передачи в Qdrant Client
    convert_to_tensor=False 
).tolist()

# для тестов запись векторов в файл
# import json
# EMBEDDINGS_FILE = "embeddings.json"
# with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
#     # Функция json.dump записывает объект Python в файл в формате JSON
#     json.dump(embeddings, f, indent=4) # indent=4 делает файл легко читаемым человеком


print(f"✅ Векторизация завершена. Получено {len(embeddings)} векторов.")





# # для тестов чтение векторов из файла
# #-------------------------------------->
# import json
# import os

# EMBEDDINGS_FILE = "embeddings.json"


# with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
#     loaded_embeddings = json.load(f)
      
# embeddings = loaded_embeddings


#----------------------------->

# TEST tut
# ----------------------------------
# Создание списка PointStruct
# ----------------------------------
points = []
# Итерируемся по вектору и его индексу
for idx, vector in enumerate(embeddings):
    # Получаем объект чанка из нашего списка для доступа к его ID и контенту
    chunk_obj = chunk_objects_to_process[idx]
    
    # 1. Генерация int ID для Qdrant: 
    # Преобразуем часть строкового UUID в большое целое число
    qdrant_id_int = int(chunk_obj.qdrant_id.replace('-', '')[:15], 16) 
    
    # 2. Формирование Payload (Метаданные)
    payload = {
        # ID для поиска в таблице document_chunks
        "chunk_id": chunk_obj.chunk_id,      
        # ID для поиска в таблице documents
        "document_id": chunk_obj.document_id, 
        # Добавляем превью для быстрой отладки
        "content_preview": chunk_obj.content[:100] + "..." 
    }
    
    # 3. Создание PointStruct
    point = PointStruct(
        id=qdrant_id_int, 
        vector=vector,
        payload=payload
    )
    points.append(point)

print(f"✅ Создано {len(points)} объектов PointStruct для загрузки.")

# ----------------------------------
# Массовая Загрузка points (Upsert) в Qdrant
# ----------------------------------

try:
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True, # Ждем подтверждения от сервера, что все векторы записаны
        points=points
    )
    print(f"✅ {len(points)} векторов успешно загружены в Qdrant (Коллекция: {COLLECTION_NAME}).")

except Exception as e:
    print(f"❌ Ошибка при загрузке в Qdrant: {e}")
    # Здесь нужно добавить логику повторной попытки или логирования ошибки


# --------------------
# Функция векторизации запроса
# ------------------------
def vectorize_query(query: str, model: SentenceTransformer) -> list[float]:
    """Векторизует один текстовый запрос."""
    # Метод encode принимает список, поэтому оборачиваем запрос в список
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True, 
        convert_to_tensor=False
    ).tolist()[0]
    
    # Нормализация: Модель Sentence Transformers делает это по умолчанию, 
    # но важно убедиться, что и запрос, и чанки нормализованы.
    return query_embedding

# -----------------
# Функция семантического поиска
# ------------------

# Принимает запрос в формате вектора и возврщает ScoredPoint
def semantic_search(query_vector: list[float], limit_k: int = 5):
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit_k,
        with_payload=True,
        search_params=SearchParams(
            exact=False, # Используем быстрый ANN поиск
            hnsw_ef=100
        )
    ).points
    return search_result


# ---------
# Извлечение полного контекста 
#----------

def retrieve_full_context(qdrant_results, session: Session) -> tuple:
    """Использует ID из Qdrant для извлечения полного текста из PostgreSQL."""

    # Определение ID Наиболее Релевантного Документа
    try:
        # Получаем document_id из метаданных самого релевантного результата
        top_document_id = qdrant_results[0].payload.get('document_id')
        print('top_document_id', top_document_id)
    except (IndexError, KeyError):
        # Если метаданные не содержат document_id, возвращаем пустой результат
        print("❌ Ошибка: Отсутствует 'document_id'.")
        return " ", None

    # ID ТОЛЬКО самых релевантных чанков 
    relevant_chunk_ids = []

    for result in qdrant_results:
        print('result.payload.get(document_id)', result.payload.get('document_id'))
        # Убеждаемся, что чанк принадлежит целевому документу
        # Это исключает релевантные чанки из менее релевантных документов
        if result.payload.get('document_id') == top_document_id:
            relevant_chunk_ids.append(result.payload.get('chunk_id'))
    print('qdrant_results ', qdrant_results)
    if not relevant_chunk_ids:
        print('not relevant_chunk_ids')
        return " ", None

    # Извлечение Только Релевантных Чанков 
    stmt = (
        select(DocumentChunk.content, Document.web_link)
        .join(Document)
        .where(
            # Условие: Чанк должен быть одним из тех, что вернул Qdrant (limit_k)
            DocumentChunk.chunk_id.in_(relevant_chunk_ids) 
        )
        .order_by(DocumentChunk.chunk_id) 
    )
    
    sql_results = session.execute(stmt).fetchall()

    if not sql_results:
        print('not sql_results')
        return " ", None

    full_context = []
    # Ссылка будет одинаковой для всех чанков этого документа
    web_link = sql_results[0].web_link 
    
    # Собираем весь контент
    for result in sql_results:
        full_context.append(result.content)
    print('Полный контекст.: ', len(full_context))
    # Объединяем чанки в один контекст
    context = "\n\n".join(full_context)
    return context, web_link

# --- Использование ---
user_query = 'Что такое налоговая база'

query_vector  = vectorize_query(user_query, embedding_model)
qdrant_results = semantic_search(query_vector)

with Session(engine) as session:
    context, web_link = retrieve_full_context(qdrant_results, session)
    
print(context, web_link)
# # TEST
# context =  """
# а также на основании имеющихся в налоговом органе сведений

# . Налоговая база и налоговая ставка (ставки). Порядок определения и корректировки налоговой базы и (или) суммы налога (сбора) 1. Налоговая база представляет собой стоимостную, физическую или иную характеристику объекта налогообложения. Налоговая база определяется применительно к конкретным налогу, сбору (пошлине). 2

# . 2. Налоговая ставка представляет собой величину налоговых начислений на единицу измерения налоговой базы, если иное не установлено настоящим Кодексом, иными актами налогового или таможенного законодательства. Налоговые ставки устанавливаются применительно к каждому налогу, сбору (пошлине). 3

# . РАЗДЕЛ II НАЛОГОВОЕ ОБЯЗАТЕЛЬСТВО ГЛАВА 4 НАЛОГОВЫЙ УЧЕТ. НАЛОГОВАЯ ДЕКЛАРАЦИЯ. НАЛОГОВОЕ ОБЯЗАТЕЛЬСТВО И ЕГО ИСПОЛНЕНИЕ Статья 39. Налоговый учет 1. Налоговым учетом признается осуществление плательщиками учета объектов налогообложения и определения налоговой базы по налогам, сборам (пошлинам) путем расчетных корректировок к данным бухгалтерского учета, если иное не установлено налоговым законодательством

# .3. обоснованность применения плательщиком налоговых ставок и налоговых льгот; 7.4 """
 # --------
# Генерация ответа пользователю
# --------

# --- Инструкции для LLM ---
SYSTEM_INSTRUCTIONS = (
    "Вы — юридический ассистент. Ваша задача — синтезировать точный, "
    "понятный и связный ответ на вопрос пользователя, используя ТОЛЬКО "
    "предоставленный ниже контекст. Если контекст не содержит информации, "
    "достаточной для ответа, вы должны ответить: 'Извините, в предоставленных "
    "документах точный ответ не найден.' Сохраняйте профессиональный тон и "
    "отвечайте на русском языке."
)
# --- Формирование финального Prompt ---
prompt = f"""
{SYSTEM_INSTRUCTIONS}

--- КОНТЕКСТ ДЛЯ АНАЛИЗА ---
{context}
---

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{user_query}
"""

print("✅ Prompt для LLM успешно сгенерирован.")
print("--- Фрагмент Prompt ---")
print(prompt[:500] + "...")




from openai import OpenAI
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
LLM_MODEL = "deepseek-chat"

def generate_rag_response(context: str, user_query: str, client: OpenAI, model_name: str = LLM_MODEL) -> str:
    """Отправляет prompt с контекстом в Gemini и возвращает ответ."""
    
    
    SYSTEM_INSTRUCTIONS = (
        "Вы — юридический ассистент. Ваша задача — синтезировать точный, "
        "понятный и связный ответ на вопрос пользователя, используя ТОЛЬКО "
        "предоставленный ниже контекст. Если контекст не содержит информации, "
        "достаточной для ответа, вы должны ответить: 'Извините, в предоставленных "
        "документах точный ответ не найден.' Сохраняйте профессиональный тон и "
        "отвечайте на русском языке."
    )
    
    
    print(f"\n Отправка запроса в {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"""{SYSTEM_INSTRUCTIONS}--- КОНТЕКСТ ---{context}"""},
                {"role": "user", "content": user_query},
        ],
            stream=False
        )
        # Возвращаем только текстовую часть ответа
        return response.choices[0].message.content # type: ignore

    except Exception as e:
        return f"❌ Произошла непредвиденная ошибка при генерации: {e}"

# --- Выполнение ---
final_answer = generate_rag_response(context, user_query, client)

print("\n--- ФИНАЛЬНЫЙ ОТВЕТ LLM ---")
print(final_answer, '\n', f'Ссылка на документ: {web_link}')


