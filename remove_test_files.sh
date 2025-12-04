#!/bin/bash
# Скрипт для удаления тестовых файлов из Git (но оставляет их локально)

echo "Удаление тестовых файлов из Git репозитория..."
echo "Файлы останутся на диске, но будут игнорироваться Git"
echo ""

# Удаляем тестовые файлы из Git
git rm --cached \
  delete_document.py \
  test_chunking_simple.py \
  test_chunking_two_docs.py \
  test_chunks_simple.txt \
  test_opensearch_connection.py \
  test_parse_pages_3_4.py \
  test_process_new_documents_test.py \
  test_rag_service_embeddings.py \
  verify_embeddings.py \
  verify_ingestion.py \
  check_db.py \
  verify_optimization.py \
  test_garbage_detection.py \
  test_paddle_minimal.py \
  chunks_verification.txt \
  chunks_verification_new.txt \
  test_output.txt \
  pages_test.txt \
  implementation_plan.md \
  image.png \
  debug_ocr_script.py \
  google_drive_listener.py \
  update_database.py \
  index_to_opensearch.py \
  view_chunks.py \
  2>/dev/null || true

# Удаляем документацию
git rm --cached \
  APP_ANALYSIS.md \
  CPU_OPTIMIZATION.md \
  DEPENDENCY_ANALYSIS.md \
  GARBAGE_DETECTION_TEST_RESULTS.md \
  HOW_IT_WORKS.md \
  INGESTION_ANALYSIS.md \
  LOGGING_GUIDE.md \
  OCR_README.md \
  OPENSEARCH_SETUP.md \
  2>/dev/null || true

# Удаляем директории
git rm -r --cached ocr_tests/ 2>/dev/null || true
git rm -r --cached ocr_results/ 2>/dev/null || true
git rm -r --cached results/ 2>/dev/null || true

echo ""
echo "✅ Готово! Файлы удалены из Git, но остались локально"
echo ""
echo "Следующие шаги:"
echo "1. Проверьте изменения: git status"
echo "2. Закоммитьте: git commit -m 'Remove test files from tracking'"
echo "3. Запушьте: git push"
