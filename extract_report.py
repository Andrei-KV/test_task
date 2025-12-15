
import os
import html
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Configuration
DB_URI = os.getenv("DB_URI")
OUTPUT_FILE = "/app/database_report_extracted.html"

async def generate_report():
    if not DB_URI:
        print("❌ Error: DB_URI environment variable not defined")
        return

    print(f"Connecting to database (Async)...")
    try:
        # DB_URI is likely 'postgresql+asyncpg://...' which works with create_async_engine
        engine = create_async_engine(DB_URI)
        chunks = []
        
        async with engine.connect() as conn:
            query = text("""
                SELECT 
                    document_id, 
                    document_title, 
                    chunk_id, 
                    content, 
                    page_number, 
                    chunk_index, 
                    content_type 
                FROM document_chunks 
                ORDER BY document_id, chunk_index
            """)
            result = await conn.execute(query)
            rows = result.fetchall()
            
            for row in rows:
                chunks.append({
                    'document_id': row.document_id,
                    'document_title': row.document_title,
                    'chunk_id': row.chunk_id,
                    'content': row.content,
                    'page_number': row.page_number,
                    'chunk_index': row.chunk_index,
                    'content_type': row.content_type
                })
        
        await engine.dispose()
                
        print(f"✅ Extracted {len(chunks)} chunks")
        
        # Group by document
        docs = {}
        for chunk in chunks:
            doc_id = int(chunk['document_id'])
            if doc_id not in docs:
                docs[doc_id] = {
                    'title': chunk['document_title'],
                    'chunks': []
                }
            docs[doc_id]['chunks'].append(chunk)

        # Generate HTML
        html_content = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extracted Database Report</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f2f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #1a1a1a; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; }
        .document { margin-bottom: 30px; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }
        .doc-header { background: #e9ecef; padding: 15px; font-weight: bold; border-bottom: 1px solid #ddd; }
        .chunk { padding: 15px; border-bottom: 1px solid #eee; }
        .chunk:last-child { border-bottom: none; }
        .chunk:hover { background: #fcfcfc; }
        .meta { font-size: 0.85em; color: #666; margin-bottom: 8px; display: flex; gap: 15px; }
        .content { white-space: pre-wrap; font-family: monospace; font-size: 0.9em; background: #fafafa; padding: 10px; border-radius: 4px; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
        .badge-text { background: #e3f2fd; color: #0d47a1; }
        .badge-ocr { background: #f3e5f5; color: #4a148c; }
        .badge-table { background: #e8f5e9; color: #1b5e20; }
        .badge-image { background: #fff3e0; color: #e65100; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📑 Extracted Database Content</h1>
        
        <div class="stats">
            <div><strong>Documents:</strong> ''' + str(len(docs)) + '''</div>
            <div><strong>Total Chunks:</strong> ''' + str(len(chunks)) + '''</div>
        </div>

'''
        
        for doc_id in sorted(docs.keys()):
            doc = docs[doc_id]
            html_content += f'''
        <div class="document">
            <div class="doc-header">
                📄 {html.escape(doc['title'])} (ID: {doc_id}) - {len(doc['chunks'])} chunks
            </div>
'''
            for chunk in doc['chunks']:
                ctype = chunk['content_type']
                badge_class = f"badge-{ctype}" if ctype in ['text', 'ocr', 'table', 'image'] else "badge-text"
                
                html_content += f'''
            <div class="chunk">
                <div class="meta">
                    <span>Page: <strong>{chunk['page_number']}</strong></span>
                    <span>Index: <strong>{chunk['chunk_index']}</strong></span>
                    <span class="badge {badge_class}">{ctype}</span>
                    <span style="margin-left: auto">ID: {chunk['chunk_id']}</span>
                </div>
                <div class="content">{html.escape(chunk.get('content') or '')}</div>
            </div>
'''
            html_content += '        </div>'

        html_content += '''
    </div>
</body>
</html>'''

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(generate_report())
