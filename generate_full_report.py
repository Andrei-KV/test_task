#!/usr/bin/env python3
import csv
import html

# Read TSV data
chunks = []
with open('/tmp/chunks_export.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['document_id']:  # Skip empty rows
            chunks.append(row)

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
    <title>Database Full Report - Legal RAG</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .document-section {
            margin: 20px;
            border: 2px solid #667eea;
            border-radius: 12px;
            overflow: hidden;
            background: white;
        }
        
        .document-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .chunk {
            border-bottom: 1px solid #eee;
            padding: 20px;
            transition: background 0.2s ease;
        }
        
        .chunk:hover {
            background: #f8f9fa;
        }
        
        .chunk:last-child {
            border-bottom: none;
        }
        
        .chunk-header {
            display: grid;
            grid-template-columns: auto auto auto 1fr;
            gap: 15px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        
        .chunk-meta {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .meta-label {
            font-weight: bold;
            color: #666;
            font-size: 0.85em;
        }
        
        .meta-value {
            color: #667eea;
            font-weight: 600;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .badge-mixed {
            background: #fff3e0;
            color: #f57c00;
        }
        
        .badge-text {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .badge-ocr {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .chunk-content {
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: none;
            overflow: visible;
        }
        
        .chunk-id {
            font-size: 0.75em;
            color: #999;
            font-family: monospace;
        }
        
        footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }
        
        .toc {
            background: #f8f9fa;
            padding: 20px;
            margin: 20px;
            border-radius: 12px;
        }
        
        .toc h2 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .toc-item {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .toc-item:hover {
            background: #667eea;
            color: white;
            transform: translateX(5px);
        }
        
        .toc-item a {
            text-decoration: none;
            color: inherit;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Full Database Report</h1>
            <div class="subtitle">Legal RAG System - Complete Chunks Content</div>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">''' + str(len(docs)) + '''</div>
                <div class="stat-label">Документов</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">''' + str(len(chunks)) + '''</div>
                <div class="stat-label">Чанков</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">''' + str(round(len(chunks) / len(docs), 1)) + '''</div>
                <div class="stat-label">Чанков/документ</div>
            </div>
        </div>
        
        <div class="toc">
            <h2>📑 Содержание</h2>
'''

for doc_id in sorted(docs.keys()):
    doc = docs[doc_id]
    html_content += f'''            <div class="toc-item">
                <a href="#doc-{doc_id}">Документ {doc_id}: {html.escape(doc['title'])} ({len(doc['chunks'])} чанков)</a>
            </div>
'''

html_content += '''        </div>
'''

# Add documents and chunks
for doc_id in sorted(docs.keys()):
    doc = docs[doc_id]
    html_content += f'''
        <div class="document-section" id="doc-{doc_id}">
            <div class="document-header">
                📄 Документ {doc_id}: {html.escape(doc['title'])}
                <div style="font-size: 0.8em; margin-top: 5px; opacity: 0.9;">
                    Всего чанков: {len(doc['chunks'])}
                </div>
            </div>
'''
    
    for chunk in doc['chunks']:
        content_type_class = f"badge-{chunk['content_type']}" if chunk['content_type'] in ['text', 'mixed', 'ocr'] else 'badge-mixed'
        
        html_content += f'''
            <div class="chunk">
                <div class="chunk-header">
                    <div class="chunk-meta">
                        <span class="meta-label">Страница:</span>
                        <span class="meta-value">{chunk['page_number']}</span>
                    </div>
                    <div class="chunk-meta">
                        <span class="meta-label">Индекс:</span>
                        <span class="meta-value">{chunk['chunk_index']}</span>
                    </div>
                    <div class="chunk-meta">
                        <span class="meta-label">Тип:</span>
                        <span class="badge {content_type_class}">{chunk['content_type']}</span>
                    </div>
                    <div class="chunk-id">
                        ID: {chunk['chunk_id']}
                    </div>
                </div>
                <div class="chunk-content">{html.escape(chunk['content'])}</div>
            </div>
'''
    
    html_content += '''        </div>
'''

html_content += '''
        <footer>
            <p>Generated: 2025-12-11 11:05 | Database: legal_rag_db | Total Chunks: ''' + str(len(chunks)) + '''</p>
            <p>✅ Full content export with all metadata fields</p>
        </footer>
    </div>
</body>
</html>
'''

# Write HTML file
with open('/home/dron/vs_code/test_rag/test_task/database_report_full.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Generated report with {len(chunks)} chunks from {len(docs)} documents")
