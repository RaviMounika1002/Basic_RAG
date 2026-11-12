cat > 02-docmind-cloud-rag/README.md <<'EOF'
# DocMind - Cloud PDF RAG

Project 02 of my AI Engineering Learning Series.

## Features
- PDF upload and text extraction
- Page-aware document chunking
- OpenAI cloud embeddings
- Supabase pgvector similarity search
- Grounded answers with page citations
- Streamlit chat interface

## Architecture
PDF -> PyMuPDF -> Chunks -> OpenAI Embeddings
    -> Supabase pgvector

Question -> OpenAI Embedding -> Vector Search
    -> Retrieved Context -> OpenAI LLM -> Answer

## Stack
Python, Streamlit, PyMuPDF, OpenAI, Supabase

## Run
1. Install requirements.txt
2. Configure Streamlit secrets
3. Run: streamlit run app.py

## Learning Outcomes
- Cloud embedding APIs
- Persistent vector databases
- SQL vector similarity search
- Metadata-aware retrieval
- RAG answer generation and citations

## Limitations
- Text-based PDFs only
- One active PDF per browser session
- No authentication or user-level data isolation
- No automatic cleanup of abandoned uploads
- Not suitable for public multi-user deployment yet
EOF