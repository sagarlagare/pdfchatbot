Chat with PDF

A lightweight RAG-based PDF Question Answering application built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Groq LLMs.

This application allows users to upload one or more PDF files, process them into embeddings, and ask questions grounded in the uploaded documents.

Features
Upload multiple PDF files
Extract and process document text
Semantic search using vector embeddings
Context-based question answering
Simple Streamlit interface

Tech Stack
Streamlit — UI for file upload and interaction
pypdf — PDF text extraction
LangChain — RAG pipeline orchestration
RecursiveCharacterTextSplitter — text chunking
HuggingFaceEmbeddings (all-MiniLM-L6-v2) — embedding generation
FAISS — vector storage and similarity search
Groq + llama-3.3-70b-versatile — answer generation
python-dotenv — environment variable management

How It Works
Upload PDF files
Extract text from PDFs
Split text into chunks
Generate embeddings for each chunk
Store embeddings in FAISS
Retrieve relevant chunks for a query
Generate an answer using the LLM

Chunking Strategy
Used:
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_size=1000 keeps enough context in each chunk
chunk_overlap=200 preserves continuity across chunk boundaries
Good default for PDFs where content may span multiple sections

Embedding Model-all-MiniLM-L6-v2
Light Weight
General purpose

LLM- llama-3.3-70b-versatile
strong instruction following
good context-based QA performance
fast inference through Groq

Temperature - 0.3 
Reduced hallucinations

Retrieval Strategy - search_kwargs={"k": 4}
Retrieves top 4 most relevant chunks
Good balance between context coverage and noise reduction

Project Structure
.
├── app.py
├── .env
├── faiss_index/
├── requirements.txt
└── README.md

Installation-
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
pip install -r requirements.txt
GROQ_API_KEY=your_groq_api_key_here
streamlit run app.py

Limitations
Works only with text-based PDFs
Scanned PDFs require OCR
New uploads overwrite the existing FAISS index
No chat history or source citation display yet

Future Improvements
OCR support for scanned PDFs
Source chunk/page citations
Conversational memory
Per-document indexing
Better prompt and retrieval tuning


