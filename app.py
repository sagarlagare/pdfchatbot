import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

# --- IMPORTS FOR STABLE VERSION 0.1.20 ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("⚡ Chat with PDF (Final Stable Version)")

# Load environment variables
load_dotenv()
# Fetch the key immediately on load
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Settings")
    
    # Check if key exists
    if GROQ_API_KEY:
        st.success("✅ API Key loaded from .env")
    else:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.info("Please add GROQ_API_KEY=gsk_... to your .env file")

    uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
    process_button = st.button("Submit & Process")

# ---------------- FUNCTIONS ----------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except: pass
    return text

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return len(chunks)


def get_answer_chain():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    template = """Use the following context to answer the question.
    If you don't know, say "I don't know".

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# ---------------- MAIN APP ----------------

if process_button and uploaded_files:
    if not GROQ_API_KEY:
        st.error("Cannot process: API Key missing.")
    else:
        with st.spinner("Processing..."):
            text = get_pdf_text(uploaded_files)
            if not text.strip():
                st.error("❌ PDF is empty or scanned.")
            else:
                count = create_vector_store(text)
                st.success(f"✅ Processed {count} chunks.")

user_question = st.text_input("Ask a question")

if user_question:
    if not GROQ_API_KEY:
        st.error("Please add GROQ_API_KEY to your .env file.")
    elif not os.path.exists("faiss_index"):
        st.error("Please process a PDF first.")
    else:
        with st.spinner("Thinking..."):
            try:
                # No longer need to pass key as argument
                chain = get_answer_chain()
                response = chain.invoke(user_question)["result"] 
                st.write("### 🤖 Answer")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")