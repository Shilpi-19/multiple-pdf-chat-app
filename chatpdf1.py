import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize database
def init_db():
    conn = sqlite3.connect("pdf_chat_sessions.db")
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_history_id TEXT NOT NULL,
        sender_type TEXT NOT NULL,
        message_type TEXT NOT NULL,
        text_content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS pdf_uploads (
        pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_history_id TEXT NOT NULL,
        pdf_name TEXT NOT NULL,
        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

init_db()

# Database operations
def get_db_connection():
    return sqlite3.connect("pdf_chat_sessions.db", check_same_thread=False)

def save_text_message(chat_history_id, sender_type, text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
        (chat_history_id, sender_type, 'text', text)
    )
    conn.commit()
    conn.close()

def save_pdf_upload(chat_history_id, pdf_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO pdf_uploads (chat_history_id, pdf_name) VALUES (?, ?)',
        (chat_history_id, pdf_name)
    )
    conn.commit()
    conn.close()

def load_messages(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT message_id, sender_type, message_type, text_content
        FROM messages
        WHERE chat_history_id = ?
        ORDER BY message_id ASC
    """, (chat_history_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{'sender': m[1], 'content': m[3]} for m in messages]

def get_uploaded_pdfs(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pdf_name
        FROM pdf_uploads
        WHERE chat_history_id = ?
        ORDER BY upload_timestamp ASC
    """, (chat_history_id,))
    pdfs = [pdf[0] for pdf in cursor.fetchall()]
    conn.close()
    return pdfs


def get_all_chat_history_ids():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""

        SELECT DISTINCT chat_history_id FROM messages

        UNION

        SELECT DISTINCT chat_history_id FROM pdf_uploads

        ORDER BY chat_history_id DESC

    """)

    chat_history_ids = [item[0] for item in cursor.fetchall()]

    conn.close()

    return chat_history_ids
 

def delete_chat_history(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_history_id = ?", (chat_history_id,))
    cursor.execute("DELETE FROM pdf_uploads WHERE chat_history_id = ?", (chat_history_id,))
    conn.commit()
    conn.close()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# PDF processing functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, chat_history_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    index_path = f"faiss_index_{chat_history_id}"
    try:
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_texts(text_chunks)
        else:
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(index_path)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
You are an expert document analyst with exceptional abilities to understand, synthesize, and explain information from multiple documents.
INSTRUCTIONS:
1. Provide detailed, comprehensive answers based on the context provided.
2. If information appears in multiple documents, synthesize it into a coherent response.
3. Cite specific sections or pages when relevant (e.g., "According to document X...").
4. If the answer isn't in the context, clearly state "This information is not available in the provided documents".
5. If the question is a greeting or general inquiry, respond appropriately.
6. Structure complex answers with headings and bullet points for clarity when appropriate.
7. Prioritize accuracy over completeness.
Context:
{context}
Question:
{question}
Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_question(user_question, chat_history_id):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        index_path = f"faiss_index_{chat_history_id}"
        if not os.path.exists(index_path):
            return "No processed documents found. Please upload and process documents first."
        
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question, k=5)
        chain = get_conversational_chain()
        result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = result.get("output_text", "No answer generated.")
        save_text_message(chat_history_id, "user", user_question)
        save_text_message(chat_history_id, "assistant", answer)
        return answer
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        save_text_message(chat_history_id, "assistant", error_message)
        return error_message

# Initialize persistent session state keys if not already set
if "session_key" not in st.session_state:
    st.session_state.session_key = "new_session"
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def main():
    st.set_page_config(page_title="Advanced PDF Chat Assistant", page_icon="📚", layout="wide")
    st.header("📚 Advanced PDF Chat Assistant")
    st.markdown("""
    Upload multiple PDF documents and chat with an AI assistant that answers questions
    based on their combined content. Your chat history and document uploads are saved for future sessions.
    """)
 
    # Sidebar for session management and document uploads
    with st.sidebar:
        st.title("Session Management")
        chat_sessions = ["new_session"] + get_all_chat_history_ids()
        current = st.session_state.current_session or st.session_state.session_key
        selected_session = st.selectbox(
            "Select or create a chat session",
            chat_sessions,
            index=chat_sessions.index(current) if current in chat_sessions else 0,
            key="session_select"  # Unique key to avoid conflicts
        )
 
        # Update session state only when the selection changes
        if selected_session != st.session_state.session_key:
            st.session_state.session_key = selected_session
            st.session_state.current_session = selected_session if selected_session != "new_session" else None
            st.session_state.pdf_processed = (
                selected_session != "new_session" and os.path.exists(f"faiss_index_{selected_session}")
            )
            st.session_state.chat_history = load_messages(selected_session)
 
        if (
            st.button("Delete Current Session", key="delete_session")
            and st.session_state.session_key != "new_session"
        ):
            delete_chat_history(st.session_state.session_key)
            if os.path.exists(f"faiss_index_{st.session_state.session_key}"):
                import shutil
                shutil.rmtree(f"faiss_index_{st.session_state.session_key}")
            st.session_state.session_key = "new_session"
            st.session_state.current_session = None
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.success("Session deleted!")
 
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files", accept_multiple_files=True, type=["pdf"], key="uploader"
        )
        if st.button("Process Documents", key="process_docs") and pdf_docs:
            with st.spinner("Processing documents..."):
                if st.session_state.session_key == "new_session":
                    new_key = get_timestamp()
                    st.session_state.session_key = new_key
                    st.session_state.current_session = new_key
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks, st.session_state.session_key)
                    if vector_store:
                        for pdf in pdf_docs:
                            save_pdf_upload(st.session_state.session_key, pdf.name)
                        st.session_state.pdf_processed = True
                        st.session_state.chat_history = load_messages(st.session_state.session_key)
                        st.success(f"✅ {len(pdf_docs)} document(s) processed successfully!")
 
        if st.session_state.session_key != "new_session":
            uploaded = get_uploaded_pdfs(st.session_state.session_key)
            if uploaded:
                st.subheader("Uploaded Documents")
                for pdf_name in uploaded:
                    st.markdown(f"📄 {pdf_name}")
 
    # Main Chat Section
    st.markdown("---")
    st.subheader("Chat with your documents")
 
    # Check if we can display the chat interface
    can_chat = (
        st.session_state.session_key != "new_session"
        and os.path.exists(f"faiss_index_{st.session_state.session_key}")
    )
 
    if can_chat:
        user_question = st.chat_input(
            "Ask a question about your documents...", key="chat_input"
        )
        if user_question:
            answer = process_user_question(user_question, st.session_state.session_key)
            st.session_state.chat_history.append({"sender": "user", "content": user_question})
            st.session_state.chat_history.append({"sender": "assistant", "content": answer})
 
        # Display chat history
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["sender"]):
                    st.write(msg["content"])
    else:
        st.info("Please upload and process documents to start chatting.")

if __name__ == "__main__":
     main()        