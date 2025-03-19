
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
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Initialize database for chat history
def init_db():
    db_path = "pdf_chat_sessions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
   
    # Create messages table
    create_messages_table = """
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_history_id TEXT NOT NULL,
        sender_type TEXT NOT NULL,
        message_type TEXT NOT NULL,
        text_content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
   
    # Create PDF tracking table
    create_pdf_table = """
    CREATE TABLE IF NOT EXISTS pdf_uploads (
        pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_history_id TEXT NOT NULL,
        pdf_name TEXT NOT NULL,
        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
   
    cursor.execute(create_messages_table)
    cursor.execute(create_pdf_table)
    conn.commit()
    conn.close()
 
init_db()
 
# Database operations
def get_db_connection():
    return sqlite3.connect("pdf_chat_sessions.db", check_same_thread=False)
 
def save_text_message(chat_history_id, sender_type, text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
                   (chat_history_id, sender_type, 'text', text))
    conn.commit()
    conn.close()
 
def save_pdf_upload(chat_history_id, pdf_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO pdf_uploads (chat_history_id, pdf_name) VALUES (?, ?)',
                   (chat_history_id, pdf_name))
    conn.commit()
    conn.close()
 
def load_messages(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT message_id, sender_type, message_type, text_content
    FROM messages
    WHERE chat_history_id = ?
    ORDER BY message_id ASC
    """
    cursor.execute(query, (chat_history_id,))
    messages = cursor.fetchall()
    chat_history = []
    for message in messages:
        message_id, sender_type, message_type, text_content = message
        chat_history.append({
            'message_id': message_id,
            'sender_type': sender_type,
            'message_type': message_type,
            'content': text_content
        })
    conn.close()
    return chat_history
 
def get_uploaded_pdfs(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT pdf_name
    FROM pdf_uploads
    WHERE chat_history_id = ?
    ORDER BY upload_timestamp ASC
    """
    cursor.execute(query, (chat_history_id,))
    pdfs = cursor.fetchall()
    pdf_names = [pdf[0] for pdf in pdfs]
    conn.close()
    return pdf_names
 
def get_all_chat_history_ids():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id DESC"
    cursor.execute(query)
    chat_history_ids = cursor.fetchall()
    chat_history_id_list = [item[0] for item in chat_history_ids]
    conn.close()
    return chat_history_id_list
 
def delete_chat_history(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Delete messages for this chat history
    cursor.execute("DELETE FROM messages WHERE chat_history_id = ?", (chat_history_id,))
    # Delete PDF upload records for this chat history
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
            text += page.extract_text()
    return text
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for more precise retrieval
        chunk_overlap=200,  # Increased overlap to avoid losing context between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks
 
def get_vector_store(text_chunks, chat_history_id):
    """Create or update vector store with text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
   
    # Create a unique index path for this chat session
    index_path = f"faiss_index_{chat_history_id}"
   
    # Check if index already exists
    if os.path.exists(index_path):
        # Load existing index
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        # Add new texts to the existing index
        vector_store.add_texts(text_chunks)
    else:
        # Create new index
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
   
    # Save the updated or new index
    vector_store.save_local(index_path)
    return vector_store
 
def get_conversational_chain():
    """Create an enhanced prompt template and QA chain"""
    prompt_template = """
    You are an expert document analyst with exceptional abilities to understand, synthesize, and explain information from multiple documents.
   
    INSTRUCTIONS:
    1. Provide detailed, comprehensive answers based on the context provided
    2. If information appears in multiple documents, synthesize it into a coherent response
    3. Cite specific sections or pages when relevant (e.g., "According to document X...")
    4. If the answer isn't in the context, clearly state "This information is not available in the provided documents"
    5. If the question is a greeting or general inquiry, respond appropriately
    6. Structure complex answers with headings and bullet points for clarity when appropriate
    7. Prioritize accuracy over completeness
   
    Context:\n {context}
   
    Question: \n{question}
   
    Answer:
    """
   
    # Use a more capable model with higher temperature for more comprehensive responses
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
 
def user_input(user_question, chat_history_id):
    """Process user query and generate response"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        index_path = f"faiss_index_{chat_history_id}"
       
        # Load vector store for this chat session
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
       
        # Retrieve more documents for comprehensive answers
        docs = vector_store.similarity_search(user_question, k=5)
       
        # Get conversational chain
        chain = get_conversational_chain()
       
        # Generate response
        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )
       
        # Save messages to database
        save_text_message(chat_history_id, "user", user_question)
        save_text_message(chat_history_id, "assistant", response["output_text"])
       
        return response["output_text"]
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        save_text_message(chat_history_id, "assistant", error_message)
        return error_message
 
def main():
    st.set_page_config(
        page_title="Advanced PDF Chat Assistant",
        page_icon="📚",
        layout="wide"
    )
   
    st.header("📚 Advanced PDF Chat Assistant")
    st.markdown("""
    Upload multiple PDF documents and chat with an AI assistant that can answer questions
    based on their combined content. Your chat history and document uploads are saved for future sessions.
    """)

    # Initialize session state
    if "session_key" not in st.session_state:
        st.session_state.session_key = "new_session"
    if "new_session_key" not in st.session_state:
        st.session_state.new_session_key = None
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    # Add a state variable to track if chat should be displayed
    if "display_chat" not in st.session_state:
        st.session_state.display_chat = False
   
    # Handle session creation
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_key = st.session_state.new_session_key
        st.session_state.new_session_key = None
 
    # Create sidebar for session management
    with st.sidebar:
        st.title("Session Management")
       
        # Session selection
        chat_sessions = ["new_session"] + get_all_chat_history_ids()
        selected_session = st.selectbox(
            "Select or create a chat session",
            chat_sessions,
            index=chat_sessions.index(st.session_state.session_key) if st.session_state.session_key in chat_sessions else 0
        )
       
        # Update session key if a different session is selected
        if selected_session != st.session_state.session_key:
            st.session_state.session_key = selected_session
            st.rerun()
       
        # Delete session button
        if st.button("Delete Current Session", type="secondary"):
            if st.session_state.session_key != "new_session":
                delete_chat_history(st.session_state.session_key)
                st.session_state.session_key = "new_session"
                st.success("Session deleted!")
                st.rerun()
       
        # PDF upload section
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )
       
        # Process button
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents... This may take a moment."):
                    # Create new session if needed
                    if st.session_state.session_key == "new_session":
                        st.session_state.new_session_key = get_timestamp()
                        current_session = st.session_state.new_session_key
                    else:
                        current_session = st.session_state.session_key
                   
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                   
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                   
                    # Create/update vector store
                    get_vector_store(text_chunks, current_session)
                   
                    # Save PDF names to database
                    for pdf in pdf_docs:
                        save_pdf_upload(current_session, pdf.name)
                   
                    st.session_state.pdf_uploaded = True
                    st.success(f"✅ {len(pdf_docs)} documents processed successfully!")
            else:
                st.error("Please upload at least one document.")
       
        # Show uploaded PDFs for current session
        if st.session_state.session_key != "new_session":
            uploaded_pdfs = get_uploaded_pdfs(st.session_state.session_key)
            if uploaded_pdfs:
                st.subheader("Uploaded Documents")
                for pdf_name in uploaded_pdfs:
                    st.markdown(f"📄 {pdf_name}")
 
    # Main chat interface
    chat_container = st.container()

    # User input for questions
    user_question = st.chat_input("Ask a question about your documents...")

    # Process user input
    if user_question:
        if st.session_state.session_key == "new_session":
            if not st.session_state.pdf_uploaded:
                st.error("Please upload and process at least one document first.")
            else:
                # Create a new session
                st.session_state.new_session_key = get_timestamp()
                current_session = st.session_state.new_session_key
                # Process the question
                response = user_input(user_question, current_session)
                # Update the session key
                st.session_state.session_key = current_session
                st.session_state.display_chat = True
                # Don't rerun immediately
        else:
            # Get vector store path for current session
            index_path = f"faiss_index_{st.session_state.session_key}"
            
            # Check if vector store exists
            if os.path.exists(index_path):
                # Process the question
                response = user_input(user_question, st.session_state.session_key)
                st.session_state.display_chat = True
                # Don't rerun immediately
            else:
                st.error("No document data found for this session. Please upload and process documents.")

    # Display chat history
    if st.session_state.session_key != "new_session" or st.session_state.display_chat:
        with chat_container:
            chat_history = load_messages(st.session_state.session_key)
            
            # Create a scrollable container for messages
            with st.container():
                for message in chat_history:
                    with st.chat_message(name=message["sender_type"]):
                        st.write(message["content"])
    
if __name__ == "__main__":
    main()