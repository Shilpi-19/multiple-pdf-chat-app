# # import warnings
# # warnings.filterwarnings("ignore")
# # import streamlit as st
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.question_answering import load_qa_chain
# # from langchain.prompts import PromptTemplate
# # from dotenv import load_dotenv
# # import sqlite3
# # from datetime import datetime

# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Database initialization
# # def init_db():
# #     db_path = "chat_sessions.db"
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()

# #     create_messages_table = """
# #     CREATE TABLE IF NOT EXISTS messages (
# #         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
# #         chat_history_id TEXT NOT NULL,
# #         sender_type TEXT NOT NULL,
# #         message_type TEXT NOT NULL,
# #         text_content TEXT
# #     );
# #     """
# #     cursor.execute(create_messages_table)
# #     conn.commit()
# #     conn.close()

# # init_db()

# # def get_db_connection():
# #     return sqlite3.connect("chat_sessions.db", check_same_thread=False)

# # def save_text_message(chat_history_id, sender_type, text):
# #     conn = get_db_connection()
# #     cursor = conn.cursor()
# #     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
# #                    (chat_history_id, sender_type, 'text', text))
# #     conn.commit()
# #     conn.close()

# # def load_messages(chat_history_id):
# #     conn = get_db_connection()
# #     cursor = conn.cursor()
# #     query = "SELECT message_id, sender_type, message_type, text_content FROM messages WHERE chat_history_id = ?"
# #     cursor.execute(query, (chat_history_id,))
# #     messages = cursor.fetchall()
# #     chat_history = []
# #     for message in messages:
# #         message_id, sender_type, message_type, text_content = message
# #         chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
# #     conn.close()
# #     return chat_history

# # def get_all_chat_history_ids():
# #     conn = get_db_connection()
# #     cursor = conn.cursor()
# #     query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC"
# #     cursor.execute(query)
# #     chat_history_ids = cursor.fetchall()
# #     chat_history_id_list = [item[0] for item in chat_history_ids]
# #     conn.close()
# #     return chat_history_id_list

# # def delete_chat_history(chat_history_id):
# #     conn = get_db_connection()
# #     cursor = conn.cursor()
# #     query = "DELETE FROM messages WHERE chat_history_id = ?"
# #     cursor.execute(query, (chat_history_id,))
# #     conn.commit()
# #     conn.close()

# # def get_timestamp():
# #     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")

# # def get_conversational_chain():
# #     prompt_template = """
# #     You are an expert in analyzing and understanding documents. Give proper responses to greetings as well.
   
# #     Answer the question as detailed as possible from the provided context. Make sure to provide all relevant details.
# #     If the answer is not in the provided context, just say "I don't have enough information to answer that question based on the documents provided."
   
# #     If the question requires combining information from multiple documents, ensure that your answer is comprehensive and covers all relevant details.\n\n
# #     Context:\n {context}?\n
# #     Question: \n{question}\n

# #     Answer:
# #     """
# #     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
# #     return chain

# # def user_input(user_question, chat_history_id):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# #     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# #     docs = new_db.similarity_search(user_question)
# #     chain = get_conversational_chain()
# #     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
# #     save_text_message(chat_history_id, "user", user_question)
# #     save_text_message(chat_history_id, "assistant", response["output_text"])
# #     return response["output_text"]

# # def main():
# #     st.set_page_config("Chat With Multiple PDF")
# #     st.header("Chat with Multiple PDF 💁")

# #     # Initialize session state
# #     if "session_key" not in st.session_state:
# #         st.session_state.session_key = "new_session"
# #     if "new_session_key" not in st.session_state:
# #         st.session_state.new_session_key = None

# #     # Handle session creation
# #     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
# #         st.session_state.session_key = st.session_state.new_session_key
# #         st.session_state.new_session_key = None

# #     # Sidebar for session management
# #     st.sidebar.title("Chat Sessions")
# #     chat_sessions = ["new_session"] + get_all_chat_history_ids()
# #     selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=chat_sessions.index(st.session_state.session_key))

# #     # Update session key if a new session is selected
# #     if selected_session != st.session_state.session_key:
# #         st.session_state.session_key = selected_session

# #     # Delete session button
# #     if st.sidebar.button("Delete Chat Session"):
# #         if st.session_state.session_key != "new_session":
# #             delete_chat_history(st.session_state.session_key)
# #             st.session_state.session_key = "new_session"
# #             st.rerun()

# #     # User input for questions
# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         if st.session_state.session_key == "new_session":
# #             st.session_state.new_session_key = get_timestamp()
# #             st.session_state.session_key = st.session_state.new_session_key
# #         response = user_input(user_question, st.session_state.session_key)
# #         st.write("Reply: ", response)

# #     # PDF upload and processing
# #     with st.sidebar:
# #         st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 get_vector_store(text_chunks)
# #                 st.success("Done")

# #     # Display chat history
# #     if st.session_state.session_key != "new_session":
# #         chat_history = load_messages(st.session_state.session_key)
# #         for message in chat_history:
# #             with st.chat_message(name=message["sender_type"]):
# #                 st.write(message["content"])

# # if __name__ == "__main__":
# #     main()


# import warnings
# warnings.filterwarnings("ignore")
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# from datetime import datetime
# import uuid
# import tempfile

# # Load environment variables
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Database initialization
# def init_db():
#     db_path = "chat_sessions.db"
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Messages table for storing chat history
#     create_messages_table = """
#     CREATE TABLE IF NOT EXISTS messages (
#         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         sender_type TEXT NOT NULL,
#         message_type TEXT NOT NULL,
#         text_content TEXT,
#         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     );
#     """
    
#     # PDFs table for tracking uploaded documents
#     create_pdfs_table = """
#     CREATE TABLE IF NOT EXISTS pdfs (
#         pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         pdf_name TEXT NOT NULL,
#         upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     );
#     """
    
#     cursor.execute(create_messages_table)
#     cursor.execute(create_pdfs_table)
#     conn.commit()
#     conn.close()

# init_db()

# def get_db_connection():
#     return sqlite3.connect("chat_sessions.db", check_same_thread=False)

# def save_text_message(chat_history_id, sender_type, text):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content, timestamp) VALUES (?, ?, ?, ?, ?)',
#                    (chat_history_id, sender_type, 'text', text, datetime.now()))
#     conn.commit()
#     conn.close()

# def save_pdf_record(chat_history_id, pdf_name):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO pdfs (chat_history_id, pdf_name, upload_timestamp) VALUES (?, ?, ?)',
#                    (chat_history_id, pdf_name, datetime.now()))
#     conn.commit()
#     conn.close()

# def load_messages(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#     SELECT message_id, sender_type, message_type, text_content, timestamp 
#     FROM messages 
#     WHERE chat_history_id = ? 
#     ORDER BY timestamp ASC
#     """
#     cursor.execute(query, (chat_history_id,))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in messages:
#         message_id, sender_type, message_type, text_content, timestamp = message
#         chat_history.append({
#             'message_id': message_id, 
#             'sender_type': sender_type, 
#             'message_type': message_type, 
#             'content': text_content,
#             'timestamp': timestamp
#         })
#     conn.close()
#     return chat_history

# def get_session_pdfs(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#     SELECT pdf_name 
#     FROM pdfs 
#     WHERE chat_history_id = ? 
#     ORDER BY upload_timestamp ASC
#     """
#     cursor.execute(query, (chat_history_id,))
#     pdfs = cursor.fetchall()
#     pdf_names = [pdf[0] for pdf in pdfs]
#     conn.close()
#     return pdf_names

# def get_all_chat_history_ids():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#     SELECT DISTINCT m.chat_history_id, MAX(m.timestamp) as last_message, 
#            (SELECT COUNT(*) FROM pdfs p WHERE p.chat_history_id = m.chat_history_id) as pdf_count
#     FROM messages m
#     GROUP BY m.chat_history_id
#     ORDER BY last_message DESC
#     """
#     cursor.execute(query)
#     chat_history_ids = cursor.fetchall()
#     chat_history_id_list = [
#         {"id": item[0], 
#          "last_active": item[1], 
#          "pdf_count": item[2]} 
#         for item in chat_history_ids
#     ]
#     conn.close()
#     return chat_history_id_list

# def delete_chat_history(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Delete messages
#     cursor.execute("DELETE FROM messages WHERE chat_history_id = ?", (chat_history_id,))
#     # Delete PDF records
#     cursor.execute("DELETE FROM pdfs WHERE chat_history_id = ?", (chat_history_id,))
#     conn.commit()
#     conn.close()
    
#     # Also delete the vector store if it exists
#     vector_store_dir = f"faiss_index_{chat_history_id}"
#     if os.path.exists(vector_store_dir):
#         import shutil
#         shutil.rmtree(vector_store_dir)

# def get_timestamp():
#     return datetime.now().strftime("%Y%m%d_%H%M%S")

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     # Better chunking strategy with smaller chunks and more overlap
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks, chat_history_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
#     # Create path for session-specific vector store
#     vector_store_dir = f"faiss_index_{chat_history_id}"
    
#     # Check if vector store already exists for this session
#     if os.path.exists(vector_store_dir):
#         try:
#             # Load existing vector store
#             vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
#             # Add new chunks to existing store
#             vector_store.add_texts(text_chunks)
#         except Exception as e:
#             st.error(f"Error loading existing vector store: {e}")
#             # Create new vector store if loading fails
#             vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     else:
#         # Create new vector store
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
#     # Save the updated vector store
#     vector_store.save_local(vector_store_dir)
#     return vector_store

# def get_conversational_chain():
#     prompt_template = """
#     You are an expert document analysis assistant specialized in extracting and synthesizing information from PDF documents.
    
#     Instructions:
#     1. Answer questions based ONLY on the provided context.
#     2. Be thorough and comprehensive in your answers, providing all relevant details from the context.
#     3. If information appears in multiple parts of the context, synthesize it into a coherent answer.
#     4. If the question has no answer in the context, respond with "I don't have sufficient information to answer this question based on the provided documents."
#     5. Do not make up information or draw on knowledge outside of the given context.
#     6. If the user greets you or asks a general question unrelated to the documents, respond appropriately.
#     7. If the user asks about what documents are loaded, refer them to check the sidebar where document names are listed.
    
#     Context:
#     {context}
    
#     Question: 
#     {question}
    
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.2)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, chat_history_id):
#     try:
#         # Create path for session-specific vector store
#         vector_store_dir = f"faiss_index_{chat_history_id}"
        
#         if not os.path.exists(vector_store_dir):
#             return "Please upload and process PDF documents before asking questions."
        
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#         vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        
#         # Retrieve more documents for better context (k=5)
#         docs = vector_store.similarity_search(user_question, k=5)
        
#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
#         save_text_message(chat_history_id, "user", user_question)
#         save_text_message(chat_history_id, "assistant", response["output_text"])
        
#         return response["output_text"]
#     except Exception as e:
#         error_message = f"An error occurred: {str(e)}"
#         save_text_message(chat_history_id, "user", user_question)
#         save_text_message(chat_history_id, "assistant", error_message)
#         return error_message

# def handle_pdf_upload(pdf_docs, chat_history_id):
#     if not pdf_docs:
#         return "No PDFs uploaded", False
    
#     try:
#         # Save temporary files for PDFs
#         temp_pdf_files = []
#         for pdf in pdf_docs:
#             # Save PDF names in database
#             save_pdf_record(chat_history_id, pdf.name)
            
#             # Create a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#                 temp_file.write(pdf.getvalue())
#                 temp_pdf_files.append(temp_file.name)
        
#         # Extract text from PDFs
#         raw_text = ""
#         for temp_file_path in temp_pdf_files:
#             with open(temp_file_path, 'rb') as file:
#                 pdf_reader = PdfReader(file)
#                 for page in pdf_reader.pages:
#                     raw_text += page.extract_text()
            
#             # Clean up temporary file
#             os.unlink(temp_file_path)
        
#         # Process text chunks
#         text_chunks = get_text_chunks(raw_text)
        
#         # Create/update vector store
#         get_vector_store(text_chunks, chat_history_id)
        
#         return f"Successfully processed {len(pdf_docs)} PDF document(s)", True
#     except Exception as e:
#         return f"Error processing PDFs: {str(e)}", False

# def main():
#     st.set_page_config(
#         page_title="PDF Chat Assistant",
#         page_icon="📚",
#         layout="wide"
#     )
    
#     # Apply CSS for better styling
#     st.markdown("""
#     <style>
#     .main {
#         padding: 2rem;
#     }
#     .chat-message {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .user-message {
#         background-color: #e6f7ff;
#     }
#     .assistant-message {
#         background-color: #f0f0f0;
#     }
#     .stButton>button {
#         width: 100%;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.title("📚 PDF Chat Assistant")
#     st.subheader("Upload PDFs and ask questions about their content")
    
#     # Initialize session state
#     if "session_key" not in st.session_state:
#         st.session_state.session_key = "new_session"
#     if "new_session_key" not in st.session_state:
#         st.session_state.new_session_key = None
#     if "pdf_processed" not in st.session_state:
#         st.session_state.pdf_processed = False
    
#     # Handle session creation
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
#         st.session_state.session_key = st.session_state.new_session_key
#         st.session_state.new_session_key = None
    
#     # Create sidebar for session management and PDF upload
#     with st.sidebar:
#         st.header("Session Management")
        
#         # Session selection
#         chat_sessions_data = get_all_chat_history_ids()
#         chat_sessions = ["new_session"] + [session["id"] for session in chat_sessions_data]
        
#         # Format session display names
#         session_display_names = ["New Session"]
#         for session in chat_sessions_data:
#             timestamp = datetime.fromisoformat(session["last_active"].replace(' ', 'T'))
#             display_date = timestamp.strftime("%b %d, %Y %H:%M")
#             session_display_names.append(f"Session {session['id'][:8]} ({display_date}) - {session['pdf_count']} PDFs")
        
#         selected_session_index = 0
#         if st.session_state.session_key in chat_sessions:
#             selected_session_index = chat_sessions.index(st.session_state.session_key)
        
#         selected_session_display = st.selectbox(
#             "Select a chat session",
#             session_display_names,
#             index=selected_session_index
#         )
        
#         # Convert display name back to session ID
#         selected_index = session_display_names.index(selected_session_display)
#         if selected_index == 0:
#             selected_session = "new_session"
#         else:
#             selected_session = chat_sessions[selected_index]
        
#         # Update session key if a new session is selected
#         if selected_session != st.session_state.session_key:
#             st.session_state.session_key = selected_session
#             st.rerun()
        
#         # Delete session button
#         if st.button("Delete Current Session") and st.session_state.session_key != "new_session":
#             delete_chat_history(st.session_state.session_key)
#             st.session_state.session_key = "new_session"
#             st.session_state.pdf_processed = False
#             st.rerun()
        
#         st.header("Document Upload")
        
#         # PDF upload and processing
#         pdf_docs = st.file_uploader(
#             "Upload PDF documents",
#             accept_multiple_files=True,
#             type=["pdf"]
#         )
        
#         # if st.button("Process Documents"):
#         #     if st.session_state.session_key == "new_session":
#         #         st.session_state.new_session_key = get_timestamp()
#         #         chat_history_id = st.session_state.new_session_key
#         #     else:
#         #         chat_history_id = st.session_state.session_key
            
#         #     with st.spinner("Processing documents..."):
#         #         result_message, success = handle_pdf_upload(pdf_docs, chat_history_id)
#         #         if success:
#         #             st.session_state.pdf_processed = True
#         #             st.success(result_message)
#         #         else:
#         #             st.error(result_message)
        
#         if st.button("Process Documents"):
#     if st.session_state.session_key == "new_session":
#         new_session_key = get_timestamp()
#         st.session_state.session_key = new_session_key  # Save new session key persistently
#         chat_history_id = new_session_key
#     else:
#         chat_history_id = st.session_state.session_key

#     with st.spinner("Processing documents..."):
#         result_message, success = handle_pdf_upload(pdf_docs, chat_history_id)
        
#         if success:
#             st.session_state.pdf_processed = True
#             st.session_state["documents_processed"] = True  # Store success flag
#             st.success(result_message)
#         else:
#             st.error(result_message)
    
#     # Force a re-run only after processing to avoid the need for pressing again
#     st.rerun()

#         # Display PDFs loaded in this session
#         if st.session_state.session_key != "new_session":
#             pdf_list = get_session_pdfs(st.session_state.session_key)
#             if pdf_list:
#                 st.header("Loaded Documents")
#                 for idx, pdf_name in enumerate(pdf_list):
#                     st.write(f"{idx+1}. {pdf_name}")
    
#     # Chat display area
#     chat_container = st.container()
    
#     # Display chat history
#     with chat_container:
#         if st.session_state.session_key != "new_session":
#             chat_history = load_messages(st.session_state.session_key)
            
#             for message in chat_history:
#                 message_type = message["sender_type"]
#                 content = message["content"]
                
#                 if message_type == "user":
#                     st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)
    
#     # User input
#     user_question = st.text_input("Ask a question about your documents:")
    
#     if user_question:
#         if st.session_state.session_key == "new_session":
#             # Create new session if asking a question in a new session
#             if not st.session_state.pdf_processed:
#                 st.warning("Please upload and process PDF documents before asking questions.")
#             else:
#                 st.session_state.new_session_key = get_timestamp()
#                 chat_history_id = st.session_state.new_session_key
#                 response = user_input(user_question, chat_history_id)
#                 st.session_state.session_key = chat_history_id
#                 st.rerun()
#         else:
#             # Use existing session
#             with st.spinner("Thinking..."):
#                 response = user_input(user_question, st.session_state.session_key)
#                 st.rerun()

# if __name__ == "__main__":
#     main()



# import warnings
# warnings.filterwarnings("ignore")
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# from datetime import datetime
# from io import BytesIO

# # Load environment variables
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Configuration
# config = {
#     "chat_config": {
#         "chat_memory_length": 10  # Number of previous messages to include in context
#     },
#     "database_path": "chat_sessions.db"
# }

# # Database initialization
# def init_db():
#     conn = sqlite3.connect(config["database_path"])
#     cursor = conn.cursor()
   
#     # Create messages table
#     create_messages_table = """
#     CREATE TABLE IF NOT EXISTS messages (
#         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         sender_type TEXT NOT NULL,
#         message_type TEXT NOT NULL,
#         text_content TEXT,
#         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     );
#     """
#     cursor.execute(create_messages_table)
   
#     # Create PDF tracking table
#     create_pdf_table = """
#     CREATE TABLE IF NOT EXISTS pdf_files (
#         pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         pdf_name TEXT NOT NULL,
#         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     );
#     """
#     cursor.execute(create_pdf_table)
   
#     conn.commit()
#     conn.close()

# init_db()

# # Database operations
# def get_db_connection():
#     return sqlite3.connect(config["database_path"], check_same_thread=False)

# def save_text_message(chat_history_id, sender_type, text):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
#                    (chat_history_id, sender_type, 'text', text))
#     conn.commit()
#     conn.close()

# def save_pdf_info(chat_history_id, pdf_name):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO pdf_files (chat_history_id, pdf_name) VALUES (?, ?)',
#                    (chat_history_id, pdf_name))
#     conn.commit()
#     conn.close()

# def get_session_pdfs(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('SELECT pdf_name FROM pdf_files WHERE chat_history_id = ?', (chat_history_id,))
#     pdfs = cursor.fetchall()
#     conn.close()
#     return [pdf[0] for pdf in pdfs]

# def load_messages(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#     SELECT message_id, sender_type, message_type, text_content
#     FROM messages
#     WHERE chat_history_id = ?
#     ORDER BY message_id ASC
#     """
#     cursor.execute(query, (chat_history_id,))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in messages:
#         message_id, sender_type, message_type, text_content = message
#         chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
#     conn.close()
#     return chat_history

# def load_last_k_messages(chat_history_id, k=10):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = """
#     SELECT sender_type, text_content
#     FROM messages
#     WHERE chat_history_id = ? AND message_type = 'text'
#     ORDER BY message_id DESC
#     LIMIT ?
#     """
#     cursor.execute(query, (chat_history_id, k))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in reversed(messages):  # Reverse to get chronological order
#         sender_type, text_content = message
#         chat_history.append({'role': sender_type, 'content': text_content})
#     conn.close()
#     return chat_history

# def get_all_chat_history_ids():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id DESC"
#     cursor.execute(query)
#     chat_history_ids = cursor.fetchall()
#     chat_history_id_list = [item[0].replace(":", "_") for item in chat_history_ids]  # Normalize keys
#     conn.close()
#     return chat_history_id_list

# def delete_chat_history(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM messages WHERE chat_history_id = ?", (chat_history_id,))
#     cursor.execute("DELETE FROM pdf_files WHERE chat_history_id = ?", (chat_history_id,))
#     conn.commit()
#     conn.close()

# def get_timestamp():
#     return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Use underscores instead of colons

# # PDF Processing
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks, session_key):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
   
#     # Save with session-specific name
#     index_path = f"faiss_index_{session_key}"
#     vector_store.save_local(index_path)
#     return index_path

# def load_vector_store(index_path):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     if os.path.exists(index_path):
#         vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#         return vector_store
#     return None

# # Chat Logic
# def get_conversational_chain():
#     prompt_template = """
#     You are an expert in analyzing and understanding documents. Give proper responses to greetings as well.
   
#     Answer the question as detailed as possible from the provided context. Make sure to provide all relevant details.
#     If the answer is not in the provided context, just say "I don't have enough information to answer that question based on the documents provided."
   
#     If the question requires combining information from multiple documents, ensure that your answer is comprehensive and covers all relevant details.
   
#     Context:
#     {context}
   
#     Question:
#     {question}
   
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, session_key, index_path):
#     if not os.path.exists(index_path):
#         return "Please upload PDF documents first."
 
#     vector_store = load_vector_store(index_path)
#     if not vector_store:
#         return "Error loading document database. Please try uploading the documents again."
   
#     docs = vector_store.similarity_search(user_question, k=5)  # Get top 5 most relevant chunks
   
#     chain = get_conversational_chain()
   
#     # Include chat history for context
#     chat_history = load_last_k_messages(session_key, config["chat_config"]["chat_memory_length"])
#     history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
   
#     # Combine with document context
#     response = chain(
#         {
#             "input_documents": docs,
#             "question": user_question
#         },
#         return_only_outputs=True
#     )
   
#     # Save the interaction
#     save_text_message(session_key, "user", user_question)
#     save_text_message(session_key, "assistant", response["output_text"])
   
#     return response["output_text"]

# # UI Helpers
# def get_session_key():
#     if st.session_state.session_key == "new_session":
#         st.session_state.new_session_key = get_timestamp()
#         return st.session_state.new_session_key
#     return st.session_state.session_key

# def get_avatar(sender_type):
#     if sender_type == "user":
#         return "👤"
#     elif sender_type == "assistant":
#         return "🤖"
#     return None

# # Main Application
# def main():
#     st.set_page_config("Multi-PDF Chat Assistant", layout="wide")
#     st.header("📚 Multi-PDF Chat Assistant")
   
#     # Initialize session state
#     if "session_key" not in st.session_state:
#         st.session_state.session_key = "new_session"
#         st.session_state.new_session_key = None
#         st.session_state.processed_files = []
#         st.session_state.index_path = None
   
#     # Update session key if needed
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
#         st.session_state.session_key = st.session_state.new_session_key
#         st.session_state.new_session_key = None
   
#     # Sidebar for session management
#     with st.sidebar:
#         st.title("📝 Chat Sessions")
#         chat_sessions = ["new_session"] + get_all_chat_history_ids()
#         selected_session = st.selectbox(
#             "Select a chat session",
#             chat_sessions,
#             index=chat_sessions.index(st.session_state.session_key.replace(":", "_"))  # Normalize key
#         )
       
#         # Update session key if a new session is selected
#         if selected_session != st.session_state.session_key.replace(":", "_"):  # Normalize key
#             st.session_state.session_key = selected_session
#             if selected_session != "new_session":
#                 st.session_state.index_path = f"faiss_index_{selected_session}"
#                 st.session_state.processed_files = get_session_pdfs(selected_session)
#             else:
#                 st.session_state.index_path = None
#                 st.session_state.processed_files = []
       
#         # Delete session button
#         if st.button("🗑️ Delete Chat Session"):
#             if st.session_state.session_key != "new_session":
#                 delete_chat_history(st.session_state.session_key)
#                 # Clean up associated index files
#                 index_path = f"faiss_index_{st.session_state.session_key}"
#                 if os.path.exists(index_path):
#                     import shutil
#                     shutil.rmtree(index_path)
#                 st.session_state.session_key = "new_session"
#                 st.session_state.index_path = None
#                 st.session_state.processed_files = []
#                 st.rerun()
       
#         # PDF upload and processing
#         st.title("📄 Upload Documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files",
#             accept_multiple_files=True,
#             type=["pdf"],
#             key="pdf_uploader"
#         )
       
#         # Process button
#         if st.button("📥 Process Documents"):
#             if pdf_docs:
#                 with st.spinner("Processing documents..."):
#                     session_key = get_session_key()
                   
#                     # Track uploaded files
#                     for pdf in pdf_docs:
#                         if pdf.name not in st.session_state.processed_files:
#                             st.session_state.processed_files.append(pdf.name)
#                             save_pdf_info(session_key, pdf.name)
                   
#                     # Extract text from PDFs
#                     raw_text = get_pdf_text(pdf_docs)
                   
#                     # Create text chunks
#                     text_chunks = get_text_chunks(raw_text)
                   
#                     # Create vector store
#                     index_path = get_vector_store(text_chunks, session_key)
#                     st.session_state.index_path = index_path
                   
#                     st.success(f"Processed {len(pdf_docs)} documents!")
       
#         # Show processed files
#         if st.session_state.processed_files:
#             st.subheader("Processed Documents:")
#             for file in st.session_state.processed_files:
#                 st.text(f"• {file}")
   
#     # Chat interface
#     st.subheader("Chat")
#     chat_container = st.container()
   
#     # User input
#     user_question = st.chat_input("Ask a question about your documents...")
   
#     # Handle user input
#     if user_question:
#         if st.session_state.session_key == "new_session":
#             st.session_state.new_session_key = get_timestamp()
#             st.session_state.session_key = st.session_state.new_session_key
       
#         if st.session_state.index_path:
#             with st.spinner("Thinking..."):
#                 response = user_input(user_question, st.session_state.session_key, st.session_state.index_path)
#         else:
#             response = "Please upload and process documents first."
#             save_text_message(st.session_state.session_key, "user", user_question)
#             save_text_message(st.session_state.session_key, "assistant", response)
   
#     # Display chat history
#     with chat_container:
#         if st.session_state.session_key != "new_session":
#             chat_history = load_messages(st.session_state.session_key)
#             for message in chat_history:
#                 with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
#                     st.write(message["content"])

# if __name__ == "__main__":
#     main()






# import warnings
# warnings.filterwarnings("ignore")
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# from datetime import datetime

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Database initialization
# def init_db():
#     db_path = "chat_sessions.db"
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     create_messages_table = """
#     CREATE TABLE IF NOT EXISTS messages (
#         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         sender_type TEXT NOT NULL,
#         message_type TEXT NOT NULL,
#         text_content TEXT
#     );
#     """
#     cursor.execute(create_messages_table)
#     conn.commit()
#     conn.close()

# init_db()

# def get_db_connection():
#     return sqlite3.connect("chat_sessions.db", check_same_thread=False)

# def save_text_message(chat_history_id, sender_type, text):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
#                    (chat_history_id, sender_type, 'text', text))
#     conn.commit()
#     conn.close()

# def load_messages(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT message_id, sender_type, message_type, text_content FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in messages:
#         message_id, sender_type, message_type, text_content = message
#         chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
#     conn.close()
#     return chat_history

# def get_all_chat_history_ids():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC"
#     cursor.execute(query)
#     chat_history_ids = cursor.fetchall()
#     chat_history_id_list = [item[0] for item in chat_history_ids]
#     conn.close()
#     return chat_history_id_list

# def delete_chat_history(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "DELETE FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     conn.commit()
#     conn.close()

# def get_timestamp():
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     You are an expert in analyzing and understanding documents. Give proper responses to greetings as well.
   
#     Answer the question as detailed as possible from the provided context. Make sure to provide all relevant details.
#     If the answer is not in the provided context, just say "I don't have enough information to answer that question based on the documents provided."
   
#     If the question requires combining information from multiple documents, ensure that your answer is comprehensive and covers all relevant details.\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, chat_history_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     save_text_message(chat_history_id, "user", user_question)
#     save_text_message(chat_history_id, "assistant", response["output_text"])
#     return response["output_text"]

# def main():
#     st.set_page_config("Chat With Multiple PDF")
#     st.header("Chat with Multiple PDF 💁")

#     # Initialize session state
#     if "session_key" not in st.session_state:
#         st.session_state.session_key = "new_session"
#     if "new_session_key" not in st.session_state:
#         st.session_state.new_session_key = None

#     # Handle session creation
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
#         st.session_state.session_key = st.session_state.new_session_key
#         st.session_state.new_session_key = None

#     # Sidebar for session management
#     st.sidebar.title("Chat Sessions")
#     chat_sessions = ["new_session"] + get_all_chat_history_ids()
#     selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=chat_sessions.index(st.session_state.session_key))

#     # Update session key if a new session is selected
#     if selected_session != st.session_state.session_key:
#         st.session_state.session_key = selected_session

#     # Delete session button
#     if st.sidebar.button("Delete Chat Session"):
#         if st.session_state.session_key != "new_session":
#             delete_chat_history(st.session_state.session_key)
#             st.session_state.session_key = "new_session"
#             st.rerun()

#     # User input for questions
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         if st.session_state.session_key == "new_session":
#             st.session_state.new_session_key = get_timestamp()
#             st.session_state.session_key = st.session_state.new_session_key
#         response = user_input(user_question, st.session_state.session_key)
#         st.write("Reply: ", response)

#     # PDF upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

#     # Display chat history
#     if st.session_state.session_key != "new_session":
#         chat_history = load_messages(st.session_state.session_key)
#         for message in chat_history:
#             with st.chat_message(name=message["sender_type"]):
#                 st.write(message["content"])

# if __name__ == "__main__":
#     main()



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