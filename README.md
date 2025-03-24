# Chat with Multiple PDFs

## Overview
This Streamlit-based application allows users to upload multiple PDF files and chat with them using Google Gemini AI. It extracts text from PDFs, creates vector embeddings, and enables users to ask questions based on the document contents.

## Features
- Upload and process multiple PDFs.
- Store chat history using SQLite.
- Retrieve context-aware answers using Google Gemini AI.
- Maintain multiple chat sessions.
- Delete chat history as needed.

## Requirements
- Python 3.8+
- Google API Key
- Required libraries (install using `pip install -r requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chat-with-pdfs.git
   cd chat-with-pdfs
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file and add:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run chatpdf1.py
   ```
2. Upload PDFs in the sidebar.
3. Ask questions in the text box.
4. View responses and chat history.

## Dependencies
- `streamlit`
- `PyPDF2`
- `langchain`
- `google-generativeai`
- `FAISS`
- `dotenv`
- `sqlite3`


## Demo video of the project
"https://www.loom.com/share/6c94b4eaba584f2db1bc27add1f63cf7?sid=2b140e1f-557d-4874-b59b-9225f6ca666b"

