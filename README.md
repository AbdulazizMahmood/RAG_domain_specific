# PDF Chatbot

## Description

This project is a Retrieval Augmented Generation (RAG) application that allows users to chat with a PDF document. Users can ask questions in natural language, and the application will retrieve relevant information from the PDF and generate an answer.

## Features

- Chat with PDF documents
- Natural language questions
- Conversation history

## Technologies Used

- Python
- Streamlit
- Langchain

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure PDF Path:**
    - Open `config.py`.
    - Update the `PDF_PATH` variable to the path of your PDF file.
    ```python
    # config.py
    PDF_PATH = "path/to/your/document.pdf"
    ```
5.  **Place your PDF:**
    - Ensure the PDF file specified in `config.py` is accessible by the application. You can place it in the `Data/` directory or any other location and update the path accordingly.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run ui_app.py
    ```
2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
3.  Type your questions about the PDF in the input box and press Enter.

## File Structure

```
.
├── Data/                     # Directory to store PDF files (optional, configure in config.py)
├── persist_directory/        # Directory for storing vector database persistence
├── __pycache__/              # Python cache files
├── config.py                 # Configuration file (e.g., PDF path)
├── rag_core.py               # Core RAG logic (building QA chain, etc.)
├── requirements.txt          # Python dependencies
├── ui_app.py                 # Streamlit user interface application
└── README.md                 # This file
```
