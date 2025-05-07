from langchain_community.document_loaders import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import CONDENSE_QUESTION_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import os


def build_qa_chain(pdf_folder_path, index_path="faiss_index"):
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Check if the FAISS index already exists
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    if os.path.exists(index_path) and os.path.isdir(index_path):
        print(f"Loading existing vector database from {index_path}")

        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
    else:
        # Process documents and create a new index
        documents = []
        if not os.path.isdir(pdf_folder_path):
            raise ValueError(f"Provided path {pdf_folder_path} is not a directory.")

        for filename in tqdm(os.listdir(pdf_folder_path), desc="Loading documents"):
            file_path = os.path.join(pdf_folder_path, filename)
            file_lower = filename.lower()
            
            try:
                loader = None
                # Select appropriate loader based on file extension
                if file_lower.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_lower.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file_lower.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file_lower.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                elif file_lower.endswith(".html") or file_lower.endswith(".htm"):
                    loader = UnstructuredHTMLLoader(file_path)
                
                if loader:
                    docs = loader.load()
                    
                    # Add additional metadata to each document
                    for doc in docs:
                        doc.metadata["filename"] = filename
                        doc.metadata["file_path"] = file_path
                        doc.metadata["creation_time"] = os.path.getctime(file_path)
                        doc.metadata["last_modified"] = os.path.getmtime(file_path)
                        
                    documents.extend(docs)
                    print(f"Successfully loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        if not documents:
            raise ValueError("No documents were successfully loaded from the directory.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
        docs = splitter.split_documents(documents)

       
        db = FAISS.from_documents(docs, embeddings) 
        
        print(f"Saving vector database to {index_path}")
        db.save_local(index_path)
        
        retriever = db.as_retriever( search_kwargs={
            "k": 4,
            "score_threshold": 0.5, 
            "include_metadata": True  
        })

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 
    
    # Define the prompt template for condensing the question
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    # Define QA prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Provide details from the source document to support your answer.\n\n{context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    # Create document chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine into a conversational retrieval chain
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain