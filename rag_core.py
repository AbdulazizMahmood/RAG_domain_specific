
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from config import CONDENSE_QUESTION_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

        for filename in tqdm(os.listdir(pdf_folder_path), desc="Loading PDFs"):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder_path, filename)
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    # Add additional metadata to each document
                    for doc in docs:
                        doc.metadata["filename"] = filename
                        doc.metadata["file_path"] = pdf_path
                        doc.metadata["creation_time"] = os.path.getctime(pdf_path)
                        doc.metadata["last_modified"] = os.path.getmtime(pdf_path)
                        
                    documents.extend(docs)
                    print(f"Successfully loaded: {pdf_path}")
                except Exception as e:
                    print(f"Error loading {pdf_path}: {str(e)}")
        
        if not documents:
            raise ValueError("No PDF documents were successfully loaded from the directory.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
        docs = splitter.split_documents(documents)

       
        db = FAISS.from_documents(docs, embeddings) 
        
        print(f"Saving vector database to {index_path}")
        db.save_local(index_path)
        
        retriever = db.as_retriever( search_kwargs={
            "k": 4,  # Number of documents to retrieve
            "score_threshold": 0.5, 
            "include_metadata": True  
        })

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 
    
    # Define the prompt template for condensing the question
    condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        verbose=True,
        return_generated_question=True,
    )

    return qa_chain