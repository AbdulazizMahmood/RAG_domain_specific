
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import os

os.environ["GOOGLE_API_KEY"] = "api_key" # replace

def build_qa_chain(pdf_path="example.pdf"):
    loader = PyPDFLoader(pdf_path) 
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    db = FAISS.from_documents(docs, embeddings) 
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain