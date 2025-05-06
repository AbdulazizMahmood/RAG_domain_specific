import streamlit as st
from rag_core import build_qa_chain 
from config import PDF_PATH

st.set_page_config(page_title="📄 PDF-Chatbot", layout="wide") 
st.title("📄 Chat with your PDF")

qa_chain = build_qa_chain(PDF_PATH) #Builds the QA chain using the specified PDF file

#Initializes the chat history in Streamlit's session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  


question = st.text_input("What would you like to know?", key="input")

if question:
    result = qa_chain.invoke({
        "question": question,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append((question, result["answer"])) 

    # Displays the chat history in reverse order (newest on top)
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**❓ Question {len(st.session_state.chat_history) - i}:** {q}")
        st.markdown(f"**🤖 Answer:** {a}")