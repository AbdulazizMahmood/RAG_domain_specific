import streamlit as st
from rag_core import build_qa_chain 
from config import PDF_PATH
import os
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory


st.set_page_config(
    page_title="DocChat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Global font size reduction */
    body, p, div, span, label, .stMarkdown, .stText, .stTextInput {
        font-size: 0.9rem !important;
    }
    
    /* Smaller headers */
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 2rem;
        color: #1E88E5;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    h3, .sidebar h2 {
        font-size: 1.2rem !important;
    }
    
    /* Chat messages alignment */
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
    
    /* User messages styling */
    [data-testid="stChatMessageContent"] {
        border-radius: 10px;
        padding: 10px 15px;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    
    /* Reduce sidebar width */
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
    }
    
    /* Optional: Adjust the main content area to use the extra space */
    .main .block-container {
        max-width: calc(100% - 250px) !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Chat input field */
    .stTextInput input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("🧑‍⚕️ DocChat Settings")
    st.write("Configure your document chat experience")
    
    # Document information
    if "qa_chain" in st.session_state:
        st.success("✅ Documents loaded successfully")
    
    st.divider()
    
    st.subheader("💬 Chat Controls")
    if st.button("Clear chat history", key="clear_chat"):
        if "chat_memory" in st.session_state:
            st.session_state.chat_memory.clear()
        st.rerun()
    

st.title("Chat with your Documents 📚", help="Ask questions about your documents using semantic search and RAG")

if "qa_chain" not in st.session_state:
    with st.spinner("📚 Loading documents..."):
        st.session_state.qa_chain = build_qa_chain(pdf_folder_path=PDF_PATH)

chain = st.session_state.qa_chain

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = StreamlitChatMessageHistory()
    st.session_state.chat_memory.add_ai_message("👋 Hello! I'm DocChat. Ask me anything about your documents.")

chat_memory = st.session_state.chat_memory
memory = ConversationBufferMemory(
    chat_memory=chat_memory,
    return_messages=True,
    memory_key="chat_history",
)


avatars = {"human": "user", "ai": "assistant"}
for msg in chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input("Ask something about your documents..."):
    chat_memory.add_user_message(user_query)

    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        use_chat_history = len(chat_memory.messages) > 2

        message_placeholder = st.empty()
        source_placeholder = st.container()
        full_response = ""
        result = None
        
        chain_input = {
            "input": user_query,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        }

        
        for chunk in chain.stream(chain_input):
            if "answer" in chunk:
                full_response += chunk["answer"]
            elif "context" in chunk:
                context_docs = chunk.get("context", [])
                        
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)

        if context_docs:
            with source_placeholder.expander("📚 Sources", expanded=False):
                tabs = st.tabs([f"Source {i+1}" for i in range(min(5, len(context_docs)))])
                
                for i, (tab, doc) in enumerate(zip(tabs, context_docs[:5])):
                    with tab:
                        filename = doc.metadata.get("filename", "Unknown source")
                        page = doc.metadata.get("page", "")
                        page_info = f" (Page {page})" if page else ""
                        
                        st.markdown(f"**{filename}{page_info}**")
                        st.markdown("---")
                        st.markdown(doc.page_content)
                        
        chat_memory.add_ai_message(full_response)
