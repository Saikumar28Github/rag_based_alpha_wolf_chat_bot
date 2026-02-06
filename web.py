#basic requirements 
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# rag system required modules 
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# for api key
load_dotenv()

# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

# llm and rag system set up 
# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
)
# Session State
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "normal"   # normal | rag

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# Load the embedding model
embedding = HuggingFaceEmbeddings()

# RAG FUNCTIONS
# -------------------------------------------------
def process_document_to_chroma_db(file_name):
    file_path = os.path.join(working_dir, file_name)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )
# retrieval system 
def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke({"query": user_question})
    return response["result"]

# web page setup
st.set_page_config(
    page_title="Alpha_Wolf",
    page_icon="🐺",
    layout="centered",
)
st.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
    width=60
)
st.title("Welcome to Alpha bot")

# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📂 Upload PDF for RAG")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        save_path = os.path.join(working_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            process_document_to_chroma_db(uploaded_file.name)

        st.success("PDF processed. RAG mode enabled.")
        st.session_state.chat_mode = "rag"
        st.session_state.rag_ready = True

    if st.button("🔄 Back to Normal Chat"):
        st.session_state.chat_mode = "normal"
        st.session_state.chat_history = []

# CHAT HISTORY
# -------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# CHAT INPUT
# -------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # show user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.chat_mode == "rag" and st.session_state.rag_ready:
                answer = answer_question(user_input)
            else:
                response = llm.invoke(
                    [{"role": "system", "content": "You are a helpful assistant"},
                     *st.session_state.chat_history]
                )
                answer = response.content

        st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

