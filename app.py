import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
K = 5

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
    api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI")

uploaded_files = st.file_uploader("Upload some files", type=("txt", "pdf"), accept_multiple_files=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP
)

# Save the in-memory uploaded_file to a real path so loaders can read it
def file_to_path(uploaded_file):
    suffix = ".txt" if uploaded_file.name.lower().endswith(".txt") else ".pdf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return tf.name

chunks = []

# Read the content of the uploaded files and split the documents
for file in uploaded_files:
    path = file_to_path(file)
    if file.name.lower().endswith(".txt"):
        loader = TextLoader(path)
    if file.name.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    documents = loader.load()
    chunks.extend(text_splitter.split_documents(documents))

# Streamlit Setup
question = st.chat_input(
    "Ask something about the files",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload one or multiple files (.txt and .pdf are supported), then ask me questions about the files."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_files:
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Index chunks into a vector db (ChromaDB)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(model="openai.text-embedding-3-large"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": K})
    
    # Retrive relavent content from files using similarity search
    retrieved_docs = retriever.invoke(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = prompt.invoke({"question": question, "context": docs_content})

    # Stream the assistant's response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                acc += chunk.content
                placeholder.markdown(acc)

    response = acc

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})