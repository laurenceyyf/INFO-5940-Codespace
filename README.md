# ChatBot with RAG
A Retrieval-Augmented Generation (RAG) web app that lets you upload multiple documents (PDF, TXT) and ask question about their content via Streamlit. The source code is `app.py`.

## Features
Support .pdf and .txt file formats
Allow to uplaod multiple documents
LLM via LangChain (ChatOpenAI) and embeddings via OpenAIEmbeddings
RAG system with RecursiveCharacterTextSplitter and Chroma

## Instructions
1. Set the OPENAI_API_KEY in the terminal.
2. From the project root, run the following command:
   ```
   streamlit run chat_with_pdf.py
   ```
   Open the URL shown in the terminal to access the Streamlit interface.
3. Upload one or more files to the chat as prompted.
4. Type a question in the chat input.
5. The assistant's answers will stream in.

## Parameters
```
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
K = 5
```
The parameters can be adjusted according to the length of the files.