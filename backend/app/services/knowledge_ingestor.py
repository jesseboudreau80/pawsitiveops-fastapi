from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # ✅ updated import
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# Paths
KB_DIR = "backend/app/knowledge_base"
VECTOR_DB_DIR = "backend/app/vector_store"

def load_documents():
    documents = []
    for root, _, files in os.walk(KB_DIR):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.txt')):
                path = os.path.join(root, file)
                loader = UnstructuredFileLoader(path)
                docs = loader.load()
                documents.extend(docs)
    return documents

def split_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def embed_and_store():
    docs = load_documents()
    chunks = split_documents(docs)

    # ✅ Pass API key explicitly from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment!")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vectorstore.persist()
    print(f"Ingested and embedded {len(chunks)} chunks.")
