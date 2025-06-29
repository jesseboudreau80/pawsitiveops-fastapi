import os
import requests
from backend.app.core.config import settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

SAM_PROMPT = """
You are SAM the Safety Dog, a cheerful mascot who helps pet center staff stay safe. 
Always speak with encouragement and warmth, and end each message on a positive note!

Context:
{context}

User:
{question}
"""

def get_sam_response(message: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai")

    try:
        # Load vector store
        vectorstore = Chroma(
            persist_directory="backend/app/vector_store",
            embedding_function=OpenAIEmbeddings()
        )

        retriever = vectorstore.as_retriever()

        prompt = PromptTemplate(
            template=SAM_PROMPT,
            input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.4,
                openai_api_key=settings.OPENAI_API_KEY
            ),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

        result = qa.invoke({"query": message})
        return result["result"]

    except Exception as e:
        return f"SAM encountered an error: {str(e)}"
