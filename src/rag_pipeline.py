"""
RAG Pipeline — Using Groq (free) for LLM + HuggingFace for embeddings (free, local)
No API payment needed!
"""

import os
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage

SYSTEM_PROMPT = """You are DocMind, an intelligent document Q&A assistant.
Answer STRICTLY based on the provided document context.
Rules:
1. Answer only from the retrieved context. Do NOT hallucinate.
2. If the answer is not in the context, say: "I couldn't find this in the uploaded documents."
3. Be concise but complete. Use bullet points for multi-part answers.
4. Cite which document/section your answer comes from when possible.

Context from documents:
{context}
"""

class RAGPipeline:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                 chunk_size: int = 500, top_k: int = 3):
        self.api_key = api_key
        # self.llm = ChatGroq(groq_api_key=api_key, model_name=model)
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.model_name = model
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.vectorstore = None

        # Groq LLM (free)
        self.llm = ChatGroq(
            model_name=model,
            temperature=0.2,
            max_tokens=1000,
            groq_api_key=api_key
        )

        # Free local embeddings (no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        all_docs = []
        for path in file_paths:
            ext = os.path.splitext(path)[-1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    continue
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        return all_docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        return chunks

    def build_index(self, file_paths: List[str]) -> Dict[str, int]:
        docs = self.load_documents(file_paths)
        if not docs:
            raise ValueError("No documents could be loaded.")
        chunks = self.chunk_documents(docs)
        if not chunks:
            raise ValueError("No text chunks extracted.")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return {
            "docs": len(set(d.metadata.get("source", "") for d in docs)),
            "chunks": len(chunks),
            "pages": len(docs)
        }

    def retrieve(self, query: str, k : int = None) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise RuntimeError("Knowledge base not built.")
        search_k = k if k is not None else self.top_k
        results = self.vectorstore.similarity_search_with_score(query, k=search_k)
        # results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        retrieved = []
        for doc, score in results:
            retrieved.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "score": float(1 - score),
                "snippet": doc.page_content[:200].replace("\n", " ").strip()
            })
        return retrieved

    def query(self, question: str, chat_history: List[Dict] = None, top_k: int = None) -> Dict[str, Any]:
        # sources = self.retrieve(question)
        k = top_k if top_k is not None else self.top_k
        sources = self.retrieve(question, k=k)
        context_parts = []
        for i, s in enumerate(sources):
            context_parts.append(
                f"[Chunk {i+1} | Source: {s['source']} | Page: {s['page']}]\n{s['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        messages = []
        if chat_history:
            for turn in chat_history[-4:]:
                messages.append(HumanMessage(content=turn["question"]))
                messages.append(AIMessage(content=turn["answer"]))
        messages.append(HumanMessage(content=question))

        system_msg = SYSTEM_PROMPT.format(context=context)
        full_messages = [SystemMessage(content=system_msg)] + messages

        response = self.llm.invoke(full_messages)
        answer = response.content

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(sources),
            "tokens_used": "N/A",
            "context_length": len(context)
        }
