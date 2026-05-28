# 🧠 DocMind — RAG-Based Intelligent Document Q&A System

A production-style Retrieval-Augmented Generation (RAG) system that lets you upload any PDF or TXT document and ask questions about it in natural language. Built with LangChain, OpenAI, FAISS, and Streamlit.

---

## 🗂 Project Structure

```
rag_project/
├── app.py                  # Streamlit web UI
├── test_rag.py             # CLI test script (no UI needed)
├── requirements.txt        # All dependencies
├── sample_docs/
│   └── sample.txt          # Sample document for testing
├── data/
│   └── uploads/            # Auto-created when you upload files
└── src/
    ├── __init__.py
    ├── rag_pipeline.py     # Core RAG logic (load → chunk → embed → retrieve → generate)
    └── utils.py            # Evaluation metrics (Precision@K, MRR)
```

---

## ⚙️ Setup (Do This Once)

### 1. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get an OpenAI API key
- Go to: https://platform.openai.com/api-keys
- Create a new secret key
- **You enter it directly in the app's sidebar** — no `.env` file needed

---

## 🚀 Run the App

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501`

**Steps in the UI:**
1. Enter your OpenAI API key in the sidebar
2. Upload one or more PDF or TXT files
3. Click **"Build Knowledge Base"**
4. Ask questions in the chat box!

---

## 🧪 Quick CLI Test (No Browser Needed)

```bash
python test_rag.py
```

This uses the sample document (`sample_docs/sample.txt`) and lets you ask questions in the terminal. Great for quick testing.

---

## 🔬 How It Works — RAG Pipeline

```
Your Documents
     │
     ▼
[1] LOAD         PyPDFLoader / TextLoader reads files
     │
     ▼
[2] CHUNK        RecursiveCharacterTextSplitter
                 chunk_size=500, overlap=75 tokens
     │
     ▼
[3] EMBED        OpenAI text-embedding-3-small
                 Each chunk → 1536-dim vector
     │
     ▼
[4] INDEX        FAISS vector store (L2 distance)
                 Fast approximate nearest neighbor search
     │
     ▼
[5] QUERY TIME:
    User Question → Embed → Similarity Search → Top-K Chunks
     │
     ▼
[6] GENERATE     GPT-3.5-turbo / GPT-4o-mini
                 System prompt + retrieved context + question → Answer
```

---

## 📊 Evaluation Metrics

The project includes lightweight retrieval evaluation in `src/utils.py`:

| Metric | What it measures |
|--------|-----------------|
| **Precision@3** | Fraction of top-3 chunks that contain relevant content |
| **MRR** | How high up the first relevant chunk appears (1/rank) |

Run evaluation via `test_rag.py` → Step 4.

---

## 💰 API Cost Estimate

| Model | Cost per ~50 questions on a 20-page PDF |
|-------|----------------------------------------|
| gpt-3.5-turbo | ~$0.05 |
| gpt-4o-mini | ~$0.15 |
| gpt-4o | ~$1.50 |

**Recommendation:** Use `gpt-3.5-turbo` for testing, `gpt-4o-mini` for demos.

---

## 🤝 Interview Talking Points

Be ready to explain these concepts clearly:

**"Why FAISS over a regular database?"**
> FAISS performs vector similarity search (cosine/L2 distance) across millions of embeddings in milliseconds. A traditional DB does exact keyword matching — it can't understand semantic meaning. FAISS finds chunks that are *semantically similar* even if no exact keywords match.

**"Why chunking with overlap?"**
> If a sentence spans a chunk boundary, it might get cut off and lose meaning. 15% overlap ensures boundary content is captured in at least one chunk. Smaller chunks give more precise retrieval; larger chunks give more context.

**"How do you prevent hallucination?"**
> The system prompt strictly instructs the LLM to answer ONLY from the retrieved context. If the answer isn't there, it says so explicitly. This grounds responses in real document content.

**"What's the difference between RAG and fine-tuning?"**
> Fine-tuning bakes knowledge into model weights — expensive, static, and hard to update. RAG keeps knowledge external in a vector DB — cheap to update (just re-index), transparent (can cite sources), and dynamically updatable without retraining.

**"How would you improve this at scale?"**
> 1. Replace FAISS with Pinecone/Weaviate for distributed storage
> 2. Add re-ranking (cross-encoder) after retrieval for better precision
> 3. Use LangSmith for prompt evaluation and observability
> 4. Add hybrid search (BM25 + dense retrieval) for better recall
> 5. Implement query expansion/HyDE for better embedding alignment

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-3.5-turbo / GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |
| UI | Streamlit |
| Language | Python 3.10+ |
