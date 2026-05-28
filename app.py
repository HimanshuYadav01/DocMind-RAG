import streamlit as st
import os
import time
from src.rag_pipeline import RAGPipeline
from src.utils import format_sources

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] h2 { 
    font-family: 'IBM Plex Mono', monospace; 
    color: #00e5a0 !important;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}

/* Main background */
.main { background: #f7f6f2; }

/* Title */
.title-block {
    background: #0d0d0d;
    color: #00e5a0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    padding: 1.5rem 2rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}
.title-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    color: #888;
    font-weight: 300;
    margin-top: 0.3rem;
}

/* Answer box */
.answer-box {
    background: #ffffff;
    border-left: 4px solid #00e5a0;
    padding: 1.5rem 1.8rem;
    border-radius: 0 8px 8px 0;
    font-size: 1rem;
    line-height: 1.7;
    color: #1a1a1a;
    margin: 1rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* Source cards */
.source-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #444;
}
.source-tag {
    background: #0d0d0d;
    color: #00e5a0;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.7rem;
    margin-right: 8px;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: #f0fdf6;
    border: 1px solid #00e5a0;
    color: #00a070;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}

/* Chat history */
.chat-user {
    background: #0d0d0d;
    color: #f0f0f0;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.chat-assistant {
    background: #f0fdf6;
    border-left: 3px solid #00e5a0;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #1a1a1a;
}

/* Input styling */
.stTextInput input, .stTextArea textarea {
    font-family: 'IBM Plex Sans', sans-serif !important;
    border: 1.5px solid #d0d0d0 !important;
    border-radius: 6px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,0.15) !important;
}

/* Button */
.stButton button {
    background: #0d0d0d !important;
    color: #00e5a0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
}
.stButton button:hover {
    background: #222 !important;
}

.divider { 
    border: none; 
    border-top: 1px solid #e0e0e0; 
    margin: 1.5rem 0; 
}
</style>
""", unsafe_allow_html=True)

# ── State Init ────────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ CONFIGURATION")
    st.markdown("---")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your key from platform.openai.com"
    )

    model_choice = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
        help="llama-3.1 is fastest; llama-3.3 is most capable"
    )

    chunk_size = st.slider("Chunk Size (tokens)", 200, 1000, 500, 50,
        help="Smaller = more precise retrieval. Larger = more context per chunk.")

    top_k = st.slider("Top-K Retrieval", 1, 8, 3,
        help="How many document chunks to retrieve per query")

    st.markdown("---")
    st.markdown("## 📂 UPLOAD DOCUMENTS")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("🔨 Build Knowledge Base") and uploaded_files and api_key:
        with st.spinner("Indexing documents..."):
            try:
                # Save uploads to temp
                os.makedirs("data/uploads", exist_ok=True)
                file_paths = []
                for f in uploaded_files:
                    path = f"data/uploads/{f.name}"
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    file_paths.append(path)

                st.session_state.rag = RAGPipeline(
                    api_key=api_key,
                    model=model_choice,
                    chunk_size=chunk_size,
                    top_k=top_k
                )
                stats = st.session_state.rag.build_index(file_paths)
                st.session_state.docs_loaded = True
                st.session_state.chat_history = []
                st.success(f"✅ Indexed {stats['chunks']} chunks from {stats['docs']} document(s)")
                st.markdown(f"""
                <span class='metric-chip'>📄 {stats['docs']} docs</span>
                <span class='metric-chip'>🧩 {stats['chunks']} chunks</span>
                <span class='metric-chip'>📐 {chunk_size} chunk size</span>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.docs_loaded:
        st.markdown("<span style='color:#00e5a0'>● Knowledge base ready</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#888'>○ No documents indexed</span>", unsafe_allow_html=True)

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='title-block'>
  🧠 DocMind
  <div class='title-sub'>RAG-powered Document Intelligence — Ask anything about your documents</div>
</div>
""", unsafe_allow_html=True)

# Show chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        st.markdown(f"<div class='chat-user'>🧑 {turn['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-assistant'>🤖 {turn['answer']}</div>", unsafe_allow_html=True)
        if turn.get("sources"):
            with st.expander("📎 Sources used"):
                for s in turn["sources"]:
                    st.markdown(f"""<div class='source-card'>
                        <span class='source-tag'>SRC</span>{s['source']} — Page {s.get('page','?')} | 
                        Score: {s.get('score', 0):.3f}<br>
                        <span style='color:#666'>…{s['snippet']}…</span>
                    </div>""", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Query input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the key findings in this document?",
        label_visibility="collapsed"
    )
with col2:
    ask_btn = st.button("ASK →")

if (ask_btn or query) and query:
    if not st.session_state.docs_loaded:
        st.warning("⚠️ Please upload documents and build the knowledge base first (sidebar).")
    elif not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Retrieving and reasoning..."):
            start = time.time()
            result = st.session_state.rag.query(
                query,
                chat_history=st.session_state.chat_history,
                top_k = top_k
            )
            elapsed = time.time() - start

        st.markdown(f"""
        <div class='answer-box'>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        st.markdown(f"""
        <span class='metric-chip'>⏱ {elapsed:.2f}s</span>
        <span class='metric-chip'>🔍 {result['chunks_retrieved']} chunks retrieved</span>
        <span class='metric-chip'>🪙 ~{result.get('tokens_used', '?')} tokens</span>
        """, unsafe_allow_html=True)

        # Sources
        if result["sources"]:
            with st.expander("📎 View source chunks"):
                for i, s in enumerate(result["sources"]):
                    st.markdown(f"""<div class='source-card'>
                        <span class='source-tag'>#{i+1}</span><b>{s['source']}</b> — Page {s.get('page','?')} | 
                        Similarity: {s.get('score', 0):.3f}<br><br>
                        <span style='color:#555'>{s['snippet']}</span>
                    </div>""", unsafe_allow_html=True)

        # Save to history
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })

# Empty state
if not st.session_state.docs_loaded and not st.session_state.chat_history:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #aaa;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>📄</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size:1rem;'>
            Upload documents in the sidebar to get started
        </div>
        <div style='font-size:0.85rem; margin-top:0.5rem;'>
            Supports PDF and TXT files
        </div>
    </div>
    """, unsafe_allow_html=True)
