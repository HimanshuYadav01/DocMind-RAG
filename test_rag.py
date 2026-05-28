"""
test_rag.py — Quick CLI test for the RAG pipeline (no Streamlit needed)
Usage: python test_rag.py
"""

import os
from src.rag_pipeline import RAGPipeline
from src.utils import format_sources, evaluate_retrieval

def main():
    print("=" * 60)
    print("  DocMind RAG Pipeline — CLI Test")
    print("=" * 60)

    # ── 1. Get API key ──
    api_key = os.getenv("OPENAI_API_KEY") or input("\nEnter your OpenAI API key: ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        return

    # ── 2. Init pipeline ──
    print("\n[1/4] Initialising pipeline...")
    rag = RAGPipeline(api_key=api_key, model="gpt-3.5-turbo", chunk_size=500, top_k=3)
    print("      ✓ Pipeline ready")

    # ── 3. Load sample document ──
    print("\n[2/4] Loading sample document...")
    sample_path = "sample_docs/sample.txt"
    if not os.path.exists(sample_path):
        print(f"      ✗ Sample doc not found at {sample_path}")
        print("        Create a sample_docs/sample.txt file with some text content.")
        return

    stats = rag.build_index([sample_path])
    print(f"      ✓ Indexed {stats['chunks']} chunks from {stats['docs']} document(s)")

    # ── 4. Interactive Q&A loop ──
    print("\n[3/4] Starting interactive Q&A (type 'quit' to exit)")
    print("-" * 60)
    chat_history = []

    while True:
        query = input("\n❓ Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        print("\n⏳ Retrieving and reasoning...\n")
        result = rag.query(query, chat_history=chat_history)

        print(f"🤖 Answer:\n{result['answer']}")
        print(f"\n📊 Metrics: {result['chunks_retrieved']} chunks retrieved | ~{result['tokens_used']} tokens used")
        print(f"\n📎 Sources:\n{format_sources(result['sources'])}")

        chat_history.append({"question": query, "answer": result["answer"]})

    # ── 5. Run evaluation ──
    print("\n[4/4] Running retrieval evaluation (Precision@3, MRR)...")
    test_cases = [
        {"query": "What is the main topic?", "keywords": ["main", "topic", "about"]},
        {"query": "Summarize the key points", "keywords": ["key", "point", "summary", "important"]},
    ]
    try:
        metrics = evaluate_retrieval(rag, test_cases)
        print(f"      Precision@3 : {metrics['avg_precision_at_3']:.3f}")
        print(f"      MRR         : {metrics['avg_mrr']:.3f}")
        print(f"      Test cases  : {metrics['num_test_cases']}")
    except Exception as e:
        print(f"      Eval skipped: {e}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
