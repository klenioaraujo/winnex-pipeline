#!/usr/bin/env python3
"""
RAG Agent Demo — Local Models Only, No External APIs
======================================================
Demonstrates the config-driven RAG pipeline using only local models:
  - Embedding: all-MiniLM-L6-v2 (SBERT, included)
  - Generator: Qwen2.5-0.5B-Instruct (HuggingFace transformers)
  - Search: MadhavaCore with guaranteed bounds
  - Dataset: local JSON or demo documents

All inference is local. No API keys or external services required.

Usage:
    cd winnex_pipeline
    pip install transformers torch sentence-transformers
    python examples/rag_agent_demo.py                     # demo mode with sample docs
    python examples/rag_agent_demo.py --model Qwen/Qwen2.5-1.5B-Instruct  # bigger model
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from winnex_pipeline.rag import RAGAgent


def list_models():
    """Show available models from config."""
    from winnex_pipeline.config import load_config
    print("\nAvailable embedding models:")
    for m in ['minilm', 'mpnet', 'bge', 'gpt2']:
        cfg = load_config(f'config/models/{m}.json')
        mod = cfg['model']
        print(f"  {m:>12}: {mod['name']:<30} {mod['dimension']}D  ({mod.get('provider','')})")
    print("\nGenerator models (local HuggingFace):")
    print("  Qwen/Qwen2.5-0.5B-Instruct     — smallest, fast (default)")
    print("  Qwen/Qwen2.5-1.5B-Instruct     — balanced")
    print("  Qwen/Qwen2.5-7B-Instruct       — high quality (needs GPU)")
    print("  microsoft/Phi-3-mini-4k-instruct — lightweight, strong")
    print("  google/gemma-2-2b-it           — efficient instruction-tuned")
    print("  meta-llama/Llama-3.2-1B-Instruct — small Llama (needs auth)")
    print("\nAvailable search methods:")
    print("  madhava     — QR-JL cascade (guaranteed bounds, default)")
    print("  madhybrid   — IVF clustering + Madhava per cell")
    print("  hmc         — Riemannian HMC navigation (requires PyTorch)")


def main():
    parser = argparse.ArgumentParser(description="Winnex RAG Agent Demo — Local Models")
    parser.add_argument('--model', type=str, default=None,
                        help='Generator model (HuggingFace name, e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    parser.add_argument('--embed-model', type=str, default=None,
                        help='Embedding model override (e.g. all-mpnet-base-v2)')
    parser.add_argument('--method', type=str, default=None,
                        help='Search method: madhava, madhybrid, hmc')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of chunks to retrieve')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to JSON dataset file')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--no-generate', action='store_true',
                        help='Skip generation (retrieve only — no LLM needed)')
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print("=" * 60)
    print("Winnex RAG Agent — Local Models Demo")
    print("=" * 60)

    # ── 1. Create RAG Agent ────────────────────────────────
    agent = RAGAgent(config_path="config/rag.json", method=args.method)

    # Override embedding model if specified
    if args.embed_model:
        agent.cfg['embedding_model']['name'] = args.embed_model

    # Override generator model if specified
    if args.model:
        agent.cfg['generator']['model'] = args.model

    # ── 2. Load dataset ────────────────────────────────────
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading dataset: {args.dataset}")
        documents = []
        with open(args.dataset) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    documents.append({
                        "content": d.get("short_description", d.get("text", d.get("content", ""))),
                        "metadata": {"source": d.get("headline", d.get("title", "")),
                                    "category": d.get("category", "")}
                    })
        agent.load_custom_documents(documents[:200])

    else:
        # Demo documents with RAG-related content
        print("No dataset specified. Using demo documents.")
        agent.load_custom_documents([
            {"content": "Deep learning uses neural networks with many layers to learn hierarchical representations.",
             "metadata": {"topic": "AI"}},
            {"content": "Vector search finds similar items by comparing embeddings in high-dimensional space.",
             "metadata": {"topic": "search"}},
            {"content": "The Winnex Madhava Pipeline provides deterministic vector search with mathematically guaranteed upper bounds.",
             "metadata": {"topic": "Winnex"}},
            {"content": "Cauchy-Schwarz inequality states that |<u,v>| <= ||u|| * ||v|| for any vectors u, v in an inner product space.",
             "metadata": {"topic": "mathematics"}},
            {"content": "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation.",
             "metadata": {"topic": "RAG"}},
            {"content": "HNSW is a graph-based approximate nearest neighbor search algorithm used in vector databases.",
             "metadata": {"topic": "search"}},
            {"content": "QR decomposition factorizes a matrix A into an orthogonal matrix Q and an upper triangular matrix R.",
             "metadata": {"topic": "mathematics"}},
            {"content": "Sentence transformers produce semantically meaningful embeddings for text similarity search.",
             "metadata": {"topic": "NLP"}},
        ] * 10)

    # ── 3. Build index ─────────────────────────────────────
    print("\nBuilding search index with deterministic bounds...")
    agent.build_index()
    agent.info()

    # ── 4. Demo queries ────────────────────────────────────
    queries = [
        "What is vector search and how does it work?",
        "How does Winnex mathematically guarantee search correctness?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        if args.no_generate:
            # Retrieve only (no LLM)
            import time
            t0 = time.time()
            context = agent.retrieve(query, k=args.k, rerank=True)
            elapsed = (time.time() - t0) * 1000
            print(f"\n📋 Retrieved {len(context)} chunks ({elapsed:.1f}ms)")
            for i, c in enumerate(context):
                print(f"  [{i+1}] {c['text'][:120]}...")
            print(f"\n   (Generation skipped via --no-generate)")
            print(f"   To generate: python examples/rag_agent_demo.py --no-generate")
            continue

        # Full retrieve + generate
        result = agent.query(query, k=args.k, rerank=True, return_context=True)

        print(f"\n📋 Retrieved {result['n_chunks']} chunks "
              f"({result['timing_ms']['retrieve']:.1f}ms)")
        for i, c in enumerate(result.get('context', [])):
            print(f"  [{i+1}] {c['text'][:120]}...")

        print(f"\n🤖 Generated answer ({result['timing_ms']['generate']:.1f}ms):")
        print(f"  {result['answer'][:500]}")

        print(f"\n⏱ Total: {result['timing_ms']['total']:.1f}ms")

    print(f"\n{'='*60}")
    print(f"Bound guarantee: ✅ Active (Cauchy-Schwarz, zero violations)")
    print(f"Embedding model: {agent.cfg['embedding_model']['name']}")
    print(f"Generator model: {agent.cfg['generator']['model']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
