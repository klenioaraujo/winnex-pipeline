#!/usr/bin/env python3
"""
RAG Agent Demo — Multi-Agent Retrieval-Augmented Generation
=============================================================
Demonstrates the config-driven RAG pipeline with:
  - Autonomous dataset loading (Kaggle / HF / local JSON)
  - Configurable embedding model (MiniLM, BGE, Qwen, custom)
  - Configurable search backend (MadhavaCore, HMC, MadHybrid)
  - MMR diversity reranking
  - Deterministic bound verification
  - LLM generation (mock / Anthropic / OpenAI)

Usage:
    cd winnex_pipeline
    python examples/rag_agent_demo.py              # uses existing dataset
    python examples/rag_agent_demo.py --fetch      # downloads from Kaggle first
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from winnex_pipeline.rag import RAGAgent


def list_datasets():
    """Show available datasets and models from config."""
    from winnex_pipeline.config import load_config
    print("Available datasets:")
    print("  Kaggle:    rmisra/news-category-dataset (210K news articles)")
    print("  HuggingFace: squad, wiki_dpr, ag_news")
    print("\nAvailable embedding models:")
    for m in ['minilm', 'mpnet', 'bge', 'openai_small', 'gpt2', 'sift']:
        cfg = load_config(f'config/models/{m}.json')
        mod = cfg['model']
        print(f"  {m:>12}: {mod['name']:<30} {mod['dimension']}D")
    print("\nAvailable search methods:")
    print("  madhava     — QR-JL cascade (guaranteed bounds, default)")
    print("  madhybrid   — IVF clustering + Madhava per cell")
    print("  hmc         — Riemannian HMC navigation (requires PyTorch)")
    print("  auto        — auto-select based on corpus size")


def main():
    parser = argparse.ArgumentParser(description="Winnex RAG Agent Demo")
    parser.add_argument('--fetch', action='store_true',
                        help='Download dataset from Kaggle')
    parser.add_argument('--model', type=str, default=None,
                        help='Embedding model override (e.g. all-mpnet-base-v2)')
    parser.add_argument('--method', type=str, default=None,
                        help='Search method: madhava, madhybrid, hmc, auto')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of chunks to retrieve')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets and models')
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # ── 1. Create RAG Agent from config ──────────────────────
    print("=" * 60)
    print("Winnex RAG Agent — Demo")
    print("=" * 60)

    agent = RAGAgent(config_path="config/rag.json", method=args.method)

    if args.model:
        agent.cfg['embedding_model']['name'] = args.model

    # ── 2. Load dataset ─────────────────────────────────────
    data_path = "data/News_Category_Dataset_v3.json"
    if args.fetch and not os.path.exists(data_path):
        print("Downloading News Category Dataset from Kaggle...")
        agent.load_dataset(source='kaggle')
    elif os.path.exists(data_path):
        print(f"Loading local dataset: {data_path}")
        agent.cfg['dataset']['path'] = data_path
        agent.cfg['dataset']['fields'] = {
            "title": "headline",
            "content": "short_description",
            "category": "category",
            "metadata": ["headline", "category"]
        }

        # Load from JSONL
        documents = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    documents.append({
                        "content": record.get("short_description", record.get("headline", "")),
                        "metadata": {
                            "headline": record.get("headline", ""),
                            "category": record.get("category", ""),
                        }
                    })
        # Use 500 docs for demo speed
        agent.load_custom_documents(documents[:500])
    else:
        print("No dataset found. Using demo documents.")
        agent.load_custom_documents([
            {"content": "Deep learning uses neural networks with many layers to learn hierarchical representations of data.",
             "metadata": {"topic": "AI"}},
            {"content": "Vector search finds similar items by comparing embeddings in high-dimensional space.",
             "metadata": {"topic": "search"}},
            {"content": "The Winnex Madhava Pipeline provides deterministic vector search with mathematically guaranteed upper bounds.",
             "metadata": {"topic": "Winnex"}},
            {"content": "Cauchy-Schwarz inequality states that |<u,v>| ≤ ||u|| · ||v|| for any vectors u, v in an inner product space.",
             "metadata": {"topic": "mathematics"}},
            {"content": "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation.",
             "metadata": {"topic": "RAG"}},
            {"content": "HNSW is a graph-based approximate nearest neighbor search algorithm used in many vector databases.",
             "metadata": {"topic": "search"}},
            {"content": "QR decomposition factorizes a matrix A into an orthogonal matrix Q and an upper triangular matrix R.",
             "metadata": {"topic": "mathematics"}},
            {"content": "Sentence transformers produce semantically meaningful embeddings for text that enable similarity search.",
             "metadata": {"topic": "NLP"}},
        ] * 10)

    # ── 3. Build index ──────────────────────────────────────
    print("\nBuilding search index...")
    agent.build_index()
    agent.info()

    # ── 4. Run queries ──────────────────────────────────────
    queries = [
        "What is vector search?",
        "How does Winnex guarantee search correctness?",
        "Explain the mathematical bounds used",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        result = agent.query(query, k=args.k, rerank=True, return_context=True)

        print(f"\n📋 Retrieved {result['n_chunks']} chunks "
              f"({result['timing_ms']['retrieve']:.1f}ms)")
        for i, c in enumerate(result.get('context', [])):
            cat = c.get('metadata', {}).get('category', '')
            head = c.get('metadata', {}).get('headline', '')
            tag = f" [{cat}]" if cat else ""
            tag += f" \"{head[:60]}\"" if head else ""
            print(f"  [{i+1}] {c['text'][:100]}...{tag}")

        print(f"\n🤖 Answer ({result['timing_ms']['generate']:.1f}ms):")
        print(f"  {result['answer'][:300]}")

        print(f"\n⏱ Total: {result['timing_ms']['total']:.1f}ms")

    print(f"\n{'='*60}")
    print("Demo complete. Configure generator in config/rag.json")
    print("to replace mock responses with real LLM answers.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
