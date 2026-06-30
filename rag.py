"""
Winnex RAG — Multi-Agent Retrieval-Augmented Generation
========================================================
Native agent pipeline with:
  - Config-driven datasets (Kaggle, HuggingFace, local)
  - Configurable embedding models (SBERT, Qwen, custom)
  - Multiple search backends (MadhavaCore, MadHybrid, HMC)
  - MMR diversity reranking
  - LLM integration (Anthropic, OpenAI, local)
  - Deterministic bound guarantees per retrieval

Usage:
    from winnex_pipeline.rag import RAGAgent

    agent = RAGAgent(config_path="config/rag.json")
    agent.load_dataset()              # from Kaggle/HF/local
    agent.build_index()               # encode + Madhava index
    answer = agent.query("What is...?")  # retrieve + generate
"""
import os, json, time, math
import numpy as np

from .config import load_config


# ═══════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════

def recursive_chunk_text(text, chunk_size=512, overlap=64,
                         separators=None):
    """Recursive chunking similar to LangChain's RecursiveCharacterTextSplitter."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Find best separator to break at
            best_pos = end
            for sep in separators:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > start:
                    best_pos = pos + len(sep)
                    break
            end = best_pos
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def chunk_documents(documents, config):
    """Chunk a list of documents based on config."""
    chunk_cfg = config.get('chunking', {})
    if not chunk_cfg.get('enabled', False):
        return [{"text": d.get("content", d.get("text", "")),
                 "metadata": {k: d.get(k) for k in
                              chunk_cfg.get('fields', {}).get('metadata', []) if k in d}}
                for d in documents]

    chunk_size = chunk_cfg.get('chunk_size', 512)
    overlap = chunk_cfg.get('chunk_overlap', 64)
    separators = chunk_cfg.get('separators', None)
    max_chunks = chunk_cfg.get('max_chunks_per_doc', 20)

    all_chunks = []
    for doc in documents:
        text = doc.get("content", doc.get("text", ""))
        metadata = {k: doc.get(k) for k in
                    chunk_cfg.get('fields', {}).get('metadata', []) if k in doc}

        doc_chunks = recursive_chunk_text(text, chunk_size, overlap, separators)
        for i, chunk_text in enumerate(doc_chunks[:max_chunks]):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "chunk_idx": i}
            })
    return all_chunks


# ═══════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════

def load_dataset_from_config(config):
    """Load dataset from Kaggle, HuggingFace, or local JSON."""
    ds_cfg = config.get('dataset', {})
    source = ds_cfg.get('source', 'local')
    max_docs = ds_cfg.get('max_documents', 50000)

    if source == 'kaggle':
        return _load_kaggle_dataset(ds_cfg, max_docs)
    elif source == 'huggingface':
        return _load_huggingface_dataset(ds_cfg, max_docs)
    elif source == 'json':
        return _load_json_dataset(ds_cfg, max_docs)
    elif source == 'local':
        return _load_json_dataset(ds_cfg, max_docs)
    else:
        raise ValueError(f"Unknown dataset source: {source}")


def _load_kaggle_dataset(ds_cfg, max_docs):
    """Load dataset from Kaggle."""
    name = ds_cfg.get('name', '')
    path = ds_cfg.get('path', 'data/rag_corpus.json')
    fields = ds_cfg.get('fields', {})

    # Check if already downloaded
    if os.path.exists(path):
        return _load_json_dataset(ds_cfg, max_docs)

    # Try to download
    print(f"Downloading Kaggle dataset: {name}...", flush=True)
    try:
        import subprocess
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', name, '-p', 'data/', '--unzip'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle download failed: {result.stderr}")

        # Find the downloaded file
        import glob
        json_files = glob.glob('data/*.json')
        if json_files:
            path = json_files[0]
            ds_cfg['path'] = path

        return _load_json_dataset(ds_cfg, max_docs)
    except Exception as e:
        print(f"Kaggle download error: {e}", flush=True)
        fallback = ds_cfg.get('fallback', {})
        if fallback.get('source') == 'huggingface':
            ds_cfg = fallback
            return _load_huggingface_dataset(ds_cfg, max_docs)
        raise


def _load_huggingface_dataset(ds_cfg, max_docs):
    """Load dataset from HuggingFace datasets."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    name = ds_cfg.get('name', 'squad')
    split = ds_cfg.get('split', 'train')

    print(f"Loading HuggingFace dataset: {name} [{split}]...", flush=True)
    ds = hf_load(name, split=split, streaming=False)

    documents = []
    for i, example in enumerate(ds):
        if i >= max_docs:
            break
        text = example.get('text') or example.get('context') or \
               example.get('content') or str(example)
        documents.append({
            'content': text,
            'metadata': {k: str(v) for k, v in example.items()
                         if k not in ('text', 'context', 'content')}
        })
    print(f"  Loaded {len(documents)} documents", flush=True)
    return documents


def _load_json_dataset(ds_cfg, max_docs):
    """Load dataset from local JSON/JSONL file."""
    path = ds_cfg.get('path', 'data/rag_corpus.json')
    fields = ds_cfg.get('fields', {})

    if fields and 'title' in fields:
        title_field = fields['title']
        content_field = fields.get('content', 'text')
        cat_field = fields.get('category', None)
    else:
        title_field = 'title'
        content_field = 'text'
        cat_field = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    documents = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = (record.get(content_field) or
                        record.get(title_field) or
                        str(record))
                doc = {
                    'content': text,
                    'metadata': {}
                }
                if cat_field and cat_field in record:
                    doc['metadata']['category'] = record[cat_field]
                if title_field and title_field in record:
                    doc['metadata']['title'] = record[title_field]
                documents.append(doc)
            except json.JSONDecodeError:
                documents.append({
                    'content': line,
                    'metadata': {'source': path}
                })

    print(f"  Loaded {len(documents)} documents from {path}", flush=True)
    return documents


# ═══════════════════════════════════════════════════════════════
# MMR DIVERSITY RERANKING
# ═══════════════════════════════════════════════════════════════

def mmr_rerank(doc_embeddings, query_embedding, indices, scores,
               lambda_param=0.5, top_k=5):
    """
    Maximum Marginal Relevance for diversity.
    Balances relevance (query similarity) and diversity (doc dissimilarity).
    """
    if len(indices) <= top_k:
        return indices[:top_k], scores[:top_k]

    selected = []
    candidate_pool = list(range(len(indices)))
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

    for _ in range(top_k):
        if not candidate_pool:
            break

        best_score = -float('inf')
        best_idx = None

        for ci in candidate_pool:
            doc_idx = indices[ci]
            doc_emb = doc_embeddings[doc_idx]
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-9)

            sim_to_query = float(doc_norm @ query_norm)
            sim_to_selected = max([
                float(doc_norm @ (doc_embeddings[s] / (np.linalg.norm(doc_embeddings[s]) + 1e-9)))
                for s in selected
            ], default=0.0)

            mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = ci

        if best_idx is not None:
            selected.append(indices[best_idx])
            candidate_pool.remove(best_idx)

    return selected, None


# ═══════════════════════════════════════════════════════════════
# GENERATOR (LLM WRAPPER)
# ═══════════════════════════════════════════════════════════════

def generate_answer(query, context_chunks, config):
    """
    Generate an answer using the configured LLM provider.

    Supports: anthropic, openai, huggingface, and mock (for testing).
    """
    gen_cfg = config.get('generator', {})
    provider = gen_cfg.get('provider', 'mock')
    api_key_env = gen_cfg.get('api_key_env', 'ANTHROPIC_API_KEY')
    model = gen_cfg.get('model', 'claude-sonnet-4-6')
    max_tokens = gen_cfg.get('max_tokens', 1024)
    temperature = gen_cfg.get('temperature', 0.3)
    template = gen_cfg.get('prompt_template',
                           "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")

    context_text = "\n\n".join([
        f"[{i+1}] {chunk['text'] if isinstance(chunk, dict) else chunk}"
        for i, chunk in enumerate(context_chunks)
    ])
    prompt = template.format(context=context_text, query=query)

    if provider == 'mock':
        return _generate_mock(query, context_chunks, config)

    elif provider == 'anthropic':
        return _generate_anthropic(prompt, model, max_tokens,
                                   temperature, api_key_env, gen_cfg)

    elif provider == 'openai':
        return _generate_openai(prompt, model, max_tokens,
                                temperature, api_key_env, gen_cfg)

    elif provider == 'mock':
        return f"[MOCK] Response based on {len(context_chunks)} chunks. Query: {query[:50]}..."

    else:
        raise ValueError(f"Unknown generator provider: {provider}")


def _generate_anthropic(prompt, model, max_tokens, temperature, api_key_env, gen_cfg):
    try:
        import anthropic
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return "[WARN] No API key found. Set " + api_key_env

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except ImportError:
        return "[WARN] anthropic not installed. pip install anthropic"


def _generate_openai(prompt, model, max_tokens, temperature, api_key_env, gen_cfg):
    try:
        import openai
        api_key = os.environ.get(api_key_env, os.environ.get('OPENAI_API_KEY'))
        if not api_key:
            return "[WARN] No OpenAI API key found"

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except ImportError:
        return "[WARN] openai not installed. pip install openai"

def _generate_mock(query, context_chunks, config):
    """Mock generator for testing — returns structured context summary."""
    chunks = []
    for c in context_chunks[:3]:
        text = c['text'] if isinstance(c, dict) else c
        chunks.append(text[:120])
    return (
        f"[Mock Answer based on {len(context_chunks)} retrieved chunks]\n\n"
        f"**Query**: {query[:80]}...\n\n"
        f"**Top Sources**:\n" +
        "\n".join(f"  {i+1}. {c[:100]}..." for i, c in enumerate(chunks)) +
        "\n\n[Configure a generator provider in config/rag.json]"
    )


# ═══════════════════════════════════════════════════════════════
# RAG AGENT
# ═══════════════════════════════════════════════════════════════

class RAGAgent:
    """
    Multi-agent RAG pipeline with config-driven datasets, models, and search.

    Args:
        config_path: path to JSON config (default: config/rag.json)
        method: search method override

    Usage:
        agent = RAGAgent()
        agent.load_dataset()
        agent.build_index()
        answer = agent.query("What is deep learning?")
    """

    def __init__(self, config_path="config/rag.json", method=None):
        self.cfg = load_config(config_path)
        self.method = method or self.cfg.get('search', {}).get('method', 'madhava')
        self.documents = []
        self.chunks = []
        self.pipeline = None
        self.doc_embeddings = None
        self.is_ready = False

    # ── Dataset ─────────────────────────────────────────────────

    def load_dataset(self, source=None):
        """
        Load documents from the configured dataset source.

        Args:
            source: override source ('kaggle', 'huggingface', 'json', 'local')
        """
        if source:
            self.cfg['dataset']['source'] = source

        print(f"Loading dataset (source={self.cfg['dataset']['source']})...")
        docs = load_dataset_from_config(self.cfg)
        self.documents = docs

        # Chunk
        print(f"Chunking {len(docs)} documents...")
        self.chunks = chunk_documents(docs, self.cfg)
        print(f"  → {len(self.chunks)} chunks")
        return self

    def load_custom_documents(self, documents):
        """
        Load custom document list directly.

        Args:
            documents: list of {"content": str, "metadata": dict}
        """
        self.documents = documents
        print(f"Chunking {len(documents)} custom documents...")
        self.chunks = chunk_documents(documents, self.cfg)
        print(f"  → {len(self.chunks)} chunks")
        return self

    # ── Index ───────────────────────────────────────────────────

    def build_index(self, embedding_model=None):
        """
        Encode chunks and build the search index.

        Args:
            embedding_model: override embedding model name
        """
        texts = [c['text'] for c in self.chunks]
        if not texts:
            raise ValueError("No documents loaded. Call load_dataset() first.")

        # Override embedding model if specified
        if embedding_model:
            self.cfg['embedding_model']['name'] = embedding_model
            self.cfg['model']['name'] = embedding_model

        from .api import WinnexPipeline
        self.pipeline = WinnexPipeline(
            config_path=None,
            method=self.method
        )
        # Override pipeline config with our settings
        self.pipeline.cfg = self.cfg

        # Use embedding_model section for encoding
        emb_cfg = self.cfg.get('embedding_model', self.cfg.get('model', {}))
        self.pipeline.cfg['model'] = emb_cfg
        self.pipeline.cfg['dimensions']['input_dim'] = emb_cfg.get('dimension', 384)

        print(f"Encoding {len(texts)} chunks with {emb_cfg.get('name')}...")
        vectors = self.pipeline.encode(texts, show_progress=True)
        self.doc_embeddings = vectors

        print(f"Building search index ({self.method})...")
        self.pipeline.build(vectors)
        self.is_ready = True

        # Verify bounds
        try:
            b = self.pipeline.check_bounds(vectors[0])
            print(f"  Bound check: {b['violations']} → {b['guarantee']}")
        except:
            pass

        return self

    # ── Retrieve ────────────────────────────────────────────────

    def retrieve(self, query, k=None, rerank=True):
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: text query string
            k: number of chunks to return
            rerank: apply MMR diversity reranking

        Returns:
            list of {"text": str, "metadata": dict, "score": float, "index": int}
        """
        if not self.is_ready:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = k or self.cfg.get('retriever', {}).get('top_k', 5)
        ret_cfg = self.cfg.get('retriever', {})

        # Encode query
        q_vec = self.pipeline.encode([query], show_progress=False)[0]

        # Search
        result = self.pipeline.search(q_vec, k=k * 3, return_profile=True)
        indices = result['indices']

        # MMR diversity
        if rerank and ret_cfg.get('mmr_lambda', 0.5) > 0:
            scores = None  # MMR returns reranked indices
            mmr_lambda = ret_cfg.get('mmr_lambda', 0.5)
            indices, _ = mmr_rerank(
                self.doc_embeddings, q_vec, indices, None,
                lambda_param=mmr_lambda, top_k=k
            )

        results = []
        for i, idx in enumerate(indices[:k]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {}),
                    'score': 1.0 - (i / max(len(indices), 1)),
                    'index': int(idx),
                })

        return results

    # ── Generate ────────────────────────────────────────────────

    def generate(self, query, context_chunks):
        """
        Generate answer from retrieved context.

        Args:
            query: original query string
            context_chunks: list from retrieve()

        Returns:
            str: generated answer
        """
        return generate_answer(query, context_chunks, self.cfg)

    # ── Query (retrieve + generate) ─────────────────────────────

    def query(self, query, k=None, rerank=True, return_context=False):
        """
        Full RAG query: retrieve → generate.

        Args:
            query: text query
            k: number of chunks to retrieve
            rerank: apply MMR diversity
            return_context: include retrieved chunks in response

        Returns:
            str or dict with 'answer' and optionally 'context'
        """
        import time
        t0 = time.time()

        context = self.retrieve(query, k=k, rerank=rerank)
        retrieve_time = (time.time() - t0) * 1000

        t1 = time.time()
        answer = self.generate(query, context)
        gen_time = (time.time() - t1) * 1000

        result = {
            'answer': answer,
            'timing_ms': {
                'retrieve': round(retrieve_time, 2),
                'generate': round(gen_time, 2),
                'total': round(retrieve_time + gen_time, 2),
            },
            'n_chunks': len(context),
        }

        if return_context:
            result['context'] = [
                {
                    'text': c['text'][:200] + ('...' if len(c['text']) > 200 else ''),
                    'score': c.get('score', 0),
                    'metadata': c.get('metadata', {}),
                }
                for c in context
            ]

        return result

    # ── Info ────────────────────────────────────────────────────

    def info(self):
        """Print agent configuration and status."""
        emb = self.cfg.get('embedding_model', {})
        gen = self.cfg.get('generator', {})
        ds = self.cfg.get('dataset', {})
        ret = self.cfg.get('retriever', {})

        print(f"{'='*60}")
        print(f"Winnex RAG Agent v{self.cfg.get('version', '12.2.0')}")
        print(f"{'='*60}")
        print(f"  Status:      {'Ready' if self.is_ready else 'Not initialized'}")
        if self.is_ready:
            print(f"  Documents:   {len(self.documents)}")
            print(f"  Chunks:      {len(self.chunks)}")
            print(f"  Method:      {self.method}")
        print(f"  Embedding:   {emb.get('name')} ({emb.get('dimension')}D)")
        print(f"  Generator:   {gen.get('provider')} / {gen.get('model')}")
        print(f"  Dataset:     {ds.get('source')}: {ds.get('name', 'local')}")
        print(f"  Retrieval:   top_k={ret.get('top_k', 5)}, "
              f"MMR={ret.get('mmr_lambda', 0.5)}")
        print(f"  Bounds:      {'✅ Active' if self.is_ready else '❌ N/A'}")
