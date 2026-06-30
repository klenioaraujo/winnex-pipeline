# Configuration Guide — Model & Search Presets

## Model Configuration

Every config has a `model` section that defines the embedding model:

```json
{
  "model": {
    "name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu",
    "normalize": true,
    "max_length": 256,
    "batch_size": 64,
    "provider": "sentence-transformers"
  }
}
```

| Field | Description | Default |
|---|---|---|
| `name` | Model identifier (SBERT, HuggingFace, or API name) | `all-MiniLM-L6-v2` |
| `dimension` | Output embedding dimension | 384 |
| `device` | `cpu`, `cuda`, `mps`, or `api` | `cpu` |
| `normalize` | L2-normalize embeddings | `true` |
| `max_length` | Max token length for encoding | 256 |
| `batch_size` | Batch size for encoding | 64 |
| `provider` | `sentence-transformers`, `transformers`, `openai` | auto-detected |

## Supported Models

| Model | Dim | Config File | Quality | Speed | Use Case |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | `config/models/minilm.json` | Good | ⚡ Fastest | General purpose, prototyping |
| **all-mpnet-base-v2** | 768 | `config/models/mpnet.json` | **Best SBERT** | 🐢 2× slower | Semantic search, clustering |
| **BAAI/bge-base-en-v1.5** | 768 | `config/models/bge.json` | **MTEB #1** | 🐢 2× slower | Retrieval, RAG pipelines |
| **text-embedding-3-small** | 1536 | `config/models/openai_small.json` | **OpenAI** | ☁️ API | Production, high quality |
| **text-embedding-3-large** | 3072 | — (custom) | **Best overall** | ☁️ API | Enterprise, maximum quality |
| **GPT-2** (hidden) | 768 | `config/models/gpt2.json` | Moderate | 🐢 Slow | Generative doc embeddings |
| **SIFT / synthetic** | 128 | `config/models/sift.json` | N/A | N/A | Pre-embedded vectors |

### Using Different Models

```python
# With a model preset
pipe = WinnexPipeline(config_path='config/models/bge.json')

# Build from pre-embedded vectors
pipe.build(my_vectors)  # automatically adapts stage_dims

# Or encode text directly
pipe.build_from_texts(["doc1", "doc2", ...])
```

### Auto-Dimension Detection

The pipeline **auto-detects and adapts** to any embedding dimension:
- `input_dim` is read from the model config's `dimension` field
- If pre-embedded vectors are passed, the actual array shape overrides the config
- Stage dimensions are automatically clamped: no stage can exceed the input dimension
- If fewer than 2 stages survive, defaults `[D//8, D//2]` are used

```python
# Works with any model — no config changes needed
vectors = my_768d_model.encode(texts)  # (N, 768)
pipe.build(vectors)  # auto-adapts: input_dim=768, stage_dims=[64, 128]
```

## Search Configuration

```json
{
  "search": {
    "adaptive_keep_base": 0.25,
    "adaptive_keep_min": 0.05,
    "adaptive_keep_max": 0.50,
    "adaptive_bounds_sensitivity": 0.12,
    "stage2_topk": 500,
    "final_results": 10
  }
}
```

| Parameter | Description | Default | Range |
|---|---|---|---|
| `adaptive_keep_base` | Base ratio for Stage 1 retention | 0.25 | 0.05–0.50 |
| `adaptive_keep_min` | Minimum retention ratio | 0.05 | 0.01–0.20 |
| `adaptive_keep_max` | Maximum retention ratio | 0.50 | 0.10–1.00 |
| `adaptive_bounds_sensitivity` | How aggressively to prune based on bound spread | 0.12 | 0.01–1.00 |
| `stage2_topk` | Candidates retained for exact cosine | 500 | 100–2000 |
| `final_results` | Number of results returned | 10 | 1–100 |

The adaptive keep formula:
```
raw_keep = base * sensitivity / max(bound_range, 0.01)
keep = clip(raw_keep, min, max)
```

**Wide bound range** (structured data like SBERT) → lower keep → faster search.
**Narrow bound range** (uniform data like SIFT) → higher keep → more accurate.

## Hybrid (MadHybrid) Configuration

```json
{
  "hybrid": {
    "enabled": true,
    "n_cells": 64,
    "n_probe": [3, 5, 8, 10, 15],
    "clustering": {
      "algorithm": "MiniBatchKMeans",
      "batch_size": 20000,
      "n_init": 3,
      "max_iter": 50
    }
  }
}
```

## Modulation (Error Backpropagation)

```json
{
  "modulation": {
    "error_backprop": true,
    "alpha_smoothing": 0.5,
    "alpha_min": 0.01,
    "alpha_max": 0.99
  }
}
```

The modulation computes a per-document learning rate:
```
α = σ((e₁ - e₂) / μ · smoothing)
score = B₁ + α · (B₂ - B₁)
```

When bounds tighten significantly from Stage 1 to Stage 2 (e₁ >> e₂), α → 1 and the Stage 2 bound dominates. When both stages agree, α → 0 and the cheaper Stage 1 bound is trusted.
