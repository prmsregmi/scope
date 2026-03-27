# Future Directions

## 1. Embedding Persistence for Faster Experimentation [DONE]

Implemented in `scope/embeddings/disk_cache.py` as `DiskEmbeddingCache`.

- Multi-level caching: memory -> disk (.npz files) -> compute
- SHA256-keyed lookups with atomic writes
- CLI: `--cache-dir PATH` (default: `.scope_cache`), `--no-cache` to disable
- Embeddings persist across runs; second run skips all embedding computation

Not implemented: KeyBERT extraction result caching (the ~50s win), `--clear-cache` flag.

---

## 2. Direct Cosine Similarity vs KeyBERT [DONE]

Implemented in `scope/modeling/keybert_similarity.py` as `_calculate_direct_cosine()`.

- `--no-keybert` flag skips KeyBERT entirely and uses direct cosine similarity
- 5 calculation modes: `st_baseline`, `jina_mixed`, `jina_bag_of_words`, `jina_full_text`, `hybrid`
- `--prefilter SIM` gates expensive KeyBERT calls via batch cosine similarity matrix
- Hybrid mode: Jina for full-text embeddings, SentenceTransformers for keyword embeddings

---

## 3. Unsupervised Topic Discovery [DONE]

Implemented in `scope/discovery/` module.

### Experimental validation (teammate results)

Tested 4 embedding methods x 2 clustering algorithms on dataset with hidden-prompt ground truth:

| Embedding | HDBSCAN Purity (mean) | High-purity clusters | NMI  | ARI  |
|-----------|----------------------|---------------------|------|------|
| **MiniLM**    | **0.75**             | **26/30**           | **0.54** | 0.16 |
| Jina-v2   | 0.61                 | 13/20               | 0.17 | 0.17 |
| TF-IDF    | 0.54                 | 18/32               | 0.14 | 0.14 |
| W2V-SIF   | 0.50                 | 9/20                | 0.08 | 0.08 |

MiniLM + HDBSCAN is the clear winner. HDBSCAN outperforms KMeans-36 (no K needed, better cluster boundaries). Low ARI across all methods indicates cluster boundaries are noisy for generic conversation but clean for specific topics.

### Implementation

Pipeline: embed all hour-texts -> UMAP (384d -> 5d) -> HDBSCAN -> c-TF-IDF labeling.

**New module: `scope/discovery/`**
- `clusterer.py`: `TopicDiscoverer` — batch embed, UMAP reduce, HDBSCAN/KMeans cluster
- `labeler.py`: `ClusterLabeler` — c-TF-IDF keyword extraction + optional mapping to predefined topics
- `block_finder.py`: `ClusterBlockFinder` — convert cluster assignments to contiguous hour blocks

**CLI flags:**
- `--discover-topics` — enables unsupervised pipeline
- `--clustering-algorithm {hdbscan,kmeans}` (default: hdbscan)
- `--hdbscan-min-cluster-size`, `--hdbscan-min-samples`
- `--kmeans-k` (required for kmeans)
- `--umap-components`, `--umap-neighbors`
- `--map-to-predefined` — maps clusters to predefined topic names via centroid cosine similarity

**Dependencies:** `umap-learn>=0.5.0` (optional extra: `unsupervised`)

### Open questions

- **Noise handling**: HDBSCAN labels generic conversation as noise (-1). Currently excluded from segments. Consider: should noise form an "Uncategorized" topic?
- **Full dataset scale**: 258k messages at 384-dim = ~380MB embeddings. Disk cache mitigates recomputation but UMAP fit is still expensive. Incremental UMAP (`transform()`) could help.
- **Ground truth evaluation**: NMI/ARI/purity metrics are implemented in `ClusteringMetrics` but require labeled data with per-message ground truth. The current `labeled_test_data.csv` (154 samples) is for supervised accuracy only.
