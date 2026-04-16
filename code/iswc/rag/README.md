# Retrieval-Augmented Generation (RAG) Evaluation

This directory contains the codebase for evaluating various retrieval-augmented generation (RAG) pipelines over Knowledge Graph question answering datasets (such as WebQSP and CWQ).

## Available Pipelines

- `native`: Uses embedding-based cosine similarity retrieval (e.g. SentenceTransformers).
- `gfrt`: Uses GFRT-enhanced retrieval.
- `sjp`: Uses our proposed SJP (Split-Join-Predict) instance completion retrieval directly from a model checkpoint.
- `csv`: Consumes pre-computed harmonized `candidates.csv` files (e.g., from RETA or SJP).

## Using the CSV Pipeline

If you have already generated a standardized candidate set using the harmonized interface (`iswc.harmonized.interface`), you can plug the resulting `candidates.csv` directly into the RAG pipeline.

Make sure you provide the same entity and relation mapping JSONs that resolve the integer IDs in the candidates CSV back to string identifiers.

**Example Command:**
```bash
python -m iswc.rag.run_eval \
    --dataset webqsp \
    --pipeline csv \
    --csv-file iswc_data/reta_fb15k/candidates.csv \
    --entity-map path/to/entity2id.json \
    --relation-map path/to/relation2id.json \
    --budget 10
```

*Note: The `entity-map` and `relation-map` should map string labels/IDs (as found in the benchmark dataset and natural language facts) to the integer IDs used internally by the CSV.*
