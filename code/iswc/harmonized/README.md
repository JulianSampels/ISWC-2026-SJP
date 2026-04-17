# Harmonized Interface (SJP + RETA + GFRT)

This package exposes a clean standardized workflow with explicit train and inference steps.

1. Generate standardized dataset from KgLoader dataset name.
2. Prepare adapter-specific dataset files.
3. Train candidate model.
4. Generate candidate CSV.
5. Train ranking model.
6. Rank candidate CSV.
7. Evaluate or compare metrics.

Run from the `code/` directory:

```bash
python -m iswc.harmonized.interface --help
```

## Module Structure

- `base_adapter.py`: shared `CandidateAdapter` base class plus candidate/ranking serialization helpers.
- `sjp_adapter.py`: all SJP-specific dataset preparation, training, candidate generation, and ranking logic.
- `reta_adapter.py`: all RETA-specific dataset preparation, dictionary/runtime helpers, training, candidate generation, and ranking logic.
- `gfrt_adapter.py`: GFRT-specific numeric dataset preparation, GFRT model training, candidate generation, and candidate re-ranking.
- `adapters.py`: compatibility exports for existing imports.

## 1) Generate Standardized Dataset

```bash
python -m iswc.harmonized.interface generate-standard-dataset \
  --dataset-name fb15k237 \
  --output-dir ./iswc_data/standard/fb15k237
```

## 2) Prepare Adapter Dataset

SJP:

```bash
python -m iswc.harmonized.interface prepare-dataset \
  --adapter sjp \
  --standard-dataset-dir ./iswc_data/standard/fb15k237 \
  --output-dir ./iswc_data/sjp/fb15k237 \
  --num-paths-per-entity 20 \
  --num-steps 10
```

RETA:

```bash
python -m iswc.harmonized.interface prepare-dataset \
  --adapter reta \
  --standard-dataset-dir ./iswc_data/standard/fb15k237 \
  --output-dir ./iswc_data/reta/fb15k237 \
  --default-entity-type Thing
```

GFRT:

```bash
python -m iswc.harmonized.interface prepare-dataset \
  --adapter gfrt \
  --standard-dataset-dir ./iswc_data/standard/fb15k237 \
  --output-dir ./iswc_data/gfrt/fb15k237
```

## 3) Train Candidate Model

SJP trains phase-1 model and returns a tuple checkpoint.

```bash
python -m iswc.harmonized.interface train-candidate-model \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --log-dir ./logs/harmonized \
  --expname sjp_fb15k237
```

RETA candidate generation is filter-based and has no trainable candidate model.

```bash
python -m iswc.harmonized.interface train-candidate-model \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237
```

GFRT trains one model that is used for both candidate generation and ranking:

```bash
python -m iswc.harmonized.interface train-candidate-model \
  --adapter gfrt \
  --path-dataset-dir ./iswc_data/gfrt/fb15k237 \
  --log-dir ./logs/harmonized \
  --expname gfrt_fb15k237 \
  --max-epochs 100 \
  --embed-dim 100 \
  --num-layers 2 \
  --top-k1 100 \
  --top-k2 30
```

## 4) Generate Candidates

SJP:

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-model-path ./logs/harmonized/sjp_fb15k237/checkpoints/<tuple.ckpt> \
  --candidate-budget 500 \
  --output-file ./results/sjp_candidates.csv
```

RETA:

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --candidate-budget 500 \
  --output-file ./results/reta_candidates.csv
```

GFRT:

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter gfrt \
  --path-dataset-dir ./iswc_data/gfrt/fb15k237 \
  --candidate-model-path ./logs/harmonized/gfrt_fb15k237/gfrt_model.pt \
  --candidate-budget 500 \
  --top-m-relations 20 \
  --top-n-tails 100 \
  --output-file ./results/gfrt_candidates.csv
```

## 5) Train Ranking Model

SJP trains phase-3 model and returns a triple checkpoint.

```bash
python -m iswc.harmonized.interface train-ranking-model \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-model-path ./logs/harmonized/sjp_fb15k237/checkpoints/<tuple.ckpt> \
  --log-dir ./logs/harmonized \
  --expname sjp_fb15k237
```

RETA trains RETA/RETA++ model via `RETA_code/main_reta_plus.py` without modifying submodule code.

```bash
python -m iswc.harmonized.interface train-ranking-model \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --model-output-dir ./results/reta_models \
  --epochs 1000
```

GFRT uses the same model for ranking; this step validates or reuses the existing checkpoint:

```bash
python -m iswc.harmonized.interface train-ranking-model \
  --adapter gfrt \
  --path-dataset-dir ./iswc_data/gfrt/fb15k237 \
  --candidate-model-path ./logs/harmonized/gfrt_fb15k237/gfrt_model.pt
```

## 6) Rank Candidates

SJP:

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-file ./results/sjp_candidates.csv \
  --ranking-model-path ./logs/harmonized/sjp_fb15k237/checkpoints/<triple.ckpt> \
  --candidate-budget 500 \
  --output-file ./results/sjp_ranked.csv
```

RETA:

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --candidate-file ./results/reta_candidates.csv \
  --ranking-model-path ./results/reta_models/<model_file> \
  --candidate-budget 500 \
  --output-file ./results/reta_ranked.csv
```

GFRT:

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter gfrt \
  --path-dataset-dir ./iswc_data/gfrt/fb15k237 \
  --candidate-file ./results/gfrt_candidates.csv \
  --ranking-model-path ./logs/harmonized/gfrt_fb15k237/gfrt_model.pt \
  --candidate-budget 500 \
  --output-file ./results/gfrt_ranked.csv
```

RETA ranking requires CUDA because RETA forward uses `.cuda(...)` in submodule model code.

## 7) Evaluate and Compare

Evaluate one output file (use the matching adapter gold file):

SJP candidate metrics:

```bash
python -m iswc.harmonized.interface evaluate \
  --stage candidates \
  --input-file ./results/sjp_candidates.csv \
  --gold-triples ./iswc_data/sjp/fb15k237/gold_test.csv \
  --k-values 1,3,5,10 \
  --output-csv ./results/sjp_candidates_metrics.csv
```

RETA candidate metrics:

```bash
python -m iswc.harmonized.interface evaluate \
  --stage candidates \
  --input-file ./results/reta_candidates.csv \
  --gold-triples ./iswc_data/reta/fb15k237/gold_test.csv \
  --k-values 1,3,5,10 \
  --output-csv ./results/reta_candidates_metrics.csv
```

GFRT candidate metrics:

```bash
python -m iswc.harmonized.interface evaluate \
  --stage candidates \
  --input-file ./results/gfrt_candidates.csv \
  --gold-triples ./iswc_data/gfrt/fb15k237/gold_test.csv \
  --k-values 1,3,5,10 \
  --output-csv ./results/gfrt_candidates_metrics.csv
```

Compare multiple outputs:

```bash
python -m iswc.harmonized.interface compare \
  --stage ranking \
  --gold-triples ./iswc_data/sjp/fb15k237/gold_test.csv \
  --k-values 1,3,5,10 \
  --method SJP ./results/sjp_ranked.csv \
  --method RETA ./results/reta_ranked.csv \
  --output-json ./results/ranking_compare.json
```

Note: use the matching adapter `gold_test.csv` (generated during `prepare-dataset`) for evaluation,
and for cross-adapter comparison ensure all `--method` files are in the same ID space as
`--gold-triples` (for example after mapping to SJP IDs).
