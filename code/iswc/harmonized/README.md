# Harmonized Interface (SJP + RETA)

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

RETA ranking requires CUDA because RETA forward uses `.cuda(...)` in submodule model code.

## 7) Evaluate and Compare

Evaluate one output file:

```bash
python -m iswc.harmonized.interface evaluate \
  --stage candidates \
  --input-file ./results/sjp_candidates.csv \
  --gold-triples ./iswc_data/sjp/fb15k237/test/triples.pt \
  --k-values 1,3,5,10 \
  --output-csv ./results/sjp_candidates_metrics.csv
```

Compare multiple outputs:

```bash
python -m iswc.harmonized.interface compare \
  --stage ranking \
  --gold-triples ./iswc_data/sjp/fb15k237/test/triples.pt \
  --k-values 1,3,5,10 \
  --method SJP ./results/sjp_ranked.csv \
  --method RETA ./results/reta_ranked.csv \
  --output-json ./results/ranking_compare.json
```
