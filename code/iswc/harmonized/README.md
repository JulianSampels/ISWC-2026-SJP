# Harmonized Interface (SJP + RETA)

This package exposes a cleaned workflow with no legacy command aliases.

1. Generate standardized dataset from KgLoader dataset name.
2. Prepare adapter-specific dataset files from standardized data.
3. Generate candidates as standardized CSV.
4. Rank candidates as standardized CSV.

Run from the `code/` directory:

```bash
python -m iswc.harmonized.interface --help
```

## 1) Generate Standardized Dataset (KgLoader)

```bash
python -m iswc.harmonized.interface generate-standard-dataset \
  --dataset-name fb15k237 \
  --output-dir ./iswc_data/standard/fb15k237
```

## 2) Prepare Predictor Dataset (Adapter Translation)

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

## 3) Generate Candidates

SJP generate candidates executes PathE phase 1 + phase 2.

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-budget 500 \
  --output-file ./results/sjp_candidates.csv
```

RETA generate candidates executes RETA-Filter.

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --candidate-budget 500 \
  --output-file ./results/reta_candidates.csv
```

## 4) Rank Candidates

SJP rank candidates executes PathE phase 3.

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-file ./results/sjp_candidates.csv \
  --candidate-budget 500 \
  --output-file ./results/sjp_ranked.csv
```

RETA rank candidates executes RETA-Grader.

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --model-path ./RETA_code/<model_file> \
  --candidate-file ./results/reta_candidates.csv \
  --candidate-budget 500 \
  --output-file ./results/reta_ranked.csv
```

RETA ranking requires CUDA because RETA forward uses `.cuda(...)` in model code.

## Python Adapters

Adapter methods:

- `prepare_dataset(...)`
- `generate_candidates(...)`
- `rank_candidates(...)`
