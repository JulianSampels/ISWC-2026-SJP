# Harmonized Interface (SJP + RETA)

This package provides a standardized four-step workflow:

1. generate dataset
2. translate dataset (adapter-specific)
3. generate candidates
4. rank candidates

All candidate/ranking outputs are saved in one standardized ranked format
(`.csv` or `.pt` with head_id, relation_id, tail_id, score, rank semantics).

## Core commands

Run from the `code/` directory:

```bash
python -m iswc.harmonized.interface --help
```

### Step 1: Generate canonical dataset

From built-in KgLoader datasets:

```bash
python -m iswc.harmonized.interface generate-dataset \
  --kg-dataset-name fb15k237 \
  --output-dir ./iswc_data/standard/fb15k237
```

From manually downloaded split files:

```bash
python -m iswc.harmonized.interface generate-dataset \
  --source-dir ./downloads/fb15k237 \
  --output-dir ./iswc_data/standard/fb15k237 \
  --triple-order hrt
```

### Step 2: Translate dataset via adapter

To SJP format:

`SJP` translation always uses manual inverse relations because PathE expects
`relation2inverseRelation.json` and the corresponding inverse ids.

```bash
python -m iswc.harmonized.interface translate-dataset \
  --adapter sjp \
  --standard-dataset-dir ./iswc_data/standard/fb15k237 \
  --output-dir ./iswc_data/sjp/fb15k237 \
  --num-paths-per-entity 20 \
  --num-steps 10
```

To RETA format:

```bash
python -m iswc.harmonized.interface translate-dataset \
  --adapter reta \
  --standard-dataset-dir ./iswc_data/standard/fb15k237 \
  --output-dir ./iswc_data/reta/fb15k237 \
  --default-entity-type Thing
```

### Step 3: Generate candidates

SJP adapter (runs SJP submodule if no `--phase2-candidate-file` is provided):

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-budget 500 \
  --output-file ./results/sjp_candidates_500.csv
```

You can pass extra runner options (for example small smoke tests):

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter sjp \
  --path-dataset-dir ./iswc_data/sjp/fb15k237 \
  --candidate-budget 500 \
  --output-file ./results/sjp_candidates_500.csv \
  --runner-arg=--max_epochs --runner-arg=1
```

RETA adapter:

```bash
python -m iswc.harmonized.interface generate-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --model-path ./RETA_code/<model_file> \
  --candidate-budget 500 \
  --output-file ./results/reta_candidates_500.csv
```

`RETA` candidate/ranking commands require a trained RETA model checkpoint and a
CUDA-capable setup, because RETA inference uses `.cuda(...)` in its forward path.

### Step 4: Rank candidates

The same adapter-specific orchestration is exposed through `rank-candidates`
for explicit final ranking runs:

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter sjp \
  --phase2-candidate-file ./logs/harmonized/harmonized_sjp/phase2_candidates/phase2_candidates_test.pt \
  --candidate-budget 500 \
  --output-file ./results/sjp_ranked_500.csv
```

```bash
python -m iswc.harmonized.interface rank-candidates \
  --adapter reta \
  --reta-code-dir ./RETA_code \
  --reta-data-dir ./iswc_data/reta/fb15k237 \
  --model-path ./RETA_code/<model_file> \
  --candidate-budget 500 \
  --output-file ./results/reta_ranked_500.csv
```

## Evaluation and comparison

```bash
python -m iswc.harmonized.interface compare \
  --gold-triples ./iswc_data/sjp/fb15k237/test/triples.pt \
  --method SJP ./results/sjp_ranked_500.csv \
  --method RETA ./results/reta_ranked_500.csv
```

## Python adapters

Adapters expose standardized methods:

- `prepare_dataset(...)`
- `generate_candidates(...)`
- `generate_final_ranking(...)`
- `evaluate(...)`

```python
from iswc.harmonized import RETAAdapter, SJPAdapter

sjp = SJPAdapter()
reta = RETAAdapter(reta_code_dir="./RETA_code", reta_data_dir="./iswc_data/reta/fb15k237")
```
