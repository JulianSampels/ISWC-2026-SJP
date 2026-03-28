# Baselines

This folder contains reproductions and adapted baselines for comparison in the ISWC paper.

## Structure

```
baselines/
├── reta/                   RETA (Rosso et al., WWW'21)
│   ├── reta_filter.py      Schema-aware candidate filter
│   └── reta_grader.py      CNN-based triple scorer/ranker
├── mvf/                    MVF/GFRT (Li, Zhang, Yu — BDR'23)
│   ├── mvf_graphs.py       Head-rel and tail-rel graph construction
│   ├── mvf_model.py        Attention-GNN + inter-view alignment
│   └── mvf_filter.py       Full filter + scoring pipeline
└── adapted/                RQ1 adapted baselines
    ├── relation_first.py   Relation-First adaptation
    ├── independent.py      Independent Combination + Tail-First
    └── (see contributions.md for design rationale)
```

## Running the baselines

All baselines accept a training triple tensor of shape `(N, 3)` with integer entity/relation ids,
matching the format output by `PathE/pathe/kgloader.py`.

### RETA (no-type variant)

```python
from iswc.baselines.reta import build_reta_filter, evaluate_filter_coverage

# Build filter
reta = build_reta_filter(train_triples, alpha=0.0, beta=1)

# Generate candidates for test entities
candidates = reta.generate_candidates_batch(test_head_ids, max_candidates=500)

# Evaluate coverage
metrics = evaluate_filter_coverage(candidates, test_triples)
print(f"Coverage: {metrics['coverage']:.4f}, Avg size: {metrics['avg_size']:.1f}")
```

### GFRT

```python
from iswc.baselines.gfrt import build_gfrt_pipeline, GFRTTrainer, GFRTFilter

# Build graphs + model
model, graph_H, graph_T = build_gfrt_pipeline(
    train_triples, num_entities, num_relations, embed_dim=64
)

# Train
trainer = GFRTTrainer(model, graph_H, graph_T, train_triples, device=device)
for epoch in range(100):
    losses = trainer.train_epoch(batch_size=256)

# Generate candidates
h_emb, rH_emb, t_emb, rT_emb = trainer.get_embeddings()
gfrt_filter = GFRTFilter(h_emb, rH_emb, t_emb, rT_emb, model, train_triples)
candidates = gfrt_filter.generate_candidates_batch(test_head_ids, candidate_budget=500)
```

### Adapted baselines (RQ1)

```python
from iswc.baselines.adapted import (
    RelationFirstBaseline, TailFirstBaseline, IndependentCombinationBaseline
)

# Frequency-based (Option A)
rf = RelationFirstBaseline(train_triples, num_relations, k_r=10, k_t=50)
candidates = rf.generate_candidates_batch(test_head_ids, max_candidates=500)

ic = IndependentCombinationBaseline(train_triples, k_r=10, k_t=100)
candidates = ic.generate_candidates_batch(test_head_ids, max_candidates=500)

# Option B: plug in learned Phase-1 scores from SJP
rf.set_learned_relation_scores(phase1_rel_scores)   # Dict[entity_id -> Dict[rel_id -> score]]
ic.set_learned_scores(phase1_rel_scores, phase1_tail_scores)
```

### Evaluation (entity-level metrics)

```python
from iswc.evaluation import evaluate_entity_centric, evaluate_at_fixed_budgets, format_results_table

# Convert candidate list format: {head -> [(h, r, t, score), ...]} -> {head -> [(r, t), ...]}
predictions = {h: [(r, t) for (_, r, t, _) in cands] for h, cands in candidates.items()}

# Full evaluation
results = evaluate_entity_centric(predictions, test_triples, k_values=[1, 5, 10, 20, 50, 100])

# Fixed-budget comparison
budget_results = evaluate_at_fixed_budgets(predictions, test_triples, budgets=[50, 100, 200, 500])

# Pretty table
from iswc.evaluation import format_results_table
print(format_results_table({"RelFirst": results, "SJP": sjp_results}))
```
