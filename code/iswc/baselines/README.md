# Baselines

This folder contains reproductions and adapted baselines for comparison in the ISWC paper.

## Structure

```
baselines/
├── adapted/                RQ1 adapted baselines
│   ├── __init__.py
│   ├── relation_first.py   Relation-First adaptation
│   └── independent.py      Independent Combination + Tail-First
├── gfrt/                   GFRT baseline (Li, Zhang, Yu — BDR'23)
│   ├── __init__.py
│   ├── gfrt_filter.py      Candidate generation + training wrappers
│   ├── gfrt_graphs.py      Head-rel and tail-rel graph construction
│   └── gfrt_model.py       Attention-GNN architecture
├── bpr_recoin.py           BPR and Recoin relation-prediction baselines (RETA paper)
└── README.md
```

## Running the baselines

All baselines accept a training triple tensor of shape `(N, 3)` with integer entity/relation ids,
matching the format output by `PathE/pathe/kgloader.py`.

### BPR (Bayesian Personalised Ranking)

BPR learns entity and relation embeddings via a pairwise ranking objective so that,
for a given head h, observed relations are scored higher than unobserved ones.
Tail candidates are retrieved from the training co-occurrence index.

```python
from iswc.baselines import BPRBaseline

bpr = BPRBaseline(
    train_triples,
    num_entities=num_entities,
    num_relations=num_relations,
    embed_dim=64,
    num_epochs=100,
    k_r=10,   # top-k relations per entity
    k_t=50,   # top-k tails per (entity, relation)
    device=device,
)
candidates = bpr.generate_candidates_batch(test_head_ids, max_candidates=500)
```

### Recoin (Collaborative Filtering for Relations)

Recoin ranks relations for entity h by their frequency among entities that
share at least one entity type with h (Boolean similarity from the RETA paper).
Requires `entity_types: Dict[entity_id -> List[type_id]]`.

```python
from iswc.baselines import RecoinBaseline

recoin = RecoinBaseline(
    train_triples,
    entity_types=entity_types,   # {entity_id: [type_id, ...]}
    k_r=10,
    k_t=50,
)
candidates = recoin.generate_candidates_batch(test_head_ids, max_candidates=500)

# Access the relation predictor directly (returns [(rel_id, score), ...])
scores = recoin.predict_relations(head_id, top_k=20)
```

### RETA / RETA++

The original RETA implementation is available as a separate submodule in `code/RETA_code/`.
Use that folder's README for end-to-end RETA and RETA++ training/evaluation commands.

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
