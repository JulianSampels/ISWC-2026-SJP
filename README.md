# Beyond Link Prediction: Relation-Agnostic Fact Suggestion for Knowledge Graph Enrichment

**Target venue:** ISWC 2026 (International Semantic Web Conference) — Research Track
**Paper working title:** *Entity-to-Fact: A Structured Framework for Relation-Agnostic Fact Suggestion in Knowledge Graphs*

---

## Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/JulianSampels/ISWC-2026-SJP.git
cd ISWC-2026-SJP
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

This fetches both code submodules under `code/RETA_code` and `code/SJP_code`.

---

## Motivation

Standard KGC assumes the query relation is known (h, r, ?). In practice — Wikidata curation,
enterprise data enrichment, entity-page completion — users start from an entity and ask:
*what important facts about this entity are missing?*

This paper formalises that as **entity-centric fact suggestion**: given anchor h, generate and
rank plausible (r, t) pairs without any prior knowledge of which relation type is relevant.

---

## Research Questions

| # | Question | Method |
|---|----------|--------|
| RQ1 | Is entity-centric fact suggestion a distinct, practically meaningful task beyond conventional relation-conditioned link prediction? | Adapted baselines (relation-first, tail-first, independent combination) |
| RQ2 | What inference strategy effectively solves entity-centric fact suggestion under large search spaces? | Split-Join-Predict vs RETA / MVF at fixed candidate budgets |
| RQ3 | Do entity-level utility metrics complement conventional tuple-level metrics? | EntityHit@K, EntityRecall@K, Budget-to-First-Hit vs MRR/Hits@K |

---

## ISWC Fit

ISWC is the premier venue for Semantic Web and Knowledge Graph research. This paper fits
the **Research Track** on the following grounds:

1. **KG Completion** — directly advances the state of practice for incomplete KGs (Wikidata, DBpedia, Freebase-derived benchmarks).
2. **Open-World Assumption** — unlike most KGC, our framework does not presuppose schema constraints or entity types.
3. **Practical Enrichment** — the task mirrors real tools: Wikidata Recoin, Wikidata Property Suggester, LinkedIn member KG completion.
4. **Novel Evaluation** — entity-level metrics are a concrete methodological contribution to the KG evaluation ecosystem.
5. **Competitive Benchmarks** — results on FB15k-237, WN18RR, JF17k against RETA and MVF (GFRT), the two closest prior systems.

---

## Paper Structure

```
1. Introduction
   - Motivating example (entity-page completion for Wikidata entity)
   - Gap: existing KGC assumes relation is known
   - Contributions summary (5 bullets)

2. Background & Problem Setting
   - KG definitions
   - Edge-specific vs edge-agnostic prediction
   - Formal definition of entity-centric fact suggestion

3. Related Work
   - Standard link prediction (TransE, RotatE, …)
   - Relation prediction (OKELE, Recoin, Wikidata property suggester)
   - Tuple / instance completion (RETA, MVF/GFRT)
   - Entity-centric enrichment applications

4. Entity-Centric Fact Suggestion (RQ1)
   - Distinction from link prediction (Table: comparison.md)
   - Adapted baselines: Relation-first, Tail-first, Independent combination
   - Experimental evidence that naive adaptations fall short

5. The Split-Join-Predict Framework (RQ2)
   5.1 PathE: path-based entity encoder
   5.2 Phase 1 — Property Prediction (relation scoring + entity scoring)
   5.3 Phase 2 — Candidate Generation (global joint scoring, top-k)
   5.4 Phase 3 — Triple Classification (discriminative re-ranking)
   5.5 Implementation & scalability

6. Evaluation
   6.1 Datasets (FB15k-237, WN18RR, JF17k)
   6.2 Baselines (RETA, MVF/GFRT, adapted baselines)
   6.3 Metrics (tuple-level + entity-level — RQ3)
   6.4 Results: candidate generation
   6.5 Results: end-to-end tuple prediction
   6.6 Ablation: loss functions, global vs local top-k, α/β weighting
   6.7 Fixed-budget analysis (k ∈ {50, 100, 200, 500})

7. Discussion & Analysis
   - Sparse vs dense entity analysis
   - Error analysis
   - Real-world application scenarios

8. Conclusion & Future Work
```

Harmonized workflow entrypoint:

```bash
cd code
python -m iswc.harmonized.interface --help
```

Detailed command examples for the standardized 7-step workflow are documented in `code/iswc/harmonized/README.md`.

---

## Repository Layout

```text
ISWC-2026-SJP/
├── README.md
├── comparison.md
├── contributions.md
├── TODO.org
└── code/
   ├── iswc/
   │   ├── baselines/
   │   │   ├── adapted/
   │   │   │   ├── relation_first.py
   │   │   │   └── independent.py
   │   │   ├── gfrt/
   │   │   │   ├── gfrt_filter.py
   │   │   │   ├── gfrt_graphs.py
   │   │   │   └── gfrt_model.py
   │   │   └── README.md
   │   ├── harmonized/
   │   │   ├── interface.py
   │   │   ├── base_adapter.py
   │   │   ├── sjp_adapter.py
   │   │   ├── reta_adapter.py
   │   │   ├── gfrt_adapter.py
   │   │   ├── adapters.py
   │   │   └── README.md
   │   └── evaluation/
   │       └── entity_metrics.py
   ├── RETA_code/               # git submodule (original RETA/RETA++)
   │   ├── main.py
   │   ├── main_reta_plus.py
   │   └── README.md
   └── SJP_code/                # git submodule (Split-Join-Predict code)
      ├── generate_dataset.py
      ├── PathE/pathe/
      ├── data/path_datasets/
      ├── figures/code/
      └── README.md
```

---

## Key Differences vs Prior Work

| Aspect                      | RETA (WWW'21)                             | MVF/GFRT (BDR'23)            | **Ours (SJP)**                               |
|-----------------------------|-------------------------------------------|------------------------------|----------------------------------------------|
| Entity types required       | **Yes** (RETA-Filter fails without types) | No                           | **No**                                       |
| Graph views                 | Single KG schema tensor                   | Head-rel + tail-rel (2 GNNs) | Single graph + inverse relations (1 encoder) |
| Candidate generation        | Per-entity schema match                   | Per-entity local score       | **Global** joint score (dataset-wide)        |
| Inductive (unseen entities) | No                                        | No                           | **Yes** (relational context only)            |
| Count-aware objectives      | No                                        | No                           | **Yes** (Poisson, NB, Hurdle)                |
| Entity-level metrics        | Not reported                              | Not reported                 | **Yes** (EntityHit, EntityRecall, B2FH)      |
| Benchmarks                  | JF17k, FB15k, HumanWiki                   | JF17k, FB15K237, UMLS        | FB15k-237, WN18RR, JF17k                     |
