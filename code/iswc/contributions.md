# Key Contributions for ISWC Submission

This document lists the contributions of the Split-Join-Predict (SJP) paper in priority order
for an ISWC research track submission. Each contribution is paired with the claim, the evidence,
and the section in the paper where it should appear.

---

## C1 — New Task Formalisation: Entity-Centric Fact Suggestion

**What:** We formally define the task of *entity-centric fact suggestion* (query form (h, ?, ?))
and distinguish it from all existing KGC paradigms (triple classification, entity prediction,
relation prediction, and tuple prediction as previously studied).

**Why it matters for ISWC:** The Semantic Web community increasingly uses KGs in enrichment
and curation workflows (Wikidata, DBpedia, enterprise graphs). The assumption that the
queried relation is already known is fundamentally at odds with how curators work.
Real tools (Wikidata Recoin, Property Suggester, LinkedIn KG completion) confirm this need.

**Claim:** Entity-centric fact suggestion is a practically meaningful, formally distinct task
that cannot be naturally solved by naively adapting conventional relation-conditioned KGC.

**Evidence:**
- Table: `comparison.md` — 10-dimension comparison between link prediction and fact suggestion.
- Experimental results of adapted baselines (RQ1) show conventional adaptations achieve
  poor candidate coverage and high Budget-to-First-Hit under realistic budgets.

**Paper section:** §2 (Problem Setting) + §4 (RQ1 baselines).

---

## C2 — The Split-Join-Predict Framework

**What:** A three-phase, fully schema-free framework for entity-centric fact suggestion:

1. **Phase 1 (Split):** Decomposes (h, ?, ?) into two independent property prediction tasks:
   - Relation scoring: P(r | h) via Entity-to-Relation head
   - Entity scoring: P(t | h) via Entity-to-Entity head
   Both use the PathE encoder with count-aware loss functions (BCE, Poisson, NB, Hurdle).

2. **Phase 2 (Join):** Combines Phase 1 scores into a global joint probabilistic score:
   ```
   S(h, r, t) = β·log s(h,t) + (1-β)·((1-α)·log s(h,r) + α·log s(t, r⁻¹))
   ```
   Uses **global top-k** selection (dataset-wide) rather than per-entity quotas — this is
   a critical design choice that benefits sparse entities.

3. **Phase 3 (Predict):** Discriminative re-ranking via a triple classifier trained on
   hard negatives produced by Phase 2, ensuring global structural consistency.

**Why it matters for ISWC:**
- Fully inductive: no entity types, no schema, works on unseen entities.
- Scalable: global top-k avoids materialising |V|×|R|×|V| search space (>14 TB on FB15k-237).
- Count-aware objectives model relational cardinality asymmetry, which is overlooked by
  all prior tuple prediction methods.

**Claim:** The structured three-phase decomposition outperforms all prior tuple prediction
methods across FB15k-237, WN18RR, and JF17k.

**Evidence:** Tables in §6 (candidate coverage + MRR/Hits@K/Recall@K end-to-end results).

**Paper section:** §5.

---

## C3 — Count-Aware Objective Functions for Property Prediction

**What:** We study four objective functions for property prediction — BCE, Poisson NLL,
Negative Binomial NLL, and Hurdle — and show that count-aware objectives (Poisson, NB, Hurdle)
outperform binary existence (BCE) on KGs with asymmetric relational multiplicities (especially
FB15k-237, which has long-tailed head-per-(tail, relation) counts).

**Why it matters:** Existing tuple prediction methods (RETA, MVF) model property existence
as binary classification. KGs are naturally count-structured (one person has one birthdate,
but may appear in hundreds of award nominations). Matching the objective to this structure
is a principled modelling contribution.

**Claim:** The choice of count objective function significantly affects candidate coverage and
tuple prediction quality, with NB/Hurdle achieving the best coverage-quality tradeoff on
dense-relation datasets.

**Evidence:** Ablation study in §6.6.

**Paper section:** §5.3.1 (Loss Functions) + §6.6 (Ablation).

---

## C4 — Global vs. Local Candidate Generation

**What:** We propose and experimentally validate a *global top-k* candidate selection strategy
that ranks candidates dataset-wide by joint probability, rather than maintaining a per-entity
quota. This removes the need to pre-specify per-entity candidate budgets (as in RETA and MVF).

**Why it matters:** Per-entity quota selection systematically under-serves dense hub entities
and over-allocates budget to sparse entities. Global selection is more sample-efficient: under
equal total candidate counts, global top-k achieves substantially higher coverage.

**Claim:** Global top-k selection improves coverage for high-degree entities without harming
sparse entities, outperforming per-entity quota approaches under matched budgets.

**Evidence:** Coverage vs candidate size curves (§6.4), fixed-budget analysis (§6.7).

**Paper section:** §5.3.2 (Phase 2).

---

## C5 — Entity-Level Evaluation Metrics

**What:** We introduce three new entity-level utility metrics for fact suggestion evaluation:

- **EntityHit@K(h)** = 1 if at least one gold fact appears in top-K suggestions for h.
- **EntityRecall@K(h)** = |TopK(h) ∩ G(h)| / |G(h)|, fraction of h's gold facts covered.
- **Budget-to-First-Hit(h)** = min{k | TopK(h) ∩ G(h) ≠ ∅}, budget needed for first correct fact.

Averaged over all test entities, these measure *practical utility for a curator* rather than
ranking quality of individual isolated facts (which is what MRR/Hits@K measure).

**Why it matters for ISWC:** Evaluation methodology is a core contribution valued at ISWC.
These metrics directly mirror the curation workflow and expose failure modes invisible to MRR:
a system can achieve good MRR by ranking one fact of a prolific entity while leaving
most facts uncovered. EntityRecall@K captures this.

**Claim:** Entity-level metrics and tuple-level metrics capture complementary aspects of
fact suggestion quality; conventional metrics alone are insufficient to evaluate enrichment-oriented systems.

**Evidence:** RQ3 analysis (§6.5), motivating example in §6.3.

**Paper section:** §6.3 (Metrics) + §6.5 (RQ3 analysis).

---

## Summary Table

| # | Contribution | Novelty | Evidence type |
|---|-------------|---------|---------------|
| C1 | Entity-centric fact suggestion as new task | Task formalisation | Baseline comparison, formal definition |
| C2 | Split-Join-Predict framework | Methodological | Experimental results (coverage + quality) |
| C3 | Count-aware objective functions | Modelling | Ablation study |
| C4 | Global top-k candidate generation | Algorithmic | Coverage curves, fixed-budget table |
| C5 | Entity-level evaluation metrics | Evaluation methodology | RQ3 analysis, motivating example |

---

## Differentiators vs RETA and MVF

### vs RETA (Rosso et al., WWW'21)
- RETA requires entity type information (RETA-Filter fails gracefully without types, but
  uses heuristic substitutes that degrade quality). Our framework is fully type-free.
- RETA uses a CNN grader that scores (triplet, schema) pairs — quadratic in entity type count.
  Our Phase 3 classifier scores (h, r, t) directly using structural path representations.
- RETA evaluates on JF17k, FB15k, HumanWiki. We additionally evaluate on WN18RR and
  FB15k-237, and report entity-level metrics not present in RETA.

### vs MVF / GFRT (Li, Zhang, Yu — Big Data Research, 2023)
- MVF trains two separate GNNs (head-rel graph + tail-rel graph) with an inter-view alignment
  module. Our framework uses a single shared encoder with inverse relation augmentation —
  fewer parameters, simpler training, yet competitive or superior results.
- MVF uses per-entity local scoring for candidate generation. Our global top-k is more
  sample-efficient under equal budgets.
- MVF evaluates on JF17k, FB15K237, UMLS. We share JF17k and FB15K237, enabling direct
  comparison, and additionally evaluate on WN18RR.
- MVF does not report entity-level metrics.

---

## Gaps to Address (open TODOs before submission)

1. **RQ1 experiments**: Run relation-first, tail-first, and independent baselines on all three
   datasets and compute EntityHit@10, EntityRecall@10, Budget-to-First-Hit.
   → See `baselines/adapted/`

2. **RETA reproduction**: Implement RETA-Filter (no-type variant) and RETA-Grader (simplified
   CNN) on JF17k and FB15k-237 with our evaluation protocol.
   → See `baselines/reta/`

3. **MVF reproduction**: Implement GFRT on JF17k and FB15k-237 with our evaluation protocol.
   → See `baselines/mvf/`

4. **Fixed-budget table** (k ∈ {50, 100, 200, 500}): compare all methods under matched budgets.

5. **Entity-level metric computation**: Integrate `evaluation/entity_metrics.py` into the
   existing `pathe_full_eval.py` evaluation pipeline.

6. **Related work survey**: Add coverage of OKELE, Wikidata Recoin, Wikidata Property Suggester,
   LinkedIn member KG completion — these ground the practical motivation.
   → See additional literature below.

---

## Additional Related Literature (not yet cited in thesis)

| Paper                                                                      | Venue      | Relevance                                         |
|----------------------------------------------------------------------------|------------|---------------------------------------------------|
| Zangerle et al., "Evaluating of property recommender systems for Wikidata" | ISWC'16    | Motivates entity-centric property suggestion task |
| Lajus & Suchanek, "Are all people married?"                                | WWW'18     | Obligatory relation detection (RQ1 context)       |
| Balaraman et al., "Recoin: relative completeness in Wikidata"              | WWW'18     | Direct application motivation                     |
| Cao et al., OKELE                                                          | WWW'20     | Open-world tuple prediction baseline              |
| Yao et al., "KG-BERT"                                                      | arXiv'19   | Triple classification baseline                    |
| Zhu et al., "Neural Bellman-Ford Networks"                                 | NeurIPS'21 | Inductive KGC (comparison point)                  |
| Galkin et al., "ULTRA"                                                     | NeurIPS'23 | Foundation model for KGC (broader context)        |
