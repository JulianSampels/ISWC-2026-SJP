"""
dataset_stats.py — Dataset statistics for KG benchmarks.

Computes and prints two groups of statistics:

  General (per split + combined):
    - # triples / facts
    - # unique entities (heads ∪ tails), heads, tails
    - # unique relations
    - Average facts per entity, out-degree, in-degree

  Path properties (train graph → test facts):
    - Test-fact reachability @1/2/3 hops
        % of test (h,t) pairs connected by a directed path of length ≤ k
        in the training graph.  The primary signal for path-based models.
    - Unique relation paths of length 2 and 3
        # distinct relation sequences (r1,r2) / (r1,r2,r3) that exist
        as walks in the training graph.  Measures compositional richness.
    - Avg paths per reachable test pair (up to length 3)
        How much redundant path evidence exists per fact.  Higher values
        give path-aggregation models (like PathE) more signal.
    - Entities in largest weakly-connected component
        % of entities reachable from the largest component.
    - Avg clustering coefficient (sampled)
        Triangle density.  High values indicate local redundancy.
    - Avg shortest path length (sampled, undirected, reachable pairs only)
        Typical graph distance.  Longer paths favour deeper path models.

Large-graph handling
--------------------
For datasets with > 2 M training triples (e.g. wiki5m), path computations
that are quadratic in graph size are automatically capped via sampling.
The sample sizes are printed next to each metric so results are reproducible.

Usage
-----
  # All datasets (default):
  python -m iswc.dataset_stats

  # One dataset:
  python -m iswc.dataset_stats --dataset fb15k237

  # Skip path stats (faster):
  python -m iswc.dataset_stats --no-path-stats

  # Control sampling budget:
  python -m iswc.dataset_stats --sample-pairs 2000 --sample-entities 2000

  # Save to CSV:
  python -m iswc.dataset_stats --output stats.csv
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOT = _THIS_DIR.parent / "iswc_data" / "standard"

# Datasets with > this many training triples use sampled path computations
_LARGE_GRAPH_THRESHOLD = 2_000_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SplitStats(NamedTuple):
    split:                str
    num_triples:          int
    num_entities:         int
    num_heads:            int
    num_tails:            int
    num_relations:        int
    avg_facts_per_entity: float
    avg_out_degree:       float
    avg_in_degree:        float


class PathStats(NamedTuple):
    """Path-related properties derived from the training graph.

    All reachability numbers use the *directed* training graph so they
    reflect what a directed path-traversal model (PathE/SJP) can observe.
    BFS for connectivity / shortest-path uses the undirected version.
    """
    # --- Test-fact reachability (directed train graph) ---
    test_reach_1hop:    float   # % test (h,t) pairs with direct train edge
    test_reach_2hop:    float   # % reachable within 2 directed hops
    test_reach_3hop:    float   # % reachable within 3 directed hops
    n_test_pairs:       int     # # pairs evaluated (all or sampled)

    # --- Compositional richness (directed) ---
    unique_relpaths_2:  int     # |{(r1,r2)}| distinct 2-hop relation sequences
    unique_relpaths_3:  int     # |{(r1,r2,r3)}| (sampled if large graph)
    relpaths_3_sampled: bool    # True if 3-hop was computed on a sample

    # --- Path redundancy (directed) ---
    avg_paths_per_pair: float   # avg # of distinct paths (len≤3) per reachable test pair
    n_pairs_redundancy: int     # # pairs used for redundancy estimate

    # --- Graph connectivity (undirected) ---
    pct_largest_cc:     float   # % entities in largest weakly-connected component

    # --- Structural (undirected, sampled) ---
    avg_clustering:     float   # avg local clustering coefficient
    n_clustering:       int     # # entities sampled for clustering
    avg_shortest_path:  float   # avg shortest-path length (reachable pairs)
    n_shortest_path:    int     # # sampled pairs for avg shortest path


class DatasetStats(NamedTuple):
    dataset:    str
    splits:     Dict[str, SplitStats]
    combined:   SplitStats
    path_stats: Optional[PathStats]   # None if --no-path-stats


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_SPLIT_FILES = {
    "train": ["train.txt"],
    "valid": ["valid.txt", "val.txt"],
    "test":  ["test.txt"],
}


def _iter_splits(data_dir: Path) -> Iterator[Tuple[str, List[Tuple[str, str, str]]]]:
    for split, candidates in _SPLIT_FILES.items():
        for fname in candidates:
            fpath = data_dir / fname
            if fpath.exists():
                yield split, _read_triples(fpath)
                break


def _read_triples(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# ---------------------------------------------------------------------------
# General statistics
# ---------------------------------------------------------------------------

def compute_split_stats(split: str, triples: List[Tuple[str, str, str]]) -> SplitStats:
    heads     = Counter(h for h, _, _ in triples)
    tails     = Counter(t for _, _, t in triples)
    relations = {r for _, r, _ in triples}
    entities  = set(heads) | set(tails)
    n, ne, nh, nt = len(triples), len(entities), len(heads), len(tails)
    return SplitStats(
        split=split,
        num_triples=n,
        num_entities=ne,
        num_heads=nh,
        num_tails=nt,
        num_relations=len(relations),
        avg_facts_per_entity=n / ne if ne else 0.0,
        avg_out_degree=n / nh if nh else 0.0,
        avg_in_degree=n / nt if nt else 0.0,
    )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_out_adj(triples: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Directed adjacency: entity → [(relation, tail)]."""
    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for h, r, t in triples:
        adj[h].append((r, t))
    return adj


def _build_undirected_neighbors(triples: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    """Undirected adjacency: entity → {neighbor entities} (no relation labels)."""
    nbrs: Dict[str, Set[str]] = defaultdict(set)
    for h, _, t in triples:
        nbrs[h].add(t)
        nbrs[t].add(h)
    return nbrs


# ---------------------------------------------------------------------------
# Path computations
# ---------------------------------------------------------------------------

def _bfs_directed_hops(
    start: str,
    target: str,
    out_adj: Dict[str, List[Tuple[str, str]]],
    max_hops: int,
) -> Optional[int]:
    """Return the shortest directed-path hop count from start to target, or None."""
    if start == target:
        return 0
    visited = {start}
    frontier = {start}
    for hop in range(1, max_hops + 1):
        nxt: Set[str] = set()
        for node in frontier:
            for _, nbr in out_adj.get(node, []):
                if nbr == target:
                    return hop
                if nbr not in visited:
                    visited.add(nbr)
                    nxt.add(nbr)
        frontier = nxt
        if not frontier:
            break
    return None


def _count_directed_paths(
    start: str,
    target: str,
    out_adj: Dict[str, List[Tuple[str, str]]],
    max_depth: int,
    max_count: int = 200,
) -> int:
    """Count distinct directed paths from start to target (length ≤ max_depth).

    Paths are acyclic (no entity repeated).  Stops early at max_count.
    """
    count = 0
    # stack items: (current_node, visited_on_path)
    stack = [(start, frozenset([start]), 0)]
    while stack and count < max_count:
        node, visited, depth = stack.pop()
        if depth >= max_depth:
            continue
        for _, nbr in out_adj.get(node, []):
            if nbr == target:
                count += 1
                if count >= max_count:
                    break
            elif nbr not in visited:
                stack.append((nbr, visited | {nbr}, depth + 1))
    return count


def _unique_relpaths_2hop(
    out_adj: Dict[str, List[Tuple[str, str]]],
    sample_heads: Optional[List[str]] = None,
) -> int:
    """Count distinct (r1, r2) relation-path sequences of length 2."""
    heads = sample_heads if sample_heads is not None else list(out_adj.keys())
    paths: Set[Tuple[str, str]] = set()
    for h in heads:
        for r1, mid in out_adj.get(h, []):
            for r2, _ in out_adj.get(mid, []):
                paths.add((r1, r2))
    return len(paths)


def _unique_relpaths_3hop(
    out_adj: Dict[str, List[Tuple[str, str]]],
    sample_heads: List[str],
) -> int:
    """Count distinct (r1, r2, r3) relation-path sequences (sampled start entities)."""
    paths: Set[Tuple[str, str, str]] = set()
    for h in sample_heads:
        for r1, mid1 in out_adj.get(h, []):
            for r2, mid2 in out_adj.get(mid1, []):
                for r3, _ in out_adj.get(mid2, []):
                    paths.add((r1, r2, r3))
    return len(paths)


def _largest_cc_fraction(
    nbrs: Dict[str, Set[str]],
    all_entities: Set[str],
) -> float:
    """Fraction of entities in the largest weakly-connected component (BFS)."""
    if not all_entities:
        return 0.0
    visited: Set[str] = set()
    largest = 0
    for seed in all_entities:
        if seed in visited:
            continue
        # BFS
        component = {seed}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for nbr in nbrs.get(node, set()):
                if nbr not in component:
                    component.add(nbr)
                    queue.append(nbr)
        visited |= component
        largest = max(largest, len(component))
    return largest / len(all_entities)


def _avg_clustering(
    nbrs: Dict[str, Set[str]],
    sample: List[str],
) -> float:
    """Average local clustering coefficient over a sample of entities.

    For node v with degree d: C(v) = (# edges among neighbors) / (d*(d-1)/2).
    """
    coeffs = []
    for v in sample:
        nv = nbrs.get(v, set())
        d = len(nv)
        if d < 2:
            coeffs.append(0.0)
            continue
        edges = sum(1 for u in nv for w in nv if w in nbrs.get(u, set()) and u < w)
        coeffs.append(edges / (d * (d - 1) / 2))
    return sum(coeffs) / len(coeffs) if coeffs else 0.0


def _avg_shortest_path(
    nbrs: Dict[str, Set[str]],
    pairs: List[Tuple[str, str]],
) -> float:
    """Average shortest undirected path length for a list of (src, tgt) pairs.

    Pairs where no path exists are excluded from the average.
    """
    lengths = []
    for src, tgt in pairs:
        if src == tgt:
            continue
        # BFS
        visited = {src}
        frontier = {src}
        dist = 0
        found = False
        while frontier and not found:
            dist += 1
            nxt: Set[str] = set()
            for node in frontier:
                for nbr in nbrs.get(node, set()):
                    if nbr == tgt:
                        lengths.append(dist)
                        found = True
                        break
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.add(nbr)
                if found:
                    break
            frontier = nxt
    return sum(lengths) / len(lengths) if lengths else float("nan")


# ---------------------------------------------------------------------------
# Main path-stats computation
# ---------------------------------------------------------------------------

def compute_path_stats(
    train_triples: List[Tuple[str, str, str]],
    test_triples:  List[Tuple[str, str, str]],
    sample_pairs:    int = 2000,
    sample_entities: int = 2000,
    seed: int = 42,
) -> PathStats:
    """Compute all path-related statistics for one dataset.

    Args:
        train_triples:   Training-split triples used to build the graph.
        test_triples:    Test-split triples whose (h,t) pairs are probed.
        sample_pairs:    Max # of (h,t) pairs for reachability / redundancy
                         and shortest-path sampling.
        sample_entities: Max # of entities for 2-hop path enumeration on
                         large graphs, and for clustering coefficient.
        seed:            RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    is_large = len(train_triples) > _LARGE_GRAPH_THRESHOLD

    out_adj  = _build_out_adj(train_triples)
    undirected = _build_undirected_neighbors(train_triples)
    train_entities = set(out_adj.keys()) | {t for _, _, t in train_triples}

    # ------------------------------------------------------------------ #
    # 1. Test-fact reachability                                           #
    # ------------------------------------------------------------------ #
    test_pairs = [(h, t) for h, _, t in test_triples if h != t]
    if len(test_pairs) > sample_pairs:
        test_pairs = rng.sample(test_pairs, sample_pairs)
    n_test = len(test_pairs)

    reach1 = reach2 = reach3 = 0
    for h, t in test_pairs:
        d = _bfs_directed_hops(h, t, out_adj, max_hops=3)
        if d is not None:
            if d <= 1: reach1 += 1
            if d <= 2: reach2 += 1
            if d <= 3: reach3 += 1

    # ------------------------------------------------------------------ #
    # 2. Unique relation paths (length 2 and 3)                          #
    # ------------------------------------------------------------------ #
    all_heads = list(out_adj.keys())

    # 2-hop: full graph if small, sampled otherwise
    heads_2hop = (rng.sample(all_heads, sample_entities)
                  if is_large and len(all_heads) > sample_entities
                  else None)
    n_rp2 = _unique_relpaths_2hop(out_adj, sample_heads=heads_2hop)

    # 3-hop: always sampled
    heads_3hop = (rng.sample(all_heads, min(sample_entities // 2, len(all_heads))))
    n_rp3 = _unique_relpaths_3hop(out_adj, sample_heads=heads_3hop)
    rp3_sampled = len(heads_3hop) < len(all_heads)

    # ------------------------------------------------------------------ #
    # 3. Path redundancy for reachable test pairs                        #
    # ------------------------------------------------------------------ #
    reachable_pairs = [
        (h, t) for h, t in test_pairs
        if _bfs_directed_hops(h, t, out_adj, max_hops=3) is not None
    ]
    redundancy_pairs = (rng.sample(reachable_pairs,
                                   min(sample_pairs // 2, len(reachable_pairs)))
                        if reachable_pairs else [])
    path_counts = [
        _count_directed_paths(h, t, out_adj, max_depth=3)
        for h, t in redundancy_pairs
    ]
    avg_paths = sum(path_counts) / len(path_counts) if path_counts else 0.0

    # ------------------------------------------------------------------ #
    # 4. Largest weakly-connected component                              #
    # ------------------------------------------------------------------ #
    pct_cc = _largest_cc_fraction(undirected, train_entities)

    # ------------------------------------------------------------------ #
    # 5. Clustering coefficient (sampled)                                #
    # ------------------------------------------------------------------ #
    clust_sample = rng.sample(list(train_entities),
                              min(sample_entities, len(train_entities)))
    avg_clust = _avg_clustering(undirected, clust_sample)

    # ------------------------------------------------------------------ #
    # 6. Average shortest path (sampled undirected pairs)                #
    # ------------------------------------------------------------------ #
    entity_list = list(train_entities)
    sp_pairs: List[Tuple[str, str]] = []
    while len(sp_pairs) < min(sample_pairs // 4, 500):
        a, b = rng.sample(entity_list, 2)
        sp_pairs.append((a, b))
    avg_sp = _avg_shortest_path(undirected, sp_pairs)

    return PathStats(
        test_reach_1hop=reach1 / n_test if n_test else 0.0,
        test_reach_2hop=reach2 / n_test if n_test else 0.0,
        test_reach_3hop=reach3 / n_test if n_test else 0.0,
        n_test_pairs=n_test,
        unique_relpaths_2=n_rp2,
        unique_relpaths_3=n_rp3,
        relpaths_3_sampled=rp3_sampled,
        avg_paths_per_pair=avg_paths,
        n_pairs_redundancy=len(redundancy_pairs),
        pct_largest_cc=pct_cc,
        avg_clustering=avg_clust,
        n_clustering=len(clust_sample),
        avg_shortest_path=avg_sp,
        n_shortest_path=len(sp_pairs),
    )


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    dataset: str,
    data_dir: Path,
    compute_paths: bool = True,
    sample_pairs: int = 2000,
    sample_entities: int = 2000,
) -> DatasetStats:
    split_stats: Dict[str, SplitStats] = {}
    split_triples: Dict[str, List[Tuple[str, str, str]]] = {}

    for split, triples in _iter_splits(data_dir):
        split_stats[split] = compute_split_stats(split, triples)
        split_triples[split] = triples

    if not split_stats:
        raise FileNotFoundError(f"No triple files found in {data_dir}")

    all_triples = [t for ts in split_triples.values() for t in ts]

    path_stats: Optional[PathStats] = None
    if compute_paths and "train" in split_triples and "test" in split_triples:
        path_stats = compute_path_stats(
            train_triples=split_triples["train"],
            test_triples=split_triples["test"],
            sample_pairs=sample_pairs,
            sample_entities=sample_entities,
        )

    return DatasetStats(
        dataset=dataset,
        splits=split_stats,
        combined=compute_split_stats("combined", all_triples),
        path_stats=path_stats,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_datasets(data_root: Path) -> List[Path]:
    if not data_root.exists():
        return []
    return sorted(p for p in data_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_SPLIT_ORDER = ["train", "valid", "test", "combined"]


def _split_key(name: str) -> int:
    try:
        return _SPLIT_ORDER.index(name)
    except ValueError:
        return len(_SPLIT_ORDER)


def print_dataset_stats(ds: DatasetStats) -> None:
    cols = [
        ("Split",         10),
        ("Triples",       12),
        ("Entities",      10),
        ("Heads",         10),
        ("Tails",         10),
        ("Relations",     11),
        ("Facts/Entity",  13),
        ("Out-deg",       10),
        ("In-deg",         9),
    ]
    header  = "".join(f"{lbl:<{w}}" for lbl, w in cols)
    divider = "─" * len(header)
    sep     = "═" * len(header)

    print(f"\n{sep}")
    print(f"  Dataset : {ds.dataset}")
    print(f"{sep}")
    print(header)
    print(divider)

    all_splits = {**ds.splits, "combined": ds.combined}
    for name, s in sorted(all_splits.items(), key=lambda kv: _split_key(kv[0])):
        if name == "combined":
            print(divider)
        label = f"{name} ←" if name == "combined" else name
        print(
            f"{label:<10}"
            f"{s.num_triples:<12,}"
            f"{s.num_entities:<10,}"
            f"{s.num_heads:<10,}"
            f"{s.num_tails:<10,}"
            f"{s.num_relations:<11,}"
            f"{s.avg_facts_per_entity:<13.2f}"
            f"{s.avg_out_degree:<10.2f}"
            f"{s.avg_in_degree:<9.2f}"
        )
    print(sep)

    if ds.path_stats is not None:
        print_path_stats(ds.path_stats, ds.dataset)


def print_path_stats(ps: PathStats, dataset: str) -> None:
    w = 52  # total width
    sep = "═" * w

    def row(label: str, value: str, note: str = "") -> None:
        note_str = f"  ({note})" if note else ""
        print(f"  {label:<38}{value}{note_str}")

    print(f"\n{sep}")
    print(f"  Path Properties — {dataset}")
    print(f"{'─' * w}")

    print(f"  {'Test-fact reachability (directed train graph)':}")
    row("  @1 hop  (direct train edge)",
        f"{ps.test_reach_1hop:>6.1%}",
        f"n={ps.n_test_pairs:,}")
    row("  @2 hops",
        f"{ps.test_reach_2hop:>6.1%}")
    row("  @3 hops",
        f"{ps.test_reach_3hop:>6.1%}")

    print(f"  {'─' * (w - 2)}")
    print(f"  {'Compositional richness (directed train graph)':}")
    rp2_note = "sampled" if ps.unique_relpaths_2 else ""
    row("  Unique 2-hop relation paths",
        f"{ps.unique_relpaths_2:>10,}")
    rp3_note = "sampled start entities" if ps.relpaths_3_sampled else "full"
    row("  Unique 3-hop relation paths",
        f"{ps.unique_relpaths_3:>10,}",
        rp3_note)

    print(f"  {'─' * (w - 2)}")
    print(f"  {'Path redundancy (reachable test pairs)':}")
    row("  Avg # distinct paths (len ≤ 3)",
        f"{ps.avg_paths_per_pair:>10.2f}",
        f"n={ps.n_pairs_redundancy:,}")

    print(f"  {'─' * (w - 2)}")
    print(f"  {'Graph structure (undirected train graph)':}")
    row("  Entities in largest component",
        f"{ps.pct_largest_cc:>6.1%}")
    row("  Avg clustering coefficient",
        f"{ps.avg_clustering:>10.4f}",
        f"n={ps.n_clustering:,}")
    sp_str = (f"{ps.avg_shortest_path:.2f}"
              if ps.avg_shortest_path == ps.avg_shortest_path  # not nan
              else "n/a")
    row("  Avg shortest path length",
        f"{sp_str:>10}",
        f"n={ps.n_shortest_path:,}")

    print(sep)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(all_stats: List[DatasetStats], output_path: str) -> None:
    general_fields = [
        "dataset", "split",
        "num_triples", "num_entities", "num_heads", "num_tails", "num_relations",
        "avg_facts_per_entity", "avg_out_degree", "avg_in_degree",
    ]
    path_fields = [
        "test_reach_1hop", "test_reach_2hop", "test_reach_3hop", "n_test_pairs",
        "unique_relpaths_2", "unique_relpaths_3", "relpaths_3_sampled",
        "avg_paths_per_pair", "n_pairs_redundancy",
        "pct_largest_cc", "avg_clustering", "n_clustering",
        "avg_shortest_path", "n_shortest_path",
    ]
    has_path = any(ds.path_stats is not None for ds in all_stats)
    fieldnames = general_fields + (path_fields if has_path else [])

    rows = []
    for ds in all_stats:
        for split_name, s in {**ds.splits, "combined": ds.combined}.items():
            row: Dict = {
                "dataset":              ds.dataset,
                "split":                split_name,
                "num_triples":          s.num_triples,
                "num_entities":         s.num_entities,
                "num_heads":            s.num_heads,
                "num_tails":            s.num_tails,
                "num_relations":        s.num_relations,
                "avg_facts_per_entity": round(s.avg_facts_per_entity, 4),
                "avg_out_degree":       round(s.avg_out_degree, 4),
                "avg_in_degree":        round(s.avg_in_degree, 4),
            }
            if has_path:
                ps = ds.path_stats
                if ps is not None and split_name == "test":
                    row.update({
                        "test_reach_1hop":    round(ps.test_reach_1hop, 4),
                        "test_reach_2hop":    round(ps.test_reach_2hop, 4),
                        "test_reach_3hop":    round(ps.test_reach_3hop, 4),
                        "n_test_pairs":       ps.n_test_pairs,
                        "unique_relpaths_2":  ps.unique_relpaths_2,
                        "unique_relpaths_3":  ps.unique_relpaths_3,
                        "relpaths_3_sampled": ps.relpaths_3_sampled,
                        "avg_paths_per_pair": round(ps.avg_paths_per_pair, 4),
                        "n_pairs_redundancy": ps.n_pairs_redundancy,
                        "pct_largest_cc":     round(ps.pct_largest_cc, 4),
                        "avg_clustering":     round(ps.avg_clustering, 4),
                        "n_clustering":       ps.n_clustering,
                        "avg_shortest_path":  round(ps.avg_shortest_path, 4)
                                              if ps.avg_shortest_path == ps.avg_shortest_path
                                              else "",
                        "n_shortest_path":    ps.n_shortest_path,
                    })
                else:
                    row.update({f: "" for f in path_fields})
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print KG dataset statistics, including path-related properties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Root directory containing dataset sub-folders. "
             "Default: <repo>/code/iswc_data/standard",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Name of a specific dataset (e.g. 'fb15k237'). "
             "Processes all datasets if omitted.",
    )
    parser.add_argument(
        "--no-path-stats", action="store_true",
        help="Skip path-property computation (faster, for quick general stats).",
    )
    parser.add_argument(
        "--sample-pairs", type=int, default=2000, metavar="N",
        help="Max # of (h,t) pairs sampled for reachability / redundancy / "
             "shortest-path estimation.",
    )
    parser.add_argument(
        "--sample-entities", type=int, default=2000, metavar="N",
        help="Max # of entities sampled for 2-hop/3-hop path enumeration "
             "and clustering coefficient on large graphs.",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE.csv",
        help="Save results to this CSV file in addition to printing.",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_dir) if args.data_dir else _DEFAULT_DATA_ROOT
    if not data_root.exists():
        sys.exit(f"Data root not found: {data_root}")

    if args.dataset:
        dataset_dirs = [data_root / args.dataset]
        if not dataset_dirs[0].exists():
            sys.exit(f"Dataset directory not found: {dataset_dirs[0]}")
    else:
        dataset_dirs = discover_datasets(data_root)
        if not dataset_dirs:
            sys.exit(f"No dataset directories found under {data_root}")

    print(f"Data root    : {data_root}")
    print(f"Datasets     : {[d.name for d in dataset_dirs]}")
    print(f"Path stats   : {'disabled' if args.no_path_stats else 'enabled'}")
    if not args.no_path_stats:
        print(f"Sample pairs : {args.sample_pairs:,}   "
              f"Sample entities : {args.sample_entities:,}")

    all_stats: List[DatasetStats] = []
    for ddir in dataset_dirs:
        print(f"\nProcessing '{ddir.name}' …", end=" ", flush=True)
        try:
            ds_stats = compute_dataset_stats(
                dataset=ddir.name,
                data_dir=ddir,
                compute_paths=not args.no_path_stats,
                sample_pairs=args.sample_pairs,
                sample_entities=args.sample_entities,
            )
            all_stats.append(ds_stats)
            print("done.")
            print_dataset_stats(ds_stats)
        except FileNotFoundError as exc:
            print(f"SKIPPED — {exc}")
        except Exception as exc:
            print(f"ERROR — {exc}")
            raise

    if args.output and all_stats:
        save_csv(all_stats, args.output)


if __name__ == "__main__":
    main()
