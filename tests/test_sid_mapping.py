from __future__ import annotations

import numpy as np

from taac.sid.analyze_recall_budget import rrf_ranks
from taac.sid.evaluate_sid import complementarity_metrics, union_prefix_metrics
from taac.sid.evaluate_vector_recall import recall_metrics
from taac.sid.tokenizer import build_collision_suffix


def test_collision_suffix_makes_every_sid_unique() -> None:
    codes = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [2, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    suffix, metadata = build_collision_suffix(codes, base=256)
    full_sids = np.concatenate([codes, suffix[:, None]], axis=1)
    unique_rows = np.unique(full_sids, axis=0)

    assert len(unique_rows) == len(codes)
    assert metadata["unique_codes_before_suffix"] == 3
    assert metadata["suffix_vocab_size"] == 2


def test_complementarity_detects_sid_only_hits() -> None:
    vector_top10 = [[1, 2], [3, 4], [5, 6], [9, 10]]
    sid_top10 = [[3, 4], [4, 3], [7, 8], [11, 12]]
    metrics = complementarity_metrics(
        vector_topk=vector_top10,
        sid_topk=sid_top10,
        targets=[1, 4, 7, 99],
        top_k=10,
    )

    assert metrics["vector_hit_users"] == 2
    assert metrics["sid_hit_users"] == 2
    assert metrics["both_hit_users"] == 1
    assert metrics["vector_only_hit_users"] == 1
    assert metrics["sid_only_hit_users"] == 1
    assert metrics["neither_hit_users"] == 1
    assert metrics["union_hit_users"] == 3


def test_balanced_union_uses_per_route_prefix() -> None:
    vector_topk = [[1, 2, 3], [4, 5, 6]]
    sid_topk = [[3, 4, 5], [1, 2, 6]]
    metrics = union_prefix_metrics(
        vector_topk=vector_topk,
        sid_topk=sid_topk,
        targets=[2, 5],
        prefix_per_route=1,
    )

    assert metrics["vector_hit_rate"] == 0.0
    assert metrics["sid_hit_rate"] == 0.0
    assert metrics["union_hit_rate"] == 0.0
    assert metrics["max_union_size"] == 2


def test_vector_recall_metrics_support_multiple_budgets() -> None:
    ranks = np.asarray([1, 25, 75, 101])
    metrics = recall_metrics(ranks, [50, 100])

    assert metrics["recall@50"] == 0.5
    assert metrics["recall@100"] == 0.75


def test_rrf_rewards_items_present_in_both_routes() -> None:
    fused = rrf_ranks(
        vector_rids=[1, 2, 3],
        sid_rids=[3, 4],
        vector_limit=3,
        sid_limit=2,
    )

    assert fused == [3, 1, 2, 4]
