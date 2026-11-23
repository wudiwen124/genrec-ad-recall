from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from taac.data.candidate_map import load_candidate_mapping
from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.evaluation.group_analysis import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vector-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--sid-user-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-budget", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def ndcg_at(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def rrf_ranks(
    vector_rids: list[int],
    sid_rids: list[int],
    vector_limit: int,
    sid_limit: int,
    rrf_constant: int = 60,
) -> list[int]:
    scores: dict[int, float] = {}
    for rank, rid in enumerate(vector_rids[:vector_limit], start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_constant + rank)
    for rank, rid in enumerate(sid_rids[:sid_limit], start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_constant + rank)
    return [
        rid
        for rid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def bootstrap_mean_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    samples: int = 1000,
    seed: int = 2026,
) -> dict[str, float]:
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(delta), size=len(delta))
        means[index] = delta[sampled].mean()
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda")
    model, model_args = load_model(
        type(
            "A",
            (),
            {
                "data_root": args.data_root,
                "checkpoint": args.vector_checkpoint,
                "feature_cache_dir": args.feature_cache_dir,
            },
        )(),
        device,
    )
    mapping = load_candidate_mapping(args.data_root)
    candidate_rids = mapping["mapped_rids"].astype(np.int64, copy=False)
    candidate_tensor = torch.from_numpy(candidate_rids).to(device)
    candidate_vectors = F.normalize(model.encode_items(candidate_tensor), dim=-1)
    rid_to_position = {
        int(rid): position for position, rid in enumerate(candidate_rids.tolist())
    }
    sid_payload = json.loads(args.sid_user_results.read_text(encoding="utf-8"))
    sid_by_user = {int(row["user_id"]): row for row in sid_payload["users"]}

    dataset = ClickSampleDataset(
        args.samples_root / "test" / "part-00000.parquet",
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=model_args["maxlen"]),
    )
    budgets = [50, 100, 200, 500, 1000, 2000]
    budgets = [budget for budget in budgets if budget <= args.max_budget]
    target_ranks: list[int] = []
    sid_hits: list[bool] = []
    sid_coverage: dict[int, list[float]] = {budget: [] for budget in budgets}
    rrf_variants = {
        "rrf_50_50": (50, 50),
        "rrf_100_100": (100, 100),
        "rrf_500_100": (500, 100),
    }
    rrf_results: dict[str, dict[str, list[float] | list[int]]] = {
        name: {"target_rank": [], "pool_size": []} for name in rrf_variants
    }

    with torch.no_grad():
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            history_sets = [
                set(int(item) for item in sequence if int(item) != 0)
                for sequence in moved["sequence"].cpu().tolist()
            ]
            targets = moved["targets"].cpu().numpy()
            user_ids = moved["user_ids"].cpu().tolist()
            query = model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["timestamps"],
                moved["valid_mask"],
            )
            scores = F.normalize(query, dim=-1) @ candidate_vectors.T
            for row_index, history in enumerate(history_sets):
                positions = [
                    rid_to_position[rid]
                    for rid in history
                    if rid in rid_to_position
                ]
                if positions:
                    scores[row_index, positions] = -torch.inf
            top_positions = torch.topk(
                scores,
                k=min(args.max_budget, scores.shape[1]),
                dim=1,
            ).indices.cpu().tolist()

            for row_index, (target, user_id) in enumerate(zip(targets, user_ids)):
                sid_row = sid_by_user[int(user_id)]
                if int(sid_row["target_rid"]) != int(target):
                    raise ValueError("SID user result order/target does not match test sample")
                sid_rids = [int(rid) for rid in sid_row["sid_topk"]]
                positions = top_positions[row_index]
                position_to_rank = {
                    int(position): rank
                    for rank, position in enumerate(positions, start=1)
                }
                target_rank = position_to_rank.get(
                    int(rid_to_position[int(target)]),
                    args.max_budget + 1,
                )
                target_ranks.append(target_rank)
                sid_hits.append(int(target) in sid_rids)

                for budget in budgets:
                    covered = sum(
                        position_to_rank.get(rid_to_position[rid], args.max_budget + 1)
                        <= budget
                        for rid in sid_rids
                    )
                    sid_coverage[budget].append(covered / len(sid_rids))

                vector_rids = [int(candidate_rids[position]) for position in positions]
                for name, (vector_limit, sid_limit) in rrf_variants.items():
                    fused = rrf_ranks(
                        vector_rids,
                        sid_rids,
                        vector_limit=vector_limit,
                        sid_limit=sid_limit,
                    )
                    try:
                        fused_rank = fused.index(int(target)) + 1
                    except ValueError:
                        fused_rank = 1_000_000
                    rrf_results[name]["target_rank"].append(fused_rank)
                    rrf_results[name]["pool_size"].append(len(fused))

    rank_array = np.asarray(target_ranks, dtype=np.int64)
    sid_hit_array = np.asarray(sid_hits, dtype=np.bool_)
    budget_metrics = {}
    for budget in budgets:
        n4_hit = rank_array <= budget
        budget_metrics[str(budget)] = {
            "n4_recall": float(n4_hit.mean()),
            "sid_hit_users": int(sid_hit_array.sum()),
            "both_hit_users": int((n4_hit & sid_hit_array).sum()),
            "n4_only_hit_users": int((n4_hit & ~sid_hit_array).sum()),
            "sid_only_hit_users": int((~n4_hit & sid_hit_array).sum()),
            "union_hit_users": int((n4_hit | sid_hit_array).sum()),
            "sid_only_share_of_sid_hits": (
                float(((~n4_hit) & sid_hit_array).sum() / sid_hit_array.sum())
                if sid_hit_array.sum()
                else 0.0
            ),
            "mean_sid_top100_coverage_in_n4": float(
                np.mean(sid_coverage[budget])
            ),
        }

    n4_metrics = {}
    for budget in budgets:
        hit = rank_array <= budget
        ndcg = np.where(hit, 1.0 / np.log2(rank_array + 1.0), 0.0)
        n4_metrics[f"recall@{budget}"] = float(hit.mean())
        n4_metrics[f"ndcg@{budget}"] = float(ndcg.mean())

    rrf_metrics = {}
    for name, data in rrf_results.items():
        fused_ranks = np.asarray(data["target_rank"], dtype=np.int64)
        pool_sizes = np.asarray(data["pool_size"], dtype=np.int64)
        n4_same_pool_recall = float(
            np.mean(rank_array <= pool_sizes)
        )
        union_candidate_recall = float(
            np.mean(fused_ranks <= pool_sizes)
        )
        rrf_metrics[name] = {
            "mean_candidate_pool": float(pool_sizes.mean()),
            "n4_recall_at_same_pool": n4_same_pool_recall,
            "union_candidate_recall": union_candidate_recall,
            "candidate_recall_delta_vs_n4_same_pool": (
                union_candidate_recall - n4_same_pool_recall
            ),
            "hr@50": float((fused_ranks <= 50).mean()),
            "ndcg@50": float(
                np.mean(
                    [
                        ndcg_at(int(rank), 50) for rank in fused_ranks
                    ]
                )
            ),
            "hr@100": float((fused_ranks <= 100).mean()),
            "ndcg@100": float(
                np.mean(
                    [
                        ndcg_at(int(rank), 100) for rank in fused_ranks
                    ]
                )
            ),
            "bootstrap_hr50_vs_n4_top50": bootstrap_mean_delta(
                (rank_array <= 50).astype(np.float64),
                (fused_ranks <= 50).astype(np.float64),
            ),
        }

    result = {
        "users": len(target_ranks),
        "candidate_items": len(candidate_rids),
        "n4_recall_curve": n4_metrics,
        "sid_relative_to_n4": budget_metrics,
        "rrf": rrf_metrics,
        "evaluation_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
