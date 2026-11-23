from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rank", type=int, default=200)
    parser.add_argument(
        "--max-history",
        type=int,
        help="Override the sequence-history cap for matched comparisons.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("common", "full"),
        default="common",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-user-ranks", type=Path)
    return parser.parse_args()


def recall_metrics(ranks: np.ndarray, budgets: list[int]) -> dict:
    result = {"users": int(len(ranks))}
    for budget in budgets:
        hit = ranks <= budget
        ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
        result[f"recall@{budget}"] = float(hit.mean()) if len(hit) else 0.0
        result[f"ndcg@{budget}"] = float(ndcg.mean()) if len(ndcg) else 0.0
    return result


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
    if args.candidate_scope == "common":
        candidate_rids = mapping["mapped_rids"].astype(np.int64, copy=False)
    else:
        candidate_rids = mapping["retrieval_to_rid"].astype(np.int64, copy=False)
    candidate_tensor = torch.from_numpy(candidate_rids).to(device)
    candidate_vectors = F.normalize(model.encode_items(candidate_tensor), dim=-1)
    rid_to_position = {}
    for position, rid in enumerate(candidate_rids.tolist()):
        if int(rid) > 0:
            rid_to_position[int(rid)] = position
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
        collate_fn=partial(
            collate_click_samples,
            maxlen=(
                args.max_history + 1
                if args.max_history is not None
                else model_args["maxlen"]
            ),
        ),
    )

    ranks: list[int] = []
    user_ids: list[int] = []
    target_rids: list[int] = []
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
            target_positions = torch.tensor(
                [rid_to_position[int(target)] for target in targets],
                device=device,
            )
            target_scores = scores[
                torch.arange(len(target_positions), device=device),
                target_positions,
            ]
            batch_ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(batch_ranks.cpu().tolist())
            user_ids.extend(moved["user_ids"].cpu().tolist())
            target_rids.extend(targets.tolist())

    rank_array = np.asarray(ranks, dtype=np.int64)
    budgets = sorted(
        {
            budget
            for budget in (10, 50, 100, 154, 200, 500, 1000, 2000, args.max_rank)
            if budget <= args.max_rank
        }
    )
    result = {
        "candidate_scope": args.candidate_scope,
        "candidate_items": int(len(candidate_rids)),
        "max_rank": args.max_rank,
        "metrics": recall_metrics(rank_array, budgets),
        "evaluation_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.save_user_ranks is not None:
        args.save_user_ranks.parent.mkdir(parents=True, exist_ok=True)
        args.save_user_ranks.write_text(
            json.dumps(
                {
                    "users": [
                        {
                            "user_id": user_id,
                            "target_rid": target_rid,
                            "vector_rank": rank,
                        }
                        for user_id, target_rid, rank in zip(
                            user_ids,
                            target_rids,
                            ranks,
                        )
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
