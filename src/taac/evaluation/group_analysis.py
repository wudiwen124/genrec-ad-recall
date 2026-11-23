from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from taac.data.candidate_map import (
    build_rid_to_candidate_index_array,
    load_candidate_mapping,
)
from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_ARRAY_MAX_LENGTHS,
    build_item_feature_matrix,
    build_user_feature_store,
    item_feature_vocab_sizes,
    load_indexer,
    user_feature_vocab_sizes,
)
from taac.models.sasrec import SASRecRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    user_count = len(indexer["u"])
    item_features, _ = build_item_feature_matrix(args.data_root, args.feature_cache_dir)
    user_scalars, user_arrays, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir,
    )
    scalar_vocab, array_vocab = user_feature_vocab_sizes(indexer)
    model = SASRecRetriever(
        user_count=user_count,
        item_count=item_count,
        maxlen=train_args["maxlen"],
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        dropout=train_args["dropout"],
        item_feature_vocab_sizes=item_feature_vocab_sizes(indexer),
        user_scalar_vocab_sizes=scalar_vocab,
        user_array_vocab_sizes=array_vocab,
        user_array_max_lengths=[
            USER_ARRAY_MAX_LENGTHS[feature_id] for feature_id in USER_ARRAY_FEATURE_IDS
        ],
        model_type=train_args.get("model_type", "sasrec"),
        use_action=train_args.get("use_action", True),
        use_time=train_args.get("use_time", True),
        use_relative_bias=train_args.get("use_relative_bias", True),
        norm_first=train_args.get("norm_first", True),
    ).to(device)
    model.item_features.copy_(torch.from_numpy(item_features))
    model.user_scalar_features.copy_(torch.from_numpy(user_scalars))
    for index, feature_id in enumerate(USER_ARRAY_FEATURE_IDS):
        getattr(model, f"user_array_features_{index}").copy_(
            torch.from_numpy(user_arrays[feature_id])
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, train_args


def metrics(ranks: np.ndarray) -> dict[str, float]:
    if len(ranks) == 0:
        return {"users": 0, "hr@10": 0.0, "ndcg@10": 0.0, "score": 0.0}
    hit = ranks <= 10
    hr = float(hit.mean())
    ndcg = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    return {
        "users": int(len(ranks)),
        "hr@10": hr,
        "ndcg@10": ndcg,
        "score": 0.31 * hr + 0.69 * ndcg,
    }


def bootstrap_ci(ranks: np.ndarray, trials: int, seed: int = 2026) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(trials):
        sample = rng.choice(ranks, size=len(ranks), replace=True)
        samples.append(metrics(sample)["score"])
    return {
        "score_mean": float(np.mean(samples)),
        "score_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    model, train_args = load_model(args, device)
    mapping = load_candidate_mapping(args.data_root)
    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    rid_index = torch.from_numpy(
        build_rid_to_candidate_index_array(mapping["retrieval_to_rid"], item_count)
    ).to(device)
    candidate_rids = torch.from_numpy(mapping["retrieval_to_rid"]).to(device)
    candidate_vectors = torch.nn.functional.normalize(
        model.encode_items(candidate_rids),
        dim=-1,
    )

    dataset = ClickSampleDataset(
        args.samples_root / args.split / "part-00000.parquet",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=train_args["maxlen"]),
    )

    all_ranks: list[np.ndarray] = []
    all_history_lengths: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            query = model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["timestamps"],
                moved["valid_mask"],
            )
            scores = torch.nn.functional.normalize(query, dim=-1) @ candidate_vectors.T
            history_indices = rid_index[moved["sequence"]]
            history_valid = history_indices >= 0
            rows = (
                torch.arange(scores.shape[0], device=device)
                .unsqueeze(1)
                .expand_as(history_indices)
            )
            scores[rows[history_valid], history_indices[history_valid]] = -torch.inf
            target_indices = rid_index[moved["targets"]]
            valid = target_indices >= 0
            if not bool(valid.any()):
                continue
            row_scores = scores[valid]
            targets = target_indices[valid]
            target_scores = row_scores[
                torch.arange(targets.numel(), device=device),
                targets,
            ]
            ranks = (row_scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            history_lengths = moved["valid_mask"].sum(dim=1)[valid] - 1
            all_ranks.append(ranks.cpu().numpy())
            all_history_lengths.append(history_lengths.cpu().numpy())
            all_targets.append(moved["targets"][valid].cpu().numpy())

    ranks = np.concatenate(all_ranks)
    history_lengths = np.concatenate(all_history_lengths)
    targets = np.concatenate(all_targets)
    train_targets = pq.read_table(
        args.samples_root / "train" / "part-00000.parquet",
        columns=["target_item_id"],
    ).column(0).to_numpy()
    target_frequency = np.bincount(train_targets, minlength=item_count + 1)
    frequencies = target_frequency[targets]

    low_len, high_len = np.quantile(history_lengths, [1 / 3, 2 / 3])
    length_groups = {
        "short": ranks[history_lengths <= low_len],
        "medium": ranks[(history_lengths > low_len) & (history_lengths <= high_len)],
        "long": ranks[history_lengths > high_len],
    }
    frequency_groups = {
        "unseen_in_train_targets": ranks[frequencies == 0],
        "low_1_5": ranks[(frequencies >= 1) & (frequencies <= 5)],
        "medium_6_100": ranks[(frequencies > 5) & (frequencies <= 100)],
        "head_gt_100": ranks[frequencies > 100],
    }

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "overall": metrics(ranks),
        "bootstrap": bootstrap_ci(ranks, args.bootstrap),
        "history_length_groups": {
            name: metrics(group) for name, group in length_groups.items()
        },
        "target_popularity_groups": {
            name: metrics(group) for name, group in frequency_groups.items()
        },
        "notes": [
            "Targets are candidate-covered historical clicks; true candidate cold-start Items cannot be targets under this local protocol.",
            "History length is used as the available user-activity proxy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
