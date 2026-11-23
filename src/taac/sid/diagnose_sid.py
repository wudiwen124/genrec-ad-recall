from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.sid.evaluate_sid import load_sid_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--sid-checkpoint", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-valid-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def distribution_metrics(counts: np.ndarray) -> dict:
    total = int(counts.sum())
    nonzero = counts[counts > 0]
    probabilities = nonzero.astype(np.float64) / max(total, 1)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = float(np.log(max(len(counts), 2)))
    sorted_counts = np.sort(counts)[::-1]
    top10_count = int(min(10, len(sorted_counts)))
    return {
        "items": total,
        "used_codes": int(len(nonzero)),
        "vocab_size": int(len(counts)),
        "utilization": float(len(nonzero) / len(counts)),
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
        "top1_share": float(sorted_counts[0] / total) if total else 0.0,
        "top10_share": (
            float(sorted_counts[:top10_count].sum() / total) if total else 0.0
        ),
        "max_count": int(sorted_counts[0]) if total else 0,
        "top_codes": [
            {"code": int(code), "count": int(count)}
            for code, count in sorted(
                enumerate(counts.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
            if count > 0
        ],
    }


def kl_divergence(target: np.ndarray, prediction: np.ndarray) -> float:
    target_prob = target.astype(np.float64) + 1e-8
    prediction_prob = prediction.astype(np.float64) + 1e-8
    target_prob /= target_prob.sum()
    prediction_prob /= prediction_prob.sum()
    return float((target_prob * np.log(target_prob / prediction_prob)).sum())


def split_target_codes(samples_root: Path, split: str, item_to_codes: np.ndarray) -> np.ndarray:
    table = pq.read_table(
        samples_root / split / "part-00000.parquet",
        columns=["target_item_id"],
    )
    target_ids = np.asarray(table["target_item_id"].to_numpy(), dtype=np.int64)
    return item_to_codes[target_ids]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    sid = np.load(args.sid_path)
    item_to_codes = sid["item_to_codes"]
    candidate_codes = sid["candidate_codes"]
    metadata = json.loads(args.sid_path.with_suffix(".json").read_text(encoding="utf-8"))
    codebook_sizes = metadata["codebook_sizes"]
    model = load_sid_model(args, device)

    dataset = ClickSampleDataset(
        args.samples_root / "valid" / "part-00000.parquet",
        max_samples=args.max_valid_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=101),
    )

    target_counts = [np.zeros(size, dtype=np.int64) for size in codebook_sizes]
    prediction_counts = [np.zeros(size, dtype=np.int64) for size in codebook_sizes]
    correct = np.zeros(model.num_sid_levels, dtype=np.int64)
    prefix_correct = np.zeros(model.num_sid_levels, dtype=np.int64)
    users = 0
    with torch.no_grad():
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            state = model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["valid_mask"],
            )
            target_codes = model.item_to_codes[moved["targets"]].long()
            logits = model.predict_logits(state, target_codes)
            predictions = torch.stack(
                [item.argmax(dim=-1) for item in logits],
                dim=1,
            )
            matches = predictions == target_codes
            correct += matches.sum(dim=0).cpu().numpy()
            prefix_correct += torch.cumprod(matches, dim=1).sum(dim=0).cpu().numpy()
            users += len(target_codes)
            for level, size in enumerate(codebook_sizes):
                target_counts[level] += np.bincount(
                    target_codes[:, level].cpu().numpy(),
                    minlength=size,
                )
                prediction_counts[level] += np.bincount(
                    predictions[:, level].cpu().numpy(),
                    minlength=size,
                )

    item_counts = [
        np.bincount(item_to_codes[1:, level].astype(np.int64), minlength=size)
        for level, size in enumerate(codebook_sizes)
    ]
    candidate_counts = [
        np.bincount(candidate_codes[:, level].astype(np.int64), minlength=size)
        for level, size in enumerate(codebook_sizes)
    ]
    split_counts = {}
    for split in ("train", "valid", "test"):
        codes = split_target_codes(args.samples_root, split, item_to_codes)
        split_counts[split] = [
            np.bincount(codes[:, level].astype(np.int64), minlength=size)
            for level, size in enumerate(codebook_sizes)
        ]

    result = {
        "checkpoint": str(args.sid_checkpoint),
        "valid_users": users,
        "levels": [],
    }
    for level, size in enumerate(codebook_sizes):
        result["levels"].append(
            {
                "level": level + 1,
                "codebook_size": size,
                "all_items": distribution_metrics(item_counts[level]),
                "candidates": distribution_metrics(candidate_counts[level]),
                "train_targets": distribution_metrics(split_counts["train"][level]),
                "valid_targets": distribution_metrics(split_counts["valid"][level]),
                "test_targets": distribution_metrics(split_counts["test"][level]),
                "teacher_forced_targets": distribution_metrics(target_counts[level]),
                "teacher_forced_predictions": distribution_metrics(
                    prediction_counts[level]
                ),
                "prediction_target_kl": kl_divergence(
                    target_counts[level],
                    prediction_counts[level],
                ),
                "teacher_forced_accuracy": float(
                    correct[level] / users if users else 0.0
                ),
                "teacher_forced_prefix_accuracy": float(
                    prefix_correct[level] / users if users else 0.0
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
