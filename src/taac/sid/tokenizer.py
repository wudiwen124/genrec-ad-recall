from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from taac.data.candidate_map import load_candidate_mapping
from taac.data.feature_store import load_indexer
from taac.evaluation.group_analysis import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--fit-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def encode_all_items(
    model: torch.nn.Module,
    item_count: int,
    device: torch.device,
    batch_size: int = 131_072,
) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for start in range(1, item_count + 1, batch_size):
            item_ids = torch.arange(
                start,
                min(start + batch_size, item_count + 1),
                device=device,
                dtype=torch.long,
            )
            vectors.append(model.encode_items(item_ids).float().cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def combine_codes(codes: np.ndarray, base: int) -> np.ndarray:
    combined = np.zeros(len(codes), dtype=np.int64)
    for level in range(codes.shape[1]):
        combined = combined * base + codes[:, level]
    return combined


def build_collision_suffix(codes: np.ndarray, base: int) -> tuple[np.ndarray, dict]:
    keys = combine_codes(codes, base)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    group_starts = np.empty(len(keys), dtype=bool)
    group_starts[0] = True
    group_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_ids = np.cumsum(group_starts) - 1
    starts = np.flatnonzero(group_starts)
    ranks = np.arange(len(keys)) - starts[group_ids]
    suffix = np.empty(len(keys), dtype=np.int64)
    suffix[order] = ranks
    unique_before = len(starts)
    return suffix, {
        "items": int(len(keys)),
        "unique_codes_before_suffix": int(unique_before),
        "collision_rate_before_suffix": float(1.0 - unique_before / len(keys)),
        "max_suffix": int(suffix.max()) if len(suffix) else 0,
        "suffix_vocab_size": int(suffix.max() + 1) if len(suffix) else 1,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    model_args = Namespace(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        feature_cache_dir=args.feature_cache_dir,
    )
    model, _ = load_model(model_args, device)
    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())

    print(f"Encoding {item_count} items...", flush=True)
    embeddings = encode_all_items(model, item_count, device)
    residuals = embeddings.copy()
    codes = np.zeros((item_count, args.levels), dtype=np.int32)
    codebooks = []
    rng = np.random.default_rng(args.seed)
    fit_size = min(args.fit_samples, item_count)

    for level in range(args.levels):
        sample_indices = rng.choice(item_count, size=fit_size, replace=False)
        print(
            f"Fitting SID level {level + 1}/{args.levels} on {fit_size} samples...",
            flush=True,
        )
        kmeans = MiniBatchKMeans(
            n_clusters=args.clusters,
            batch_size=8192,
            n_init=3,
            max_iter=100,
            random_state=args.seed + level,
            verbose=0,
        )
        kmeans.fit(residuals[sample_indices])
        centers = kmeans.cluster_centers_.astype(np.float32)
        codebooks.append(centers)
        for start in range(0, item_count, 131_072):
            end = min(start + 131_072, item_count)
            labels = kmeans.predict(residuals[start:end])
            codes[start:end, level] = labels
            residuals[start:end] -= centers[labels]
        utilization = len(np.unique(codes[:, level])) / args.clusters
        print(f"Level {level + 1} utilization={utilization:.4f}", flush=True)

    suffix, collision_stats = build_collision_suffix(codes, args.clusters)
    all_codes = np.concatenate([codes, suffix[:, None].astype(np.int32)], axis=1)
    item_to_codes = np.zeros((item_count + 1, all_codes.shape[1]), dtype=np.int32)
    item_to_codes[1:] = all_codes
    reconstruction_mse = float(np.mean(residuals.astype(np.float64) ** 2))

    mapping = load_candidate_mapping(args.data_root)
    candidate_rids = mapping["mapped_rids"].astype(np.int64, copy=False)
    candidate_codes = item_to_codes[candidate_rids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        item_to_codes=item_to_codes,
        candidate_rids=candidate_rids,
        candidate_codes=candidate_codes,
        **{f"codebook_{index}": codebook for index, codebook in enumerate(codebooks)},
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "item_count": item_count,
        "levels": args.levels,
        "clusters_per_level": args.clusters,
        "codebook_sizes": [
            *[args.clusters] * args.levels,
            collision_stats["suffix_vocab_size"],
        ],
        "reconstruction_mse": reconstruction_mse,
        "candidate_items": int(len(candidate_rids)),
        **collision_stats,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
