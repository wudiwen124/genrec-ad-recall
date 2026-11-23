from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from taac.data.candidate_map import (
    build_rid_to_candidate_index_array,
    load_candidate_mapping,
)
from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.data.feature_store import (
    build_item_feature_matrix,
    build_user_feature_store,
    item_feature_vocab_sizes,
    load_indexer,
    user_feature_vocab_sizes,
    USER_ARRAY_MAX_LENGTHS,
    USER_ARRAY_FEATURE_IDS,
)
from taac.evaluation.retrieval import evaluate_full_candidate
from taac.models.sasrec import SASRecRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--filter-history", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    user_count = len(indexer["u"])

    item_feature_matrix, _ = build_item_feature_matrix(
        args.data_root,
        args.feature_cache_dir or (args.checkpoint.parent / "feature_cache"),
    )
    user_scalar_matrix, user_array_matrices, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir or (args.checkpoint.parent / "feature_cache"),
    )
    user_scalar_vocab_sizes, user_array_vocab_sizes = user_feature_vocab_sizes(indexer)
    model = SASRecRetriever(
        user_count=user_count,
        item_count=item_count,
        maxlen=train_args["maxlen"],
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        dropout=train_args["dropout"],
        item_feature_vocab_sizes=item_feature_vocab_sizes(indexer),
        user_scalar_vocab_sizes=user_scalar_vocab_sizes,
        user_array_vocab_sizes=user_array_vocab_sizes,
        user_array_max_lengths=[
            USER_ARRAY_MAX_LENGTHS[feature_id] for feature_id in USER_ARRAY_FEATURE_IDS
        ],
        model_type=train_args.get("model_type", "sasrec"),
        use_action=train_args.get("use_action", True),
        use_time=train_args.get("use_time", True),
        use_relative_bias=train_args.get("use_relative_bias", True),
        norm_first=train_args.get("norm_first", True),
    ).to(device)
    model.item_features.copy_(torch.from_numpy(item_feature_matrix))
    model.user_scalar_features.copy_(torch.from_numpy(user_scalar_matrix))
    for index, feature_id in enumerate(USER_ARRAY_FEATURE_IDS):
        getattr(model, f"user_array_features_{index}").copy_(
            torch.from_numpy(user_array_matrices[feature_id])
        )
    model.load_state_dict(checkpoint["model"])

    dataset = ClickSampleDataset(
        args.samples_root / args.split / "part-00000.parquet",
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=train_args["maxlen"]),
    )
    mapping = load_candidate_mapping(args.data_root)
    result = evaluate_full_candidate(
        model=model,
        loader=loader,
        candidate_rids=mapping["retrieval_to_rid"],
        rid_to_candidate_index=build_rid_to_candidate_index_array(
            mapping["retrieval_to_rid"],
            item_count,
        ),
        device=device,
        max_batches=args.max_batches,
        filter_history=args.filter_history,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "max_samples": args.max_samples,
        "max_batches": args.max_batches,
        "filter_history": args.filter_history,
        "candidate_count": int(len(mapping["retrieval_to_rid"])),
        "mapped_candidate_count": int(len(mapping["mapped_rids"])),
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
