from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--negatives", type=int, default=256)
    parser.add_argument("--model-type", choices=["sasrec", "hstu"], default="sasrec")
    parser.add_argument("--loss-type", choices=["info_nce", "bce"], default="info_nce")
    parser.add_argument("--negative-mode", choices=["uniform", "inbatch", "mixed"], default="uniform")
    parser.add_argument("--use-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-relative-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logq", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--hard-negative",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--hard-negative-multiplier", type=int, default=2)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment identifier written to the result JSON.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_negative_logq(
    train_dataset: ClickSampleDataset,
    *,
    item_count: int,
    negative_pool_size: int,
    negative_mode: str,
) -> torch.Tensor:
    target_ids = train_dataset.table["target_item_id"].to_numpy()
    counts = np.bincount(target_ids, minlength=item_count + 1).astype(np.float64)
    target_distribution = counts / max(1.0, counts.sum())
    uniform_probability = 1.0 / max(1, negative_pool_size)
    if negative_mode == "inbatch":
        probability = target_distribution
    elif negative_mode == "mixed":
        probability = 0.5 * target_distribution + 0.5 * uniform_probability
    else:
        probability = np.full(item_count + 1, uniform_probability, dtype=np.float64)
    probability = np.maximum(probability, 1e-12)
    return torch.from_numpy(np.log(probability).astype(np.float32))


def sample_training_logits(
    *,
    model: torch.nn.Module,
    query: torch.Tensor,
    target_items: torch.Tensor,
    negative_pool: torch.Tensor,
    negatives: int,
    negative_mode: str,
    logq: torch.Tensor | None,
    hard_negative: bool,
    hard_topk: int,
    hard_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(query, dim=-1)
    positive_vectors = F.normalize(model.encode_items(target_items), dim=-1)
    positive_logits = (query * positive_vectors).sum(dim=-1, keepdim=True)

    sample_count = negatives * hard_multiplier if hard_negative else negatives
    uniform_ids = negative_pool[
        torch.randint(
            0,
            len(negative_pool),
            (target_items.shape[0], sample_count),
            device=target_items.device,
        )
    ]
    same_as_target = uniform_ids == target_items.unsqueeze(1)
    if bool(same_as_target.any()):
        uniform_ids = uniform_ids.clone()
        uniform_ids[same_as_target] = negative_pool[
            torch.randint(
                0,
                len(negative_pool),
                (int(same_as_target.sum()),),
                device=target_items.device,
            )
        ]

    if negative_mode == "uniform":
        negative_ids = uniform_ids
        negative_logits = torch.bmm(
            F.normalize(model.encode_items(negative_ids), dim=-1),
            query.unsqueeze(-1),
        ).squeeze(-1)
    elif negative_mode == "inbatch":
        negative_ids = target_items.unsqueeze(0).expand(target_items.shape[0], -1)
        negative_vectors = F.normalize(model.encode_items(target_items), dim=-1)
        negative_logits = query @ negative_vectors.T
        diagonal = torch.eye(len(target_items), dtype=torch.bool, device=target_items.device)
        negative_logits = negative_logits.masked_fill(diagonal, -torch.inf)
    else:
        inbatch_ids = target_items.unsqueeze(0).expand(target_items.shape[0], -1)
        inbatch_vectors = F.normalize(model.encode_items(target_items), dim=-1)
        inbatch_logits = query @ inbatch_vectors.T
        diagonal = torch.eye(len(target_items), dtype=torch.bool, device=target_items.device)
        inbatch_logits = inbatch_logits.masked_fill(diagonal, -torch.inf)
        uniform_logits = torch.bmm(
            F.normalize(model.encode_items(uniform_ids), dim=-1),
            query.unsqueeze(-1),
        ).squeeze(-1)
        negative_ids = torch.cat([uniform_ids, inbatch_ids], dim=1)
        negative_logits = torch.cat([uniform_logits, inbatch_logits], dim=1)

    if hard_negative:
        top_values, top_indices = negative_logits.topk(
            min(hard_topk, negative_logits.shape[1]),
            dim=1,
        )
        negative_ids = negative_ids.gather(1, top_indices)
        negative_logits = top_values

    if logq is not None:
        positive_logits = positive_logits - logq[target_items].unsqueeze(1)
        negative_logits = negative_logits - logq[negative_ids]

    return positive_logits, negative_logits


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the first-version training run")
    feature_cache_dir = args.feature_cache_dir or (args.output_dir / "feature_cache")

    mapping = load_candidate_mapping(args.data_root)
    candidate_rids = mapping["retrieval_to_rid"]
    negative_pool_cpu = torch.from_numpy(mapping["mapped_rids"])

    train_dataset = ClickSampleDataset(
        args.samples_root / "train" / "part-00000.parquet",
        max_samples=args.max_train_samples,
    )
    valid_dataset = ClickSampleDataset(
        args.samples_root / "valid" / "part-00000.parquet",
        max_samples=args.max_valid_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=args.maxlen),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=args.maxlen),
    )

    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    user_count = len(indexer["u"])
    rid_to_index = build_rid_to_candidate_index_array(candidate_rids, item_count)
    item_feature_matrix, item_feature_metadata = build_item_feature_matrix(
        args.data_root,
        feature_cache_dir,
    )
    user_scalar_matrix, user_array_matrices, user_feature_metadata = build_user_feature_store(
        args.data_root,
        feature_cache_dir,
    )
    user_scalar_vocab_sizes, user_array_vocab_sizes = user_feature_vocab_sizes(indexer)

    model = SASRecRetriever(
        user_count=user_count,
        item_count=item_count,
        maxlen=args.maxlen,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        item_feature_vocab_sizes=item_feature_vocab_sizes(indexer),
        user_scalar_vocab_sizes=user_scalar_vocab_sizes,
        user_array_vocab_sizes=user_array_vocab_sizes,
        user_array_max_lengths=[
            USER_ARRAY_MAX_LENGTHS[feature_id] for feature_id in USER_ARRAY_FEATURE_IDS
        ],
        model_type=args.model_type,
        use_action=args.use_action,
        use_time=args.use_time,
        use_relative_bias=args.use_relative_bias,
        norm_first=args.norm_first,
    ).to(device)
    model.item_features.copy_(torch.from_numpy(item_feature_matrix))
    model.user_scalar_features.copy_(torch.from_numpy(user_scalar_matrix))
    for index, feature_id in enumerate(USER_ARRAY_FEATURE_IDS):
        getattr(model, f"user_array_features_{index}").copy_(
            torch.from_numpy(user_array_matrices[feature_id])
        )

    sparse_parameters = list(model.item_emb.parameters()) + list(model.user_emb.parameters())
    sparse_ids = {id(parameter) for parameter in sparse_parameters}
    dense_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in sparse_ids
    ]
    sparse_optimizer = torch.optim.SparseAdam(sparse_parameters, lr=args.lr)
    dense_optimizer = torch.optim.AdamW(dense_parameters, lr=args.lr, weight_decay=1e-5)

    negative_pool = negative_pool_cpu.to(device)
    logq = (
        build_negative_logq(
            train_dataset,
            item_count=item_count,
            negative_pool_size=len(negative_pool),
            negative_mode=args.negative_mode,
        ).to(device)
        if args.logq
        else None
    )
    history: list[dict[str, float | int]] = []
    started = time.time()
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        epoch_started = time.time()
        hard_enabled = args.hard_negative and (not args.curriculum or epoch >= 2)
        if args.curriculum and args.epochs > 1:
            curriculum_progress = (epoch - 1) / (args.epochs - 1)
            hard_topk = max(16, int(16 + (args.negatives - 16) * curriculum_progress))
        else:
            hard_topk = args.negatives
        for step, batch in enumerate(train_loader, start=1):
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            target_items = moved["targets"]

            dense_optimizer.zero_grad(set_to_none=True)
            sparse_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                query = model.encode(
                    moved["sequence"],
                    moved["actions"],
                    moved["time_buckets"],
                    moved["user_ids"],
                    moved["timestamps"],
                    moved["valid_mask"],
                )
                positive_logits, negative_logits = sample_training_logits(
                    model=model,
                    query=query,
                    target_items=target_items,
                    negative_pool=negative_pool,
                    negatives=args.negatives,
                    negative_mode=args.negative_mode,
                    logq=logq,
                    hard_negative=hard_enabled,
                    hard_topk=hard_topk,
                    hard_multiplier=args.hard_negative_multiplier,
                )
                if args.loss_type == "bce":
                    positive_loss = F.binary_cross_entropy_with_logits(
                        positive_logits.float(),
                        torch.ones_like(positive_logits, dtype=torch.float32),
                    )
                    negative_loss = F.binary_cross_entropy_with_logits(
                        negative_logits.float(),
                        torch.zeros_like(negative_logits, dtype=torch.float32),
                    )
                    loss = positive_loss + negative_loss
                else:
                    logits = torch.cat([positive_logits, negative_logits], dim=1).float()
                    labels = torch.zeros(len(logits), dtype=torch.long, device=device)
                    loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
            sparse_optimizer.step()
            dense_optimizer.step()

            total_loss += float(loss.item()) * len(target_items)
            total_rows += len(target_items)
            if step % 100 == 0:
                print(
                    f"epoch={epoch} step={step} loss={total_loss / total_rows:.6f} "
                    f"elapsed={time.time() - epoch_started:.1f}s",
                    flush=True,
                )

        validation = evaluate_full_candidate(
            model=model,
            loader=valid_loader,
            candidate_rids=candidate_rids,
            rid_to_candidate_index=rid_to_index,
            device=device,
            max_batches=args.max_valid_batches,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows if total_rows else 0.0,
            "train_rows": total_rows,
            "train_seconds": round(time.time() - epoch_started, 3),
            "validation": validation,
        }
        history.append(epoch_result)
        if validation["score"] > best_score:
            best_score = validation["score"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            },
            args.output_dir / "latest.pt",
        )
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping after epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break

    result = {
        "args": vars(args),
        "experiment_id": args.experiment_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "train_users": len(train_dataset),
        "valid_users": len(valid_dataset),
        "candidate_count": int(len(candidate_rids)),
        "mapped_candidate_count": int(len(mapping["mapped_rids"])),
        "item_feature_metadata": item_feature_metadata,
        "user_feature_metadata": user_feature_metadata,
        "epochs": history,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "total_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "train_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
