from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_ARRAY_MAX_LENGTHS,
    build_user_feature_store,
    load_indexer,
    user_feature_vocab_sizes,
)
from taac.models.sid_generator import SIDGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-valid-samples", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_sid(
    model: SIDGenerator,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    correct = np.zeros(model.num_sid_levels, dtype=np.int64)
    exact = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            state = model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["valid_mask"],
            )
            target_codes = model.item_to_codes[moved["targets"]].long()
            logits = model.predict_logits(state, target_codes)
            predictions = torch.stack([item.argmax(dim=-1) for item in logits], dim=1)
            matches = predictions == target_codes
            correct += matches.sum(dim=0).cpu().numpy()
            exact += matches.all(dim=1).sum().item()
            total += len(target_codes)
    return {
        "users": total,
        "level_accuracy": [float(value / total) for value in correct],
        "exact_sid_accuracy": float(exact / total) if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    indexer = load_indexer(args.data_root)
    user_count = len(indexer["u"])
    item_count = max(int(value) for value in indexer["i"].values())
    sid = np.load(args.sid_path)
    metadata = json.loads(args.sid_path.with_suffix(".json").read_text(encoding="utf-8"))
    codebook_sizes = metadata["codebook_sizes"]

    user_scalars, user_arrays, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir,
    )
    scalar_vocab, array_vocab = user_feature_vocab_sizes(indexer)
    model = SIDGenerator(
        user_count=user_count,
        item_count=item_count,
        codebook_sizes=codebook_sizes,
        maxlen=101,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        user_scalar_vocab_sizes=scalar_vocab,
        user_array_vocab_sizes=array_vocab,
        user_array_max_lengths=[
            USER_ARRAY_MAX_LENGTHS[feature_id] for feature_id in USER_ARRAY_FEATURE_IDS
        ],
    ).to(device)
    model.item_to_codes.copy_(torch.from_numpy(sid["item_to_codes"]))
    model.user_scalar_features.copy_(torch.from_numpy(user_scalars))
    for index, feature_id in enumerate(USER_ARRAY_FEATURE_IDS):
        getattr(model, f"user_array_features_{index}").copy_(
            torch.from_numpy(user_arrays[feature_id])
        )

    train_dataset = ClickSampleDataset(args.samples_root / "train" / "part-00000.parquet")
    valid_dataset = ClickSampleDataset(
        args.samples_root / "valid" / "part-00000.parquet",
        max_samples=args.max_valid_samples,
    )
    collate = partial(collate_click_samples, maxlen=101)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    sparse_parameters = list(model.user_emb.parameters())
    sparse_ids = {id(parameter) for parameter in sparse_parameters}
    dense_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in sparse_ids
    ]
    sparse_optimizer = torch.optim.SparseAdam(sparse_parameters, lr=args.lr)
    optimizer = torch.optim.AdamW(dense_parameters, lr=args.lr, weight_decay=1e-5)
    best_accuracy = -1.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for step, batch in enumerate(train_loader, start=1):
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            target_codes = model.item_to_codes[moved["targets"]].long()
            sparse_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = model.encode(
                    moved["sequence"],
                    moved["actions"],
                    moved["time_buckets"],
                    moved["user_ids"],
                    moved["valid_mask"],
                )
                logits = model.predict_logits(state, target_codes)
                loss = sum(
                    F.cross_entropy(item.float(), target_codes[:, level])
                    for level, item in enumerate(logits)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
            sparse_optimizer.step()
            optimizer.step()
            total_loss += float(loss.item()) * len(target_codes)
            total_rows += len(target_codes)
            if step % 100 == 0:
                print(
                    f"epoch={epoch} step={step} loss={total_loss / total_rows:.6f}",
                    flush=True,
                )
        validation = evaluate_sid(model, valid_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        if validation["exact_sid_accuracy"] > best_accuracy:
            best_accuracy = validation["exact_sid_accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "codebook_sizes": codebook_sizes,
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output_dir / "best.pt",
            )
    (args.output_dir / "train_result.json").write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "history": history,
                "best_exact_sid_accuracy": best_accuracy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
