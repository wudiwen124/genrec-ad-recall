from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from taac.data.click_dataset import time_gap_buckets
from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_SCALAR_FEATURE_IDS,
    build_user_feature_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("warmup", "lora"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--freeze-heads", action="store_true")
    parser.add_argument("--sampling-alpha", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QwenSidDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        item_to_codes: np.ndarray,
        token_ids_by_level: list[list[int]],
        max_samples: int,
        max_history: int,
        max_length: int,
        eos_token_id: int,
        context_token_ids: dict,
        user_scalars: np.ndarray,
        user_arrays: dict[str, np.ndarray],
        sampling_alpha: float,
        seed: int,
    ):
        self.table = pq.read_table(
            parquet_path,
            columns=[
                "user_id",
                "history_item_ids",
                "history_action_types",
                "history_timestamps",
                "target_item_id",
            ],
        )
        if max_samples and max_samples < len(self.table):
            if sampling_alpha > 0:
                target_ids = self.table["target_item_id"].to_numpy()
                counts = np.bincount(target_ids.astype(np.int64))
                weights = (
                    1.0 / np.maximum(counts[target_ids.astype(np.int64)], 1)
                ) ** sampling_alpha
                probabilities = weights / weights.sum()
                self.indices = np.random.default_rng(seed).choice(
                    len(self.table),
                    size=max_samples,
                    replace=True,
                    p=probabilities,
                )
            else:
                self.indices = np.linspace(
                    0,
                    len(self.table) - 1,
                    num=max_samples,
                    dtype=np.int64,
                )
        else:
            self.indices = None
        self.item_to_codes = item_to_codes
        self.token_ids_by_level = [
            np.asarray(ids, dtype=np.int64) for ids in token_ids_by_level
        ]
        self.max_history = max_history
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.action_tokens = np.asarray(
            context_token_ids["action"],
            dtype=np.int64,
        )
        self.time_tokens = np.asarray(
            context_token_ids["time"],
            dtype=np.int64,
        )
        self.scalar_tokens = {
            feature_id: np.asarray(ids, dtype=np.int64)
            for feature_id, ids in context_token_ids["scalar"].items()
        }
        self.array_tokens = {
            feature_id: np.asarray(ids, dtype=np.int64)
            for feature_id, ids in context_token_ids["array"].items()
        }
        self.user_scalars = user_scalars
        self.user_arrays = user_arrays

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.table)

    def encode_item(self, item_id: int) -> list[int]:
        codes = self.item_to_codes[item_id]
        return [
            int(self.token_ids_by_level[level][int(code)])
            for level, code in enumerate(codes)
        ]

    def encode_user_profile(self, user_id: int) -> list[int]:
        tokens = []
        scalar_values = self.user_scalars[user_id]
        for feature_index, feature_id in enumerate(USER_SCALAR_FEATURE_IDS):
            value = int(scalar_values[feature_index])
            if value > 0:
                tokens.append(int(self.scalar_tokens[feature_id][value]))
        for feature_id in USER_ARRAY_FEATURE_IDS:
            for value in self.user_arrays[feature_id][user_id].tolist():
                value = int(value)
                if value > 0:
                    tokens.append(int(self.array_tokens[feature_id][value]))
        return tokens

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        source_index = int(self.indices[index]) if self.indices is not None else index
        user_id = int(self.table["user_id"][source_index].as_py())
        history = [
            int(item)
            for item in self.table["history_item_ids"][source_index].as_py()
            if int(item) > 0
        ][-self.max_history :]
        actions = [
            int(value)
            for value in self.table["history_action_types"][source_index].as_py()
        ][-self.max_history :]
        timestamps = [
            int(value)
            for value in self.table["history_timestamps"][source_index].as_py()
        ][-self.max_history :]
        target = int(self.table["target_item_id"][source_index].as_py())

        history_tokens = []
        gaps = time_gap_buckets(timestamps)
        for item, action, gap in zip(history, actions, gaps):
            history_tokens.append(int(self.action_tokens[min(action, 4)]))
            history_tokens.append(int(self.time_tokens[min(gap, 63)]))
            history_tokens.extend(self.encode_item(item))
        profile_tokens = self.encode_user_profile(user_id)
        target_tokens = self.encode_item(target) + [self.eos_token_id]
        prefix_tokens = profile_tokens + history_tokens
        available_prefix = max(0, self.max_length - len(target_tokens))
        if len(prefix_tokens) > available_prefix:
            prefix_tokens = prefix_tokens[-available_prefix:]
        input_ids = prefix_tokens + target_tokens
        labels = [-100] * len(prefix_tokens) + target_tokens
        return {"input_ids": input_ids, "labels": labels}


def collate_rows(rows: list[dict[str, list[int]]], pad_token_id: int) -> dict:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), max_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"])
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(row["labels"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation: int,
    log_every: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_batches = 0
    for step, batch in enumerate(loader, start=1):
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**moved).loss
        (loss / gradient_accumulation).backward()
        if step % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
        total_batches += 1
        if step % log_every == 0:
            print(
                f"step={step}/{len(loader)} "
                f"loss={total_loss / total_batches:.6f} "
                f"gpu={torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                flush=True,
            )
    if len(loader) % gradient_accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    metadata = json.loads(
        (args.tokenizer_path / "sid_tokens.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.resize_token_embeddings(metadata["total_vocab_size"])
    model.config.use_cache = False

    sid = np.load(args.sid_path)
    user_scalars, user_arrays, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir,
    )
    dataset = QwenSidDataset(
        parquet_path=args.samples_root / "train" / "part-00000.parquet",
        item_to_codes=sid["item_to_codes"],
        token_ids_by_level=metadata["token_ids_by_level"],
        max_samples=args.max_samples,
        max_history=args.max_history,
        max_length=args.max_length,
        eos_token_id=metadata["eos_token_id"],
        context_token_ids=metadata["context_token_ids"],
        user_scalars=user_scalars,
        user_arrays=user_arrays,
        sampling_alpha=args.sampling_alpha,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_rows, pad_token_id=metadata["pad_token_id"]),
    )

    if args.mode == "warmup":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        learning_rate = args.learning_rate or 5e-4
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.0,
        )
    else:
        if args.warmup_checkpoint is None:
            raise ValueError("--warmup-checkpoint is required for LoRA")
        head_state = torch.load(
            args.warmup_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.get_input_embeddings().weight.data.copy_(
            head_state["embed_tokens"].to(device)
        )
        model.get_output_embeddings().weight.data.copy_(
            head_state["lm_head"].to(device)
        )
        from peft import LoraConfig, TaskType, get_peft_model

        model.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=[] if args.freeze_heads else ["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, lora_config)
        if args.freeze_heads:
            model.get_input_embeddings().weight.requires_grad_(False)
            model.get_output_embeddings().weight.requires_grad_(False)
        learning_rate = args.learning_rate or 2e-4
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )

    trainable, total = trainable_parameter_summary(model)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "samples": len(dataset),
                "steps": len(loader),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "learning_rate": learning_rate,
            },
            indent=2,
        ),
        flush=True,
    )

    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            args.gradient_accumulation,
            args.log_every,
        )
        losses.append(loss)
        print(f"epoch={epoch} mean_loss={loss:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    if args.mode == "warmup":
        torch.save(
            {
                "embed_tokens": model.get_input_embeddings().weight.detach().cpu(),
                "lm_head": model.get_output_embeddings().weight.detach().cpu(),
                "losses": losses,
                "metadata": metadata,
            },
            args.output_dir / "warmup_heads.pt",
        )
    else:
        model.save_pretrained(args.output_dir)
    (args.output_dir / "train_result.json").write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "losses": losses,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
