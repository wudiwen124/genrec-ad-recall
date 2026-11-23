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

from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.sid.evaluate_sid import build_candidate_maps, load_sid_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--sid-checkpoint", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=256)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--beam-size", type=int, required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def decode_batch(
    model: torch.nn.Module,
    moved: dict[str, torch.Tensor],
    children: list[dict],
    code_to_rid: dict[tuple[int, ...], int],
    beam_size: int,
    candidate_k: int,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = model.encode(
        moved["sequence"],
        moved["actions"],
        moved["time_buckets"],
        moved["user_ids"],
        moved["valid_mask"],
    )
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started

    beam_width = 1
    beam_hidden = state.unsqueeze(1)
    beam_scores = torch.zeros((state.shape[0], 1), device=state.device)
    beam_histories: list[list[tuple[int, ...]]] = [[()] for _ in range(state.shape[0])]

    for level in range(model.num_sid_levels):
        flat_hidden = beam_hidden.reshape(-1, model.hidden_size)
        log_probs = F.log_softmax(
            model.output_heads[level](
                flat_hidden + model.level_position.weight[level]
            ).float(),
            dim=-1,
        )
        candidate_options: list[list[tuple[float, int, int]]] = [
            [] for _ in range(state.shape[0])
        ]
        for owner in range(state.shape[0]):
            for beam_index, prefix in enumerate(beam_histories[owner]):
                allowed = children[level].get(prefix, [])
                if not allowed:
                    continue
                flat_index = owner * beam_width + beam_index
                allowed_tensor = torch.tensor(
                    allowed,
                    device=state.device,
                    dtype=torch.long,
                )
                values = (
                    log_probs[flat_index, allowed_tensor]
                    + beam_scores[owner, beam_index]
                )
                for value, code in zip(values.tolist(), allowed):
                    candidate_options[owner].append(
                        (float(value), int(code), beam_index)
                    )

        next_histories: list[list[tuple[int, ...]]] = [
            [] for _ in range(state.shape[0])
        ]
        source_indices = []
        selected_codes = []
        selected_scores = []
        next_width = min(
            beam_size,
            max(1, min(len(options) for options in candidate_options)),
        )
        for owner, options in enumerate(candidate_options):
            options.sort(reverse=True, key=lambda item: item[0])
            options = options[:next_width]
            owner_sources = []
            owner_codes = []
            owner_scores = []
            for value, code, beam_index in options:
                next_histories[owner].append(
                    beam_histories[owner][beam_index] + (code,)
                )
                owner_sources.append(owner * beam_width + beam_index)
                owner_codes.append(code)
                owner_scores.append(value)
            source_indices.append(owner_sources)
            selected_codes.append(owner_codes)
            selected_scores.append(owner_scores)

        source_tensor = torch.tensor(
            source_indices,
            device=state.device,
            dtype=torch.long,
        )
        code_tensor = torch.tensor(
            selected_codes,
            device=state.device,
            dtype=torch.long,
        )
        selected_hidden = flat_hidden[source_tensor.reshape(-1)].reshape(
            state.shape[0],
            next_width,
            model.hidden_size,
        )
        if level < model.num_sid_levels - 1:
            code_embedding = model.sid_embeddings[level](code_tensor)
            selected_hidden = selected_hidden + model.level_transforms[level](
                torch.cat([selected_hidden, code_embedding], dim=-1)
            )
        beam_hidden = selected_hidden
        beam_scores = torch.tensor(selected_scores, device=state.device)
        beam_histories = next_histories
        beam_width = next_width

    # Materialize a fixed candidate count to include Python SID mapping cost.
    for histories in beam_histories:
        mapped = []
        seen = set()
        for prefix in histories:
            rid = code_to_rid.get(prefix)
            if rid is None or rid in seen:
                continue
            mapped.append(rid)
            seen.add(rid)
            if len(mapped) == candidate_k:
                break
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    return encode_seconds, total_seconds


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    model = load_sid_model(args, device)
    _, _, code_to_rid, children = build_candidate_maps(args)
    dataset = ClickSampleDataset(
        args.samples_root / "test" / "part-00000.parquet",
        max_samples=args.users,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=101),
    )
    batches = [
        {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        for batch in loader
    ]
    if not batches:
        raise RuntimeError("No profiling batches were created")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        for batch in batches:
            decode_batch(
                model,
                batch,
                children,
                code_to_rid,
                args.beam_size,
                args.candidate_k,
            )

    batch_latencies = []
    encode_latencies = []
    measured_users = 0
    measured_batches = 0
    for _ in range(args.repeats):
        for batch in batches:
            encode_seconds, total_seconds = decode_batch(
                model,
                batch,
                children,
                code_to_rid,
                args.beam_size,
                args.candidate_k,
            )
            encode_latencies.append(encode_seconds * 1000)
            batch_latencies.append(total_seconds * 1000)
            measured_users += batch["user_ids"].shape[0]
            measured_batches += 1

    total_measured_seconds = sum(batch_latencies) / 1000
    result = {
        "users": measured_users,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "candidate_k": args.candidate_k,
        "repeats": args.repeats,
        "wavefront_batches": measured_batches,
        "encode": {
            "p50_ms": percentile(encode_latencies, 50),
            "p95_ms": percentile(encode_latencies, 95),
            "p99_ms": percentile(encode_latencies, 99),
        },
        "end_to_end_batch": {
            "p50_ms": percentile(batch_latencies, 50),
            "p95_ms": percentile(batch_latencies, 95),
            "p99_ms": percentile(batch_latencies, 99),
            "per_request_p50_ms": (
                percentile(batch_latencies, 50) / args.batch_size
            ),
            "per_request_p95_ms": (
                percentile(batch_latencies, 95) / args.batch_size
            ),
            "qps": measured_users / total_measured_seconds,
        },
        "peak_gpu_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_gpu_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
