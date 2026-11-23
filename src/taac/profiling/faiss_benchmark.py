from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from taac.data.candidate_map import (
    build_rid_to_candidate_index_array,
    load_candidate_mapping,
)
from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.evaluation.group_analysis import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--index-type", choices=["flat", "hnsw"], default="hnsw")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(array, 0.50) * 1000),
        "p95_ms": float(np.quantile(array, 0.95) * 1000),
        "p99_ms": float(np.quantile(array, 0.99) * 1000),
        "mean_ms": float(array.mean() * 1000),
        "qps": float(1.0 / array.mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    model, train_args = load_model(args, device)
    mapping = load_candidate_mapping(args.data_root)
    from taac.data.feature_store import load_indexer

    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    rid_to_candidate = build_rid_to_candidate_index_array(
        mapping["retrieval_to_rid"],
        item_count,
    )

    torch.cuda.reset_peak_memory_stats()
    candidate_rids = torch.from_numpy(mapping["retrieval_to_rid"]).to(device)
    embed_started = time.perf_counter()
    with torch.no_grad():
        candidate_vectors = torch.nn.functional.normalize(
            model.encode_items(candidate_rids),
            dim=-1,
        )
        candidate_vectors_cpu = candidate_vectors.cpu().numpy().astype(np.float32)
    embedding_seconds = time.perf_counter() - embed_started
    peak_gpu_bytes = torch.cuda.max_memory_allocated()

    dimension = candidate_vectors_cpu.shape[1]
    if args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatIP(dimension)
    index_started = time.perf_counter()
    index.add(candidate_vectors_cpu)
    index_seconds = time.perf_counter() - index_started

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_path))
    index_size = args.index_path.stat().st_size

    dataset = ClickSampleDataset(
        args.samples_root / args.split / "part-00000.parquet",
        max_samples=args.queries,
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=train_args["maxlen"]),
    )

    queries = []
    targets = []
    history_indices = []
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
            query = torch.nn.functional.normalize(query, dim=-1).cpu().numpy().astype(np.float32)
            batch_history = rid_to_candidate[moved["sequence"].cpu().numpy()]
            for row_index in range(len(query)):
                queries.append(query[row_index])
                targets.append(int(rid_to_candidate[int(moved["targets"][row_index])]))
                history_indices.append(set(int(value) for value in batch_history[row_index] if value >= 0))

    queries_array = np.stack(queries)
    targets_array = np.asarray(targets)
    results: dict[str, dict[str, float]] = {}
    for batch_size in [1, 16, 64]:
        for _ in range(10):
            index.search(queries_array[:batch_size], args.top_k)
        latencies = []
        hit10 = 0
        valid = 0
        started = time.perf_counter()
        for start in range(0, len(queries_array), batch_size):
            query_batch = queries_array[start : start + batch_size]
            batch_started = time.perf_counter()
            distances, indices = index.search(query_batch, args.top_k)
            latencies.append(time.perf_counter() - batch_started)
            for offset, row_indices in enumerate(indices):
                global_index = start + offset
                filtered = [
                    int(candidate_index)
                    for candidate_index in row_indices
                    if candidate_index >= 0
                    and int(candidate_index) not in history_indices[global_index]
                ][:10]
                if int(targets_array[global_index]) in filtered:
                    hit10 += 1
                valid += 1
        elapsed = time.perf_counter() - started
        summary = latency_summary(latencies)
        summary["queries"] = valid
        summary["batch_qps"] = valid / elapsed
        summary["hr@10"] = hit10 / valid if valid else 0.0
        results[f"batch_{batch_size}"] = summary

    recall100 = float(
        np.mean(
            [
                targets_array[row] in set(int(value) for value in indices if value >= 0)
                for row, indices in enumerate(index.search(queries_array, args.top_k)[1])
            ]
        )
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "queries": len(queries_array),
        "candidate_count": len(candidate_vectors_cpu),
        "dimension": dimension,
        "index_type": args.index_type,
        "top_k": args.top_k,
        "embedding_seconds": embedding_seconds,
        "index_build_seconds": index_seconds,
        "index_size_bytes": index_size,
        "peak_gpu_bytes": peak_gpu_bytes,
        "recall@100": recall100,
        "search": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
