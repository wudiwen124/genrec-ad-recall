from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


UNKNOWN_ACTION = 3


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def candidate_rids(root: Path, indexer: dict[str, Any]) -> set[int]:
    table = pq.read_table(next((root / "candidate").glob("*.parquet")), columns=["item_id"])
    oid_to_rid = indexer["i"]
    result = {
        int(oid_to_rid[item_oid])
        for item_oid in table.column("item_id").to_pylist()
        if item_oid in oid_to_rid
    }
    return result


def select_last_click_if_candidate(
    events: Iterable[dict[str, Any]],
    valid_targets: set[int],
) -> tuple[int, int, int] | None:
    """Return the user's last click only when that target is in the candidate pool."""
    last_click: tuple[int, int, int] | None = None
    for index, event in enumerate(events):
        action = event.get("action_type")
        item_id = event.get("item_id")
        timestamp = event.get("timestamp")
        if action == 1 and item_id is not None and timestamp is not None:
            last_click = (int(item_id), int(timestamp), index)
    if last_click is None or last_click[0] not in valid_targets:
        return None
    return last_click


def quantile_boundaries(timestamps: np.ndarray) -> tuple[int, int]:
    if len(timestamps) < 3:
        raise ValueError("At least 3 users are required to create train/valid/test splits")
    q70, q85 = np.quantile(timestamps, [0.70, 0.85], method="nearest")
    return int(q70), int(q85)


def assign_split(timestamp: int, q70: int, q85: int) -> str:
    if timestamp <= q70:
        return "train"
    if timestamp <= q85:
        return "valid"
    return "test"


def iter_target_candidates(
    root: Path,
    valid_targets: set[int],
    *,
    limit_users: int | None,
) -> Iterable[tuple[int, list[dict[str, Any]], tuple[int, int, int]]]:
    seen_users = 0
    for file in parquet_files(root / "seq"):
        print(f"[target scan] {file.name}", flush=True)
        for batch in pq.ParquetFile(file).iter_batches(batch_size=20_000, columns=["user_id", "seq"]):
            data = batch.to_pydict()
            for user_id, events in zip(data["user_id"], data["seq"]):
                if limit_users is not None and seen_users >= limit_users:
                    return
                seen_users += 1
                target = select_last_click_if_candidate(events, valid_targets)
                if target is not None:
                    yield int(user_id), events, target


def history_before_target(
    events: list[dict[str, Any]],
    target_timestamp: int,
    max_history: int,
) -> tuple[list[int], list[int], list[int]]:
    history = [
        event
        for event in events
        if event.get("timestamp") is not None and int(event["timestamp"]) < target_timestamp
    ][-max_history:]
    item_ids = [int(event["item_id"]) for event in history if event.get("item_id") is not None]
    action_types = [
        int(event["action_type"]) if event.get("action_type") is not None else UNKNOWN_ACTION
        for event in history
    ]
    timestamps = [int(event["timestamp"]) for event in history]
    if not (len(item_ids) == len(action_types) == len(timestamps)):
        raise ValueError("History arrays have inconsistent lengths")
    return item_ids, action_types, timestamps


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("user_id", pa.int64()),
            pa.field("target_item_id", pa.int64()),
            pa.field("target_timestamp", pa.int64()),
            pa.field("history_item_ids", pa.list_(pa.int64())),
            pa.field("history_action_types", pa.list_(pa.int8())),
            pa.field("history_timestamps", pa.list_(pa.int64())),
            pa.field("history_len", pa.int16()),
        ]
    )


def write_sample_batch(
    split: str,
    records: list[dict[str, Any]],
    writers: dict[str, pq.ParquetWriter],
) -> None:
    if not records:
        return
    table = pa.Table.from_pylist(records, schema=schema())
    writers[split].write_table(table)
    records.clear()


def build_samples(
    root: Path,
    output: Path,
    *,
    max_history: int = 100,
    limit_users: int | None = None,
) -> dict[str, Any]:
    with (root / "indexer.pkl").open("rb") as handle:
        indexer = pickle.load(handle)
    valid_targets = candidate_rids(root, indexer)

    # Pass 1: locate the last candidate-covered click for every user.
    targets: dict[int, tuple[int, int, int]] = {}
    for user_id, _, target in iter_target_candidates(
        root,
        valid_targets,
        limit_users=limit_users,
    ):
        targets[user_id] = target
    if not targets:
        raise RuntimeError("No candidate-covered click targets were found")

    timestamps = np.fromiter((target[1] for target in targets.values()), dtype=np.int64)
    q70, q85 = quantile_boundaries(timestamps)

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        split_path = output / split
        split_path.mkdir(parents=True, exist_ok=True)
        for stale in split_path.glob("*.parquet"):
            stale.unlink()

    writers = {
        split: pq.ParquetWriter(output / split / "part-00000.parquet", schema())
        for split in ("train", "valid", "test")
    }
    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    split_counts: Counter[str] = Counter()
    empty_history = 0
    null_action_events = 0

    try:
        # Pass 2: materialize only events strictly before the selected target.
        for user_id, events, target in iter_target_candidates(
            root,
            valid_targets,
            limit_users=limit_users,
        ):
            target_item, target_timestamp, _ = target
            history_items, history_actions, history_timestamps = history_before_target(
                events,
                target_timestamp,
                max_history=max_history,
            )
            if not history_items:
                empty_history += 1
                continue
            null_action_events += sum(action == UNKNOWN_ACTION for action in history_actions)
            split = assign_split(target_timestamp, q70, q85)
            records[split].append(
                {
                    "user_id": user_id,
                    "target_item_id": target_item,
                    "target_timestamp": target_timestamp,
                    "history_item_ids": history_items,
                    "history_action_types": history_actions,
                    "history_timestamps": history_timestamps,
                    "history_len": len(history_items),
                }
            )
            split_counts[split] += 1
            if len(records[split]) >= 10_000:
                write_sample_batch(split, records[split], writers)
    finally:
        for split in ("train", "valid", "test"):
            write_sample_batch(split, records[split], writers)
            writers[split].close()

    manifest = {
        "data_root": str(root),
        "output": str(output),
        "max_history": max_history,
        "limit_users": limit_users,
        "target_protocol": "user_last_click_then_filter_by_released_candidate_pool",
        "split_protocol": "global_target_timestamp_70_15_15",
        "boundary_q70": q70,
        "boundary_q85": q85,
        "candidate_rid_count": len(valid_targets),
        "selected_users": len(targets),
        "materialized_users": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "empty_history_users": empty_history,
        "unknown_action_history_events": null_action_events,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build next-click samples from TencentGR-1M.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--limit-users", type=int, default=None)
    args = parser.parse_args()

    manifest = build_samples(
        args.data_root,
        args.output,
        max_history=args.max_history,
        limit_users=args.limit_users,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
