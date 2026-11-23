from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


class SafeUnpickler(pickle.Unpickler):
    """Allow only the mapping types needed by the official indexer pickle."""

    ALLOWED = {
        ("builtins", "dict"),
        ("collections", "defaultdict"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Blocked pickle import: {module}.{name}")
        return super().find_class(module, name)


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def file_summary(path: Path) -> dict[str, Any]:
    files = parquet_files(path)
    return {
        "files": len(files),
        "bytes": sum(file.stat().st_size for file in files),
        "schemas": {
            str(file.name): str(pq.ParquetFile(file).schema_arrow)
            for file in files[:1]
        },
        "row_groups": sum(pq.ParquetFile(file).num_row_groups for file in files),
    }


def audit_simple_table(path: Path) -> dict[str, Any]:
    files = parquet_files(path)
    rows = 0
    columns: set[str] = set()
    null_counts: Counter[str] = Counter()

    for file in files:
        parquet = pq.ParquetFile(file)
        columns.update(parquet.schema_arrow.names)
        for batch in parquet.iter_batches(batch_size=100_000):
            rows += batch.num_rows
            for name in batch.schema.names:
                nulls = batch.column(batch.schema.get_field_index(name)).null_count
                null_counts[name] += nulls

    return {
        "files": len(files),
        "rows": rows,
        "columns": sorted(columns),
        "null_counts": dict(sorted(null_counts.items())),
    }


def audit_sequences(path: Path) -> dict[str, Any]:
    files = parquet_files(path)
    users = 0
    actions: Counter[str] = Counter()
    sequence_lengths: Counter[int] = Counter()
    unique_items: set[int] = set()
    timestamp_min: int | None = None
    timestamp_max: int | None = None
    users_with_click = 0
    click_targets = 0
    malformed_rows = 0

    for file_index, file in enumerate(files, start=1):
        parquet = pq.ParquetFile(file)
        print(f"[seq {file_index}/{len(files)}] {file.name}", flush=True)
        for batch in parquet.iter_batches(batch_size=20_000, columns=["user_id", "seq"]):
            seq_array = batch.column(batch.schema.get_field_index("seq"))
            lengths = pc.list_value_length(seq_array)
            for length in lengths.to_pylist():
                sequence_lengths[int(length or 0)] += 1
            users += batch.num_rows

            flat = pc.list_flatten(seq_array)
            if len(flat) == 0:
                continue

            action_array = pc.struct_field(flat, "action_type")
            item_array = pc.struct_field(flat, "item_id")
            timestamp_array = pc.struct_field(flat, "timestamp")

            action_counter = pc.value_counts(action_array)
            for entry in action_counter.to_pylist():
                value = entry["values"]
                key = "null" if value is None else str(int(value))
                actions[key] += int(entry["counts"])

            batch_items = pc.unique(item_array).to_pylist()
            unique_items.update(int(item) for item in batch_items if item is not None)

            ts_stats = pc.min_max(timestamp_array).as_py()
            if ts_stats["min"] is not None:
                current_min = int(ts_stats["min"])
                current_max = int(ts_stats["max"])
                timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
                timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

            # Count users with at least one click and per-user click counts.
            if "user_id" in batch.schema.names:
                user_array = batch.column(batch.schema.get_field_index("user_id")).to_pylist()
                seq_list = seq_array.to_pylist()
                if len(user_array) != len(seq_list):
                    malformed_rows += abs(len(user_array) - len(seq_list))
                for seq in seq_list:
                    if not seq:
                        continue
                    clicks = sum(
                        1
                        for event in seq
                        if event.get("action_type") is not None
                        and int(event["action_type"]) == 1
                    )
                    if clicks:
                        users_with_click += 1
                        click_targets += clicks

    total_length = sum(length * count for length, count in sequence_lengths.items())
    nonempty_lengths = [
        length for length, count in sequence_lengths.items() if length > 0 for _ in range(count)
    ]
    nonempty_lengths.sort()

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
        return values[index]

    return {
        "files": len(files),
        "users": users,
        "events": total_length,
        "unique_items": len(unique_items),
        "action_counts": dict(sorted(actions.items())),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "sequence_length": {
            "min": min(nonempty_lengths) if nonempty_lengths else 0,
            "p50": percentile(nonempty_lengths, 0.50),
            "p90": percentile(nonempty_lengths, 0.90),
            "p95": percentile(nonempty_lengths, 0.95),
            "p99": percentile(nonempty_lengths, 0.99),
            "max": max(nonempty_lengths) if nonempty_lengths else 0,
            "mean": round(total_length / users, 4) if users else 0.0,
        },
        "users_with_click": users_with_click,
        "click_events": click_targets,
        "malformed_rows": malformed_rows,
        "_unique_items_set": unique_items,
    }


def load_indexer(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return SafeUnpickler(handle).load()


def summarize_indexer(indexer: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"top_level_keys": sorted(indexer.keys())}
    for key in ("u", "i"):
        value = indexer.get(key)
        if isinstance(value, dict):
            summary[key] = {
                "type": type(value).__name__,
                "size": len(value),
                "key_type": type(next(iter(value), None)).__name__,
                "value_type": type(next(iter(value.values()), None)).__name__,
            }
    features = indexer.get("f")
    if isinstance(features, dict):
        summary["f"] = {
            "feature_count": len(features),
            "feature_ids": sorted(str(key) for key in features),
        }
    return summary


def audit_candidate_and_coverage(
    candidate_path: Path,
    indexer: dict[str, Any],
    sequence_items: set[int],
) -> dict[str, Any]:
    files = parquet_files(candidate_path)
    rows = 0
    candidate_oids: set[int] = set()
    retrieval_ids: set[int] = set()
    cold_feature_nonzero: Counter[str] = Counter()

    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=100_000):
            rows += batch.num_rows
            names = batch.schema.names
            item_column = "item_id" if "item_id" in names else None
            retrieval_column = "retrieval_id" if "retrieval_id" in names else None
            if item_column:
                candidate_oids.update(
                    int(value)
                    for value in batch.column(item_column).to_pylist()
                    if value is not None
                )
            if retrieval_column:
                retrieval_ids.update(
                    int(value)
                    for value in batch.column(retrieval_column).to_pylist()
                    if value is not None
                )

            for name in names:
                if name in {"item_id", "retrieval_id"}:
                    continue
                column = batch.column(name)
                if not str(column.type).startswith("struct"):
                    continue
                cold_values = pc.struct_field(column, "cold_start")
                nonzero = pc.sum(pc.cast(cold_values, "int64")).as_py()
                cold_feature_nonzero[name] += int(nonzero or 0)

    oid_to_rid = indexer.get("i", {})
    mapped_rids = {
        int(oid_to_rid[oid])
        for oid in candidate_oids
        if oid in oid_to_rid
    }
    candidate_in_sequence = mapped_rids & sequence_items

    return {
        "files": len(files),
        "rows": rows,
        "unique_candidate_oids": len(candidate_oids),
        "unique_retrieval_ids": len(retrieval_ids),
        "candidate_with_indexer_rid": len(mapped_rids),
        "candidate_without_indexer_rid": len(candidate_oids) - len(mapped_rids),
        "candidate_in_sequence": len(candidate_in_sequence),
        "candidate_not_in_sequence": len(mapped_rids - sequence_items),
        "sequence_items_in_candidate": len(sequence_items & mapped_rids),
        "sequence_items_not_in_candidate": len(sequence_items - mapped_rids),
        "cold_start_nonzero_by_feature": dict(sorted(cold_feature_nonzero.items())),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, set):
        return sorted(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TencentGR-1M parquet data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_root
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    indexer_path = root / "indexer.pkl"
    indexer = load_indexer(indexer_path)

    print("Auditing sequences...", flush=True)
    sequences = audit_sequences(root / "seq")
    sequence_items = sequences.pop("_unique_items_set")

    print("Auditing item features...", flush=True)
    item_features = audit_simple_table(root / "item_feat")
    print("Auditing user features...", flush=True)
    user_features = audit_simple_table(root / "user_feat")

    print("Auditing candidates and coverage...", flush=True)
    candidates = audit_candidate_and_coverage(
        root / "candidate",
        indexer=indexer,
        sequence_items=sequence_items,
    )

    report = {
        "data_root": str(root),
        "file_summary": {
            "seq": file_summary(root / "seq"),
            "item_feat": file_summary(root / "item_feat"),
            "user_feat": file_summary(root / "user_feat"),
            "candidate": file_summary(root / "candidate"),
        },
        "indexer": summarize_indexer(indexer),
        "sequences": sequences,
        "item_features": item_features,
        "user_features": user_features,
        "candidates": candidates,
    }

    output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Audit report written to {output}", flush=True)


if __name__ == "__main__":
    main()
