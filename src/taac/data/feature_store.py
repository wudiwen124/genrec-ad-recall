from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


ITEM_FEATURE_IDS = [
    "100",
    "101",
    "102",
    "112",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
]

USER_SCALAR_FEATURE_IDS = ["103", "104", "105", "109"]
USER_ARRAY_FEATURE_IDS = ["106", "107", "108", "110"]
USER_ARRAY_MAX_LENGTHS = {"106": 8, "107": 1, "108": 2, "110": 2}


def load_indexer(data_root: Path) -> dict:
    with (data_root / "indexer.pkl").open("rb") as handle:
        return pickle.load(handle)


def item_feature_vocab_sizes(indexer: dict) -> list[int]:
    return [len(indexer["f"][feature_id]) + 1 for feature_id in ITEM_FEATURE_IDS]


def build_item_feature_matrix(
    data_root: Path,
    cache_dir: Path,
) -> tuple[np.ndarray, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "item_features.npy"
    metadata_path = cache_dir / "item_features.json"
    if matrix_path.exists() and metadata_path.exists():
        return np.load(matrix_path, mmap_mode=None), json.loads(metadata_path.read_text())

    indexer = load_indexer(data_root)
    max_rid = max(int(value) for value in indexer["i"].values())
    matrix = np.zeros((max_rid + 1, len(ITEM_FEATURE_IDS)), dtype=np.int32)

    files = sorted((data_root / "item_feat").glob("*.parquet"))
    for file_index, file in enumerate(files, start=1):
        print(f"[item features {file_index}/{len(files)}] {file.name}", flush=True)
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(
            batch_size=200_000,
            columns=["item_id", *ITEM_FEATURE_IDS],
        ):
            ids = pc.fill_null(batch.column("item_id"), 0).to_numpy()
            for feature_index, feature_id in enumerate(ITEM_FEATURE_IDS):
                values = pc.fill_null(batch.column(feature_id), 0).to_numpy()
                matrix[ids, feature_index] = values.astype(np.int32, copy=False)

    metadata = {
        "feature_ids": ITEM_FEATURE_IDS,
        "vocab_sizes": item_feature_vocab_sizes(indexer),
        "item_count": max_rid,
        "shape": list(matrix.shape),
    }
    np.save(matrix_path, matrix)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata


def user_feature_vocab_sizes(indexer: dict) -> tuple[list[int], list[int]]:
    scalar = [len(indexer["f"][feature_id]) + 1 for feature_id in USER_SCALAR_FEATURE_IDS]
    arrays = [len(indexer["f"][feature_id]) + 1 for feature_id in USER_ARRAY_FEATURE_IDS]
    return scalar, arrays


def build_user_feature_store(
    data_root: Path,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scalar_path = cache_dir / "user_scalars.npy"
    metadata_path = cache_dir / "user_features.json"
    array_paths = {
        feature_id: cache_dir / f"user_array_{feature_id}.npy"
        for feature_id in USER_ARRAY_FEATURE_IDS
    }
    if scalar_path.exists() and metadata_path.exists() and all(path.exists() for path in array_paths.values()):
        return (
            np.load(scalar_path),
            {feature_id: np.load(path) for feature_id, path in array_paths.items()},
            json.loads(metadata_path.read_text()),
        )

    indexer = load_indexer(data_root)
    max_user_rid = max(int(value) for value in indexer["u"].values())
    scalar_matrix = np.zeros(
        (max_user_rid + 1, len(USER_SCALAR_FEATURE_IDS)),
        dtype=np.int32,
    )
    array_matrices = {
        feature_id: np.zeros(
            (max_user_rid + 1, USER_ARRAY_MAX_LENGTHS[feature_id]),
            dtype=np.int32,
        )
        for feature_id in USER_ARRAY_FEATURE_IDS
    }

    files = sorted((data_root / "user_feat").glob("*.parquet"))
    for file_index, file in enumerate(files, start=1):
        print(f"[user features {file_index}/{len(files)}] {file.name}", flush=True)
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(
            batch_size=200_000,
            columns=["user_id", *USER_SCALAR_FEATURE_IDS, *USER_ARRAY_FEATURE_IDS],
        ):
            user_ids = pc.fill_null(batch.column("user_id"), 0).to_numpy()
            for feature_index, feature_id in enumerate(USER_SCALAR_FEATURE_IDS):
                values = pc.fill_null(batch.column(feature_id), 0).to_numpy()
                scalar_matrix[user_ids, feature_index] = values.astype(np.int32, copy=False)
            for feature_id in USER_ARRAY_FEATURE_IDS:
                values = batch.column(feature_id).to_pylist()
                max_length = USER_ARRAY_MAX_LENGTHS[feature_id]
                output = np.zeros((len(values), max_length), dtype=np.int32)
                for row_index, value_list in enumerate(values):
                    if not value_list:
                        continue
                    truncated = value_list[:max_length]
                    output[row_index, : len(truncated)] = truncated
                array_matrices[feature_id][user_ids] = output

    metadata = {
        "scalar_feature_ids": USER_SCALAR_FEATURE_IDS,
        "array_feature_ids": USER_ARRAY_FEATURE_IDS,
        "array_max_lengths": USER_ARRAY_MAX_LENGTHS,
        "scalar_vocab_sizes": user_feature_vocab_sizes(indexer)[0],
        "array_vocab_sizes": user_feature_vocab_sizes(indexer)[1],
        "user_count": max_user_rid,
    }
    np.save(scalar_path, scalar_matrix)
    for feature_id, matrix in array_matrices.items():
        np.save(array_paths[feature_id], matrix)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scalar_matrix, array_matrices, metadata
