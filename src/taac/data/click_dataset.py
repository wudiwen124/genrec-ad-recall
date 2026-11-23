from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


ACTION_PAD = 4
TIME_PAD = 0


def time_gap_buckets(timestamps: list[int], num_buckets: int = 64) -> list[int]:
    gaps = []
    previous = None
    for timestamp in timestamps:
        if previous is None:
            gaps.append(0)
        else:
            gap_minutes = max(0, timestamp - previous) / 60.0
            bucket = int(round(np.log1p(gap_minutes) * 4.0))
            gaps.append(min(num_buckets - 1, bucket))
        previous = timestamp
    return gaps


class ClickSampleDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, max_samples: int | None = None):
        self.path = path
        self.table = pq.read_table(
            path,
            columns=[
                "user_id",
                "target_item_id",
                "history_item_ids",
                "history_action_types",
                "history_timestamps",
            ],
        )
        if max_samples and max_samples < len(self.table):
            self.sample_indices = np.linspace(
                0,
                len(self.table) - 1,
                num=max_samples,
                dtype=np.int64,
            )
        else:
            self.sample_indices = None
        self.length = len(self.sample_indices) if self.sample_indices is not None else len(self.table)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        source_index = int(self.sample_indices[index]) if self.sample_indices is not None else index
        return {
            "user_id": int(self.table["user_id"][source_index].as_py()),
            "target_item_id": int(self.table["target_item_id"][source_index].as_py()),
            "history_item_ids": self.table["history_item_ids"][source_index].as_py(),
            "history_action_types": self.table["history_action_types"][source_index].as_py(),
            "history_timestamps": self.table["history_timestamps"][source_index].as_py(),
        }


def collate_click_samples(
    rows: list[dict[str, object]],
    maxlen: int = 101,
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    sequence = np.zeros((batch_size, maxlen), dtype=np.int64)
    actions = np.full((batch_size, maxlen), ACTION_PAD, dtype=np.int64)
    times = np.full((batch_size, maxlen), TIME_PAD, dtype=np.int64)
    timestamps = np.zeros((batch_size, maxlen), dtype=np.int64)
    valid_mask = np.zeros((batch_size, maxlen), dtype=np.bool_)
    user_ids = np.zeros(batch_size, dtype=np.int64)
    targets = np.zeros(batch_size, dtype=np.int64)

    for row_index, row in enumerate(rows):
        history_items = row["history_item_ids"][-maxlen + 1 :]
        history_actions = row["history_action_types"][-maxlen + 1 :]
        history_times = row["history_timestamps"][-maxlen + 1 :]
        length = len(history_items)
        start = maxlen - length - 1

        user_ids[row_index] = row["user_id"]
        targets[row_index] = row["target_item_id"]
        sequence[row_index, start] = 0  # BOS item slot; user embedding is added separately.
        actions[row_index, start] = ACTION_PAD
        times[row_index, start] = TIME_PAD
        timestamps[row_index, start] = history_times[0] if history_times else 0
        valid_mask[row_index, start] = True
        sequence[row_index, start + 1 :] = history_items
        actions[row_index, start + 1 :] = history_actions
        times[row_index, start + 1 :] = time_gap_buckets(history_times)
        timestamps[row_index, start + 1 :] = history_times
        valid_mask[row_index, start + 1 :] = True

    return {
        "sequence": torch.from_numpy(sequence),
        "actions": torch.from_numpy(actions),
        "time_buckets": torch.from_numpy(times),
        "timestamps": torch.from_numpy(timestamps),
        "valid_mask": torch.from_numpy(valid_mask),
        "user_ids": torch.from_numpy(user_ids),
        "targets": torch.from_numpy(targets),
    }
