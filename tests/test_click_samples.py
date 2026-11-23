from taac.data.click_samples import (
    assign_split,
    history_before_target,
    select_last_click_if_candidate,
)


def test_select_last_click_skips_exposure_and_filters_by_candidate():
    events = [
        {"item_id": 10, "action_type": 0, "timestamp": 1},
        {"item_id": 11, "action_type": 1, "timestamp": 2},
        {"item_id": 12, "action_type": 1, "timestamp": 3},
        {"item_id": 11, "action_type": 1, "timestamp": 4},
    ]
    assert select_last_click_if_candidate(events, {11}) == (11, 4, 3)


def test_does_not_fall_back_to_earlier_candidate_click():
    events = [
        {"item_id": 11, "action_type": 1, "timestamp": 2},
        {"item_id": 99, "action_type": 1, "timestamp": 3},
    ]
    assert select_last_click_if_candidate(events, {11}) is None


def test_history_uses_strictly_earlier_events_and_unknown_action():
    events = [
        {"item_id": 1, "action_type": 0, "timestamp": 10},
        {"item_id": 2, "action_type": None, "timestamp": 20},
        {"item_id": 3, "action_type": 1, "timestamp": 30},
    ]
    items, actions, timestamps = history_before_target(events, 30, max_history=100)
    assert items == [1, 2]
    assert actions == [0, 3]
    assert timestamps == [10, 20]


def test_history_respects_max_length():
    events = [
        {"item_id": index, "action_type": 0, "timestamp": index}
        for index in range(10)
    ]
    items, _, _ = history_before_target(events, 100, max_history=3)
    assert items == [7, 8, 9]


def test_assign_split_boundaries():
    assert assign_split(70, 70, 85) == "train"
    assert assign_split(71, 70, 85) == "valid"
    assert assign_split(86, 70, 85) == "test"
