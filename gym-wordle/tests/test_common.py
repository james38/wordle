import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
    word_feature_matrix,
)


def test_invalid_from_mask():
    mask = np.array([True, False, True, False])
    assert invalid_from_mask(mask).tolist() == [1, 3]
    assert invalid_from_mask(np.ones(3, dtype=bool)).shape == (0,)


def test_masked_argmax_single_mask_applies_to_all_rows():
    q = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    a, v = masked_argmax(q, np.array([1]))
    assert a.tolist() == [[2], [0]] and v.tolist() == [[3.0], [9.0]]
    assert a.dtype == torch.int64


def test_masked_argmax_list_of_masks_one_per_row():
    q = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    a, _ = masked_argmax(q, [np.array([1]), np.array([0, 2])])
    assert a.tolist() == [[2], [1]]


def test_masked_argmax_list_with_one_row_does_not_crash():
    q = torch.tensor([[1.0, 5.0, 3.0]])
    a, v = masked_argmax(q, [np.array([1])])
    assert a.tolist() == [[2]] and v.tolist() == [[3.0]]


def test_masked_argmax_does_not_mutate_input():
    q = torch.tensor([[1.0, 5.0]])
    masked_argmax(q, np.array([1]))
    assert q.tolist() == [[1.0, 5.0]]


@pytest.fixture
def buf():
    return ReplayBuffer(capacity=4, obs_size=3, rng=np.random.default_rng(0))


def _add(buf, reward=0.0, done=False):
    return buf.add(np.zeros(3, np.int8), 1, reward, np.ones(3, np.int8), done, np.array([0]))


def test_new_rows_get_current_max_priority(buf):
    _add(buf)
    assert buf.priority[0] == 1.0
    buf.update_priority(np.array([0]), np.array([0.25]))
    _add(buf)
    assert buf.priority[1] == 0.25
    buf.update_priority(np.array([1]), np.array([2.0]))
    _add(buf)
    assert buf.priority[2] == 2.0


def test_ring_wraps_and_len_caps(buf):
    for _ in range(6):
        _add(buf)
    assert len(buf) == 4 and buf.n_seen == 6


def test_update_priority_floors_at_eps(buf):
    _add(buf)
    buf.update_priority(np.array([0]), np.array([0.0]))
    assert buf.priority[0] == buf.eps


def test_sample_prioritized_shapes_and_weight_range(buf):
    for r in [0.0, 1.0, 0.0]:
        _add(buf, r)
    buf.update_priority(np.array([0, 1, 2]), np.array([0.1, 1.0, 0.5]))
    idx, w = buf.sample_prioritized(8, beta=0.4)
    assert idx.shape == (8,) and w.shape == (8,) and w.dtype == np.float32
    assert idx.max() < 3
    assert 0 < w.min() and w.max() <= 1.0 + 1e-6


def test_sample_with_probs(buf):
    for _ in range(3):
        _add(buf)
    idx = buf.sample_with_probs(5, np.array([0.0, 1.0, 0.0]))
    assert idx.tolist() == [1] * 5


def test_keep_random_subset(buf):
    for r in [0.0, 1.0, 2.0, 3.0]:
        _add(buf, r)
    buf.keep_random_subset(2)
    assert len(buf) == 2 and buf.n_seen == 2
    assert set(buf.reward[:2].tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert len(buf.reward[:2].tolist()) == len(set(buf.reward[:2].tolist()))
    assert all(isinstance(x, np.ndarray) for x in buf.invalid_next[:2])


class Tiny(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.register_buffer("table", torch.zeros(n_out))
        self.ctor_kwargs = {"n_in": n_in, "n_out": n_out}

    def forward(self, x):
        return self.lin(x) + self.table


def test_checkpoint_roundtrip(tmp_path):
    m = Tiny(3, 2)
    m.table[:] = torch.tensor([1.0, 2.0])
    path = tmp_path / "models" / "m.pt"  # directory does not exist yet
    save_checkpoint(m, str(path))
    assert path.exists() and (tmp_path / "models" / "m.pt.json").exists()
    meta = json.load(open(str(path) + ".json"))
    assert meta == {"class": "Tiny", "kwargs": {"n_in": 3, "n_out": 2}}
    m2 = load_checkpoint(Tiny, str(path), torch.device("cpu"))
    x = torch.randn(4, 3)
    assert torch.allclose(m(x), m2(x))


def test_word_feature_matrix():
    phi = word_feature_matrix(["apple", "stale", "slate"], 5)
    assert phi.shape == (3, 156) and phi.dtype == torch.float32
    row = phi[0]
    assert row[:130].sum() == 5  # one letter per position
    assert row[130:].sum() == 5  # counts sum to word length
    assert row[130 + ord("p") - ord("a")] == 2
    assert torch.equal(phi[1, 130:], phi[2, 130:])  # anagrams share counts
    assert not torch.equal(phi[1, :130], phi[2, :130])  # but not positions
