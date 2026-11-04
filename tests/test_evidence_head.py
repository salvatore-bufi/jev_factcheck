"""Offline checks of the evidence head, its losses and the metrics (CPU, no model, no data files).

    python tests/test_evidence_head.py          # or: python -m pytest tests

What is checked (doc 08 §8, doc 09 §6):
    - output shapes, with and without pair scores, with one sentence, with no transformer;
    - padding invariance: a row gets the same scores alone or in a batch with a longer row;
    - the numeric examples of the plan: reference loss 1.975; ZLPR 2.526 -> 1.473; mil_neg ~ 0 far below threshold;
    - a row with evidence term "none" gives zero loss and zero gradient;
    - the metrics examples of doc 10 §2 and the selection rule.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence.head import EvidenceHead  # noqa: E402
from evidence.losses import TERMS, pair_loss, ref_loss, zlpr_loss  # noqa: E402
from evidence.metrics import select, sentence_metrics  # noqa: E402

D, L, A = 32, 4, 128           # small fake model size, 4 layers, 128 attention features


def fake_batch(lengths, seed=0):
    """Random inputs for rows with the given numbers of sentences (padded to the longest)."""
    g = torch.Generator().manual_seed(seed)
    B, N = len(lengths), max(lengths)
    mask = torch.zeros(B, N)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    return (torch.randn(B, L, D, generator=g), torch.randn(B, N, L, D, generator=g),
            torch.randn(B, N, L, D, generator=g), torch.rand(B, N, A, generator=g),
            torch.randint(1, 30, (B, N), generator=g).float(), mask)


def test_shapes():
    for kw in ({}, {"use_pair": True}, {"layers": 0}, {"use_mix": False, "use_mean": False, "use_att": False}):
        head = EvidenceHead(D, L, A, **kw).eval()
        s, s0, S = head(*fake_batch([5, 3]))
        assert s.shape == (2, 5) and s0.shape == (2,)
        assert (S is not None) == bool(kw.get("use_pair")) and (S is None or S.shape == (2, 5, 5))
    s, s0, _ = EvidenceHead(D, L, A).eval()(*fake_batch([1]))           # one sentence
    assert s.shape == (1, 1)


def test_padding_invariance():
    torch.manual_seed(0)
    head = EvidenceHead(D, L, A, use_pair=True).eval()
    q, he, hm, att, nt, mask = fake_batch([4, 9], seed=1)
    s_batch, s0_batch, _ = head(q, he, hm, att, nt, mask)
    s_alone, s0_alone, _ = head(q[:1], he[:1, :4], hm[:1, :4], att[:1, :4], nt[:1, :4], mask[:1, :4])
    assert torch.allclose(s_batch[0, :4], s_alone[0], atol=1e-4), (s_batch[0, :4], s_alone[0])
    assert torch.allclose(s0_batch[0], s0_alone[0], atol=1e-4)


def test_ref_loss_example():
    # 40 sentences, evidence = sentences 1 and 40 (indices 0 and 39), every probability 0.02, alpha 10, omega 1
    s = torch.full((1, 40), math.log(0.02 / 0.98))
    gold = torch.zeros(1, 40)
    gold[0, [0, 39]] = 1
    loss = ref_loss(s, gold, torch.ones(1, 40), torch.ones(1, 40), alpha=10.0, gamma=0.0)
    assert abs(loss.item() - 1.975) < 1e-3, loss.item()


def test_zlpr_examples():
    gold = torch.zeros(1, 40)
    gold[0, [0, 39]] = 1
    mask, omega, sup = torch.ones(1, 40), torch.ones(1, 40), torch.tensor([TERMS["supervised"]])
    before = torch.full((1, 40), -3.0)
    before[0, 0], before[0, 39], before[0, 5] = 2.0, -0.5, 0.5          # sentence 5 resembles the claim
    after = before.clone()
    after[0, 39], after[0, 5] = 1.5, -1.0
    l_before = zlpr_loss(before, torch.zeros(1), gold, mask, omega, sup).item()
    l_after = zlpr_loss(after, torch.zeros(1), gold, mask, omega, sup).item()
    assert abs(l_before - 2.526) < 1e-3 and abs(l_after - 1.473) < 1e-3, (l_before, l_after)
    far_below = zlpr_loss(torch.full((1, 40), -20.0), torch.zeros(1), torch.zeros(1, 40), mask, omega,
                          torch.tensor([TERMS["mil_neg"]])).item()
    assert far_below < 1e-6


def test_none_term_has_no_gradient():
    s = torch.randn(1, 6, requires_grad=True)
    s0 = torch.zeros(1, requires_grad=True)
    loss = zlpr_loss(s, s0, torch.zeros(1, 6), torch.ones(1, 6), torch.ones(1, 6), torch.tensor([TERMS["none"]]))
    loss.sum().backward()
    assert loss.item() == 0.0 and s.grad.abs().sum().item() == 0.0


def test_pair_loss_runs():
    S = torch.randn(2, 6, 6, requires_grad=True)
    gold = torch.zeros(2, 6)
    gold[0, [1, 4]] = 1
    loss = pair_loss(S, gold, torch.ones(2, 6))
    loss.backward()
    assert loss.item() > 0 and S.grad is not None


def test_metrics_examples():
    m = sentence_metrics([1, 4], [[1, 4], [7]])
    assert m["precision"] == 1 and abs(m["recall"] - 2 / 3) < 1e-9 and abs(m["f1"] - 0.8) < 1e-9
    assert m["full_set"] and m["exact"]
    m = sentence_metrics([1], [[1, 4]])
    assert m["recall"] == 0.5 and abs(m["f1"] - 2 / 3) < 1e-9 and not m["full_set"]
    assert select([0.1, 0.9, 0.5, 0.7], 0.6) == [1, 3]
    assert select([0.1, 0.2], 5.0) == [1]                                   # at least the best one
    assert len(select([float(k) for k in range(20)], -1.0)) == 5           # at most five


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
