"""STEP 7b — The losses of the evidence head (doc 08 §6, doc 15 §4).

Where it sits in the workflow:
    EvidenceHead -> s, s0  ->  [losses]  ->  backward  (in evidence.train_head)

Every row has an evidence term (field evidence_term, set by the mix):
    supervised  positive with known evidence     gold sentences above the threshold, the others below
    mil_neg     negative                         NO sentence above the threshold
    mil_pos     positive without known evidence  at least one sentence above, but few  (not used by default)
    none        no evidence loss                 0
and two weights: row_weight (how much the row counts) and omega (weight of the non-gold sentences: low where the
annotators may have missed valid evidence).

Two losses:
    ref_loss   reference (experiment H0): weighted per-sentence BCE + a ranking term; FIXED threshold chosen on dev
    zlpr_loss  main: ZLPR with the document's own threshold s0 (a sentence is evidence iff s[k] > s0)
pair_loss is the extra term of H-PAIR.

All functions take padded batches: tensors [B, N] with sent_mask = 1 on real sentences.
"""
import torch
import torch.nn.functional as F

TERMS = {"supervised": 0, "mil_neg": 1, "mil_pos": 2, "none": 3}


def ref_loss(s, gold, sent_mask, omega, alpha=10.0, gamma=0.5, margin=1.0):
    """Reference loss (H0): every sentence is an independent yes/no, plus a ranking term.

    Input:  s [B, N] scores, gold [B, N] 0/1, sent_mask [B, N], omega [B, N] weight of non-gold sentences,
            alpha  weight of the gold sentences (they are few: ~2 out of 50),
            gamma  weight of the ranking term, margin  how much a gold sentence must beat the best non-gold one.
    Output: [B] loss per row. Meant for supervised rows only (train_head gives it no other rows).
    """
    bce = F.binary_cross_entropy_with_logits(s, gold.float(), reduction="none")
    w = torch.where(gold > 0, torch.full_like(s, alpha), omega) * sent_mask
    l_bce = (bce * w).sum(1) / sent_mask.sum(1).clamp_min(1)
    best_neg = s.masked_fill((gold > 0) | (sent_mask == 0), float("-inf")).amax(1, keepdim=True)
    rank = F.softplus(margin - s + best_neg).nan_to_num(posinf=0.0) * (gold > 0)
    l_rank = rank.sum(1) / (gold > 0).sum(1).clamp(min=1)
    return l_bce + gamma * l_rank


def zlpr_loss(s, s0, gold, sent_mask, omega, term, mu_neg=0.3, mu_pos=0.3, rho=0.1):
    """Main loss: ZLPR with the document threshold s0 as the "threshold class".

    Input:  s [B, N], s0 [B], gold [B, N], sent_mask [B, N], omega [B, N],
            term [B] long: 0 supervised, 1 mil_neg, 2 mil_pos, 3 none (see TERMS),
            mu_neg / mu_pos  weights of the negative / weak-positive terms, rho  sparsity weight of mil_pos.
    Output: [B] loss per row:
        supervised:  log(1 + sum_{i not gold} omega_i e^(s_i - s0))  +  log(1 + sum_{j gold} e^(s0 - s_j))
                     (grows if a non-gold sentence is above s0)        (grows if a gold sentence is below s0)
        mil_neg:     mu_neg * log(1 + sum_i e^(s_i - s0))
        mil_pos:     mu_pos * [ log(1 + e^(s0 - LSE_j s_j)) + rho * mean_i sigmoid(s_i - s0) ]
        none:        0
    """
    ninf = torch.finfo(s.dtype).min
    real = sent_mask.bool()
    d = s - s0[:, None]                                           # distance from the threshold
    is_neg = (gold == 0) & real
    # log(1 + sum exp(x)) = softplus(logsumexp(x)); omega enters as + log(omega)
    l_neg = F.softplus(torch.logsumexp((d + omega.clamp_min(1e-6).log()).masked_fill(~is_neg, ninf), 1))
    l_pos = F.softplus(torch.logsumexp((-d).masked_fill(~(gold > 0), ninf), 1)) * (gold.sum(1) > 0)
    supervised = l_neg + l_pos
    lse = torch.logsumexp(d.masked_fill(~real, ninf), 1)
    mil_neg = mu_neg * F.softplus(lse)
    spars = (torch.sigmoid(d) * sent_mask).sum(1) / sent_mask.sum(1).clamp_min(1)
    mil_pos = mu_pos * (F.softplus(-lse) + rho * spars)
    out = torch.stack([supervised, mil_neg, mil_pos, torch.zeros_like(lse)], 1)
    return out.gather(1, term[:, None]).squeeze(1)


def pair_loss(S, gold, sent_mask, n_hard=5):
    """H-PAIR term: pair score high between two gold sentences, low between a gold one and its hardest non-gold ones.

    Input:  S [B, N, N] pair scores, gold [B, N], sent_mask [B, N], n_hard (non-gold partners per gold sentence).
    Output: scalar loss (0 if the batch has no pair of gold sentences).
    """
    g = gold.bool() & sent_mask.bool()
    eye = torch.eye(S.size(1), dtype=torch.bool, device=S.device)
    pos_pairs = g[:, :, None] & g[:, None, :] & ~eye
    if not pos_pairs.any():
        return S.sum() * 0.0
    l_pos = F.binary_cross_entropy_with_logits(S[pos_pairs], torch.ones_like(S[pos_pairs]))
    neg_cand = S.masked_fill(~(g[:, :, None] & (~g & sent_mask.bool())[:, None, :]), float("-inf"))
    hard = neg_cand.topk(min(n_hard, S.size(1)), -1).values
    hard = hard[torch.isfinite(hard)]
    l_neg = F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard)) if hard.numel() else S.sum() * 0.0
    return l_pos + l_neg
