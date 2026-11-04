"""STEP 7a — The evidence head: a small network that scores every sentence of the document (doc 08).

Where it sits in the workflow:
    features (saved or live)  ->  [EvidenceHead]  ->  s[k] (score of sentence k), s0 (threshold of this document)
                                                  ->  evidence.metrics.select: sentences with s[k] > s0 + delta

The network, in four stages (B = rows in the batch, N = sentences, L = layers read, D = model size, d = 256):
    A. LAYER MIX        m(v) = LayerNorm( sum_l softmax(alpha)_l * v[l] )     same weights for h_end, h_mean and q
    B. SENTENCE VECTOR  x[k] = W_h [m(h_end[k]) ; m(h_mean[k])] + W_a [att[k] ; log n_tok[k]] + P[k] + T_sentence
       ANSWER VECTOR    x0   = W_q m(q) + T_answer
    C. COMMUNICATION    a NON-causal transformer encoder (2 layers, 4 heads) over [x0, x[0], ..., x[N-1]]:
                        every sentence sees every other one and x0 (this solves sparse evidence: in the causal model
                        sentence 1 cannot see sentence 40; here it can)
    D. SCORES           s[k] = w_o . X[k] + b_o        s0 = w_o . X0 + b_o   (same output layer: same scale)
    (optional H-PAIR)   s[k] += tau * max_j (S[k, j] + s[j])   with a low-rank pair score S[k, j]

Every part can be switched off with an option, so all experiments use this one class (doc 11):
    use_mix=False   only the last layer read (no mix)        use_mean=False  only the last token of a sentence
    use_att=False   no attention features                    use_pos=False   no sentence-position vector
    layers=0        no transformer: each sentence scored alone (experiment H3a)
    use_pair=True   pair scores (H-PAIR)
~3.7 M parameters for KnowLine-4B (D = 2,560) with everything on.
"""
import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, d_model, n_mix, n_att, d=256, layers=2, heads=4, max_sent=512, use_mix=True, use_mean=True,
                 use_att=True, use_pos=True, use_pair=False, pair_rank=64, att_dropout=0.1, dropout=0.1):
        """Build the head.

        Input:  d_model   size of the model's hidden vectors (2,560 for KnowLine-4B)
                n_mix     number of layers in the input tensors (1 or 4)
                n_att     number of attention features per sentence (128 = 8 layers x 16 heads)
                d         size inside the head (256)
                layers    transformer layers between the sentences (0 = none)
                heads     attention heads of that transformer
                max_sent  most sentences per document (size of the position table)
                use_*     switches described in the file header
                pair_rank rank of the pair score (H-PAIR)
                att_dropout  probability of zeroing ALL attention features of a sentence during training
                dropout   dropout inside the transformer
        """
        super().__init__()
        self.use_mix, self.use_mean, self.use_att = use_mix, use_mean, use_att
        self.use_pos, self.use_pair, self.att_dropout = use_pos, use_pair, att_dropout
        self.mix = nn.Parameter(torch.zeros(n_mix))                       # alpha: 0 -> equal weights
        self.ln = nn.LayerNorm(d_model)                                    # layers have very different scales
        self.proj_h = nn.Linear(d_model * (2 if use_mean else 1), d)       # [last token ; mean] -> d
        self.proj_q = nn.Linear(d_model, d)                                # answer position -> d
        self.proj_a = nn.Linear(n_att + 1, d)                              # attention + log(n tokens) -> d
        self.pos = nn.Embedding(max_sent, d)                               # position of the sentence
        self.kind = nn.Embedding(2, d)                                     # 0 = answer vector x0, 1 = sentence
        if layers > 0:
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        else:
            self.encoder = None
        self.out = nn.Linear(d, 1)                                         # one output for sentences and threshold
        if use_pair:
            self.pair = nn.ModuleList([nn.Linear(d, pair_rank, bias=False), nn.Linear(d, pair_rank, bias=False)])
            self.pair_tau = nn.Parameter(torch.tensor(0.1))

    def mixed(self, x):
        """Stage A on one input.

        Input:  x [..., n_mix, D].
        Output: [..., D]: the weighted sum of the layers (or only the last layer if use_mix=False), normalised.
        """
        if not self.use_mix:
            return self.ln(x[..., -1, :])
        return self.ln((self.mix.softmax(0)[:, None] * x).sum(-2))

    def layer_weights(self):
        """softmax(alpha): how much each layer counts (read after training: a layer under 10% can be dropped)."""
        return self.mix.softmax(0).detach().cpu().tolist()

    def forward(self, q, h_end, h_mean, att, n_tok, sent_mask):
        """Score the sentences of a batch of documents.

        Input:  q          [B, L, D]       answer-position vectors
                h_end      [B, N, L, D]    last-token vectors of each sentence
                h_mean     [B, N, L, D]    mean vectors (ignored if use_mean=False; may be None then)
                att        [B, N, n_att]   attention features (ignored if use_att=False)
                n_tok      [B, N]          tokens per sentence
                sent_mask  [B, N]          1 = real sentence, 0 = padding (documents of a batch have different N)
        Output: s  [B, N]   score of each sentence (padding positions: meaningless, mask them)
                s0 [B]      threshold of each document: sentence k is evidence if s[k] > s0
                S  [B, N, N] pair scores, or None without H-PAIR
        """
        h = self.mixed(h_end.float())
        if self.use_mean:
            h = torch.cat([h, self.mixed(h_mean.float())], -1)
        x = self.proj_h(h) + self.kind.weight[1]                           # [B, N, d]
        if self.use_pos:
            x = x + self.pos(torch.arange(h.size(1), device=h.device))
        if self.use_att:
            a = att.float()
            if self.training and self.att_dropout:                         # whole sentences, not single features
                a = a * (torch.rand_like(a[..., :1]) >= self.att_dropout)
            x = x + self.proj_a(torch.cat([a, n_tok.clamp_min(1).float().log()[..., None]], -1))
        x0 = (self.proj_q(self.mixed(q.float())) + self.kind.weight[0])[:, None]   # [B, 1, d]
        seq = torch.cat([x0, x], 1)                                        # [B, 1 + N, d]
        if self.encoder is not None:
            pad = torch.cat([torch.zeros_like(sent_mask[:, :1]), 1 - sent_mask], 1).bool()
            seq = self.encoder(seq, src_key_padding_mask=pad)
        o = self.out(seq)[..., 0]                                          # [B, 1 + N]
        s, s0, S = o[:, 1:], o[:, 0], None
        if self.use_pair:
            xs = seq[:, 1:]
            S = self.pair[0](xs) @ self.pair[1](xs).transpose(1, 2)        # [B, N, N]
            invalid = ~sent_mask.bool()[:, None, :] | torch.eye(S.size(1), dtype=torch.bool, device=S.device)
            support = (S + s.detach()[:, None, :]).masked_fill(invalid, float("-inf")).amax(-1)
            s = s + self.pair_tau * support.nan_to_num(neginf=0.0)         # no partner: no contribution
        return s, s0, S


def build_head(cfg, d_model, n_att=128):
    """An EvidenceHead from an experiment config (evidence/configs/<id>.json, key "head").

    Input:  cfg (dict: the "head" options + "layers_read"), d_model, n_att.
    Output: EvidenceHead. n_mix = number of layers the config reads.
    """
    opts = dict(cfg["head"])
    return EvidenceHead(d_model=d_model, n_mix=len(cfg["layers_read"]), n_att=n_att, **opts)


def load_run(run_dir, device="cpu"):
    """Load a trained head from its run folder (written by evidence.train_head) or a release folder.

    Input:  run_dir (Path with config.json, head.safetensors, selection.json), device.
    Output: (head in eval mode, config dict, selection rule dict).
    """
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    rule = json.loads((run_dir / "selection.json").read_text())
    head = build_head(cfg, cfg["d_model"])
    head.load_state_dict(load_file(str(run_dir / "head.safetensors")))
    return head.to(device).eval(), cfg, rule
