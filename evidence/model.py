"""STEP 3 — Load KnowLine-4B and run ONE forward that gives both the verdict and the per-sentence features.

Where it sits in the workflow:
    evidence.render  ->  [evidence.model.forward_sentences]  ->  evidence.extract (saved to disk) / evidence.serve

What one forward returns (doc 01 §7):
    answer_logits [2]        log-probabilities of the labels A (Yes) and B (No) at the answer position -> P(supported)
    q       [4, D]           hidden vector of the answer position, in the 4 SENTENCE_LAYERS (19, 23, 27, 31)
    h_end   [n, 4, D]        hidden vector of the LAST token of each sentence, same 4 layers
    h_mean  [n, 4, D]        MEAN of the hidden vectors of each sentence's tokens, same 4 layers
    att     [n, 128]         attention that the answer position gives to each sentence: for each of the 8
                             full-attention layers and each of their 16 heads, the sum of the weights on its tokens
    n_tok   [n]              tokens of each sentence
(D = 2,560 for KnowLine-4B, n = number of sentences.)

How (without changing the model):
    * forward HOOKS on the 4 layers: small functions called by PyTorch with each layer's output; they compute the
      per-sentence summaries at once on the GPU (the vectors of all tokens are never kept);
    * an ATTENTION FUNCTION registered in transformers ("sdpa_capture"): it runs the normal attention and, in the
      8 full-attention layers, recomputes ONE row of weights: answer position -> all tokens (fast attention kernels
      do not return weights, and asking for all of them would not fit in memory on long documents).

Run the self-test (GPU):  python -m evidence.model --selftest
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402

# ------------------------------------------------------------------ attention capture
# Shared state between forward_sentences() and the attention function (the model calls it, we cannot pass arguments).
CAPTURE = {"on": False, "rows": {}}


def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """Attention function used by the model: normal SDPA attention, plus the attention row of the last position.

    Input (given by transformers, Qwen3_5Attention.forward):
        module          the attention module (has .layer_idx)
        query           [B, 16, T, head_dim]  queries (rotary position already applied)
        key, value      [B, 4, T, head_dim]   keys and values (4 key heads shared by groups of 4 query heads)
        attention_mask  the model's mask (None for plain causal attention)
        scaling         1 / sqrt(head_dim)
    Output: what SDPA returns (attention output, None), unchanged.
    Side effect: when CAPTURE["on"], stores CAPTURE["rows"][layer] = [B, 16, T] softmax weights of the last query.
    The model is causal and we run batch 1, so the last position may see every token: no mask is needed for that row.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    out = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
    if CAPTURE["on"] and module.layer_idx in config.FULL_ATTENTION_LAYERS:
        k = key.repeat_interleave(query.size(1) // key.size(1), dim=1)          # 4 key heads -> 16
        scores = (query[:, :, -1:].float() @ k.float().transpose(-1, -2)) * scaling   # [B, 16, 1, T], fp32
        if attention_mask is not None and attention_mask.dim() == 4:            # only if the model passed a mask
            m = attention_mask[:, :, -1:, : scores.size(-1)]
            scores = scores.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else scores + m.float()
        CAPTURE["rows"][module.layer_idx] = scores.softmax(-1)[:, :, 0]         # [B, 16, T]
    return out


def register_attention():
    """Register "sdpa_capture" in transformers (once), so from_pretrained(attn_implementation="sdpa_capture") works."""
    from transformers import AttentionInterface
    AttentionInterface.register("sdpa_capture", sdpa_capture)


# ------------------------------------------------------------------ model
def load_model(device="cuda"):
    """Load KnowLine-4B (bf16) with the capturing attention, as knowline_server.HF loads it.

    Input:  device (str).
    Output: the model in eval mode, gradients off.
    """
    import transformers
    from transformers import AutoConfig
    from evidence.render import snapshot_dir
    register_attention()
    folder = str(snapshot_dir())
    cfg = AutoConfig.from_pretrained(folder)
    multimodal = hasattr(cfg, "vision_config") or hasattr(cfg, "audio_config")
    cls = getattr(transformers, "AutoModelForMultimodalLM", transformers.AutoModelForImageTextToText) if multimodal \
        else transformers.AutoModelForCausalLM
    model = cls.from_pretrained(folder, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa_capture")
    model.eval().requires_grad_(False)
    return model


def text_layers(model):
    """The list (nn.ModuleList) of the 32 text decoder layers (a multimodal model also has a vision tower).

    Input:  model.  Output: the ModuleList.
    """
    found = [m for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == config.N_LAYERS
             and "visual" not in name and "vision" not in name]
    if len(found) != 1:
        raise RuntimeError(f"expected one list of {config.N_LAYERS} layers, found {len(found)}")
    return found[0]


# ------------------------------------------------------------------ per-sentence summaries
class SentenceTap:
    """Forward hooks that summarise each sentence in some layers while the forward runs.

    Use:
        tap = SentenceTap(text_layers(model), config.SENTENCE_LAYERS)
        tap.spans = LongTensor [n, 2] (first, last token of each sentence, inclusive); tap.answer_pos = int
        model(...)                      -> tap.out[layer] = {"q": [D], "end": [n, D], "mean": [n, D]}
        tap.remove()                    when done (the hooks stay on the model otherwise)
    """

    def __init__(self, layers, layer_idx):
        """Input: layers (ModuleList of the model), layer_idx (list of layer numbers to read)."""
        self.out, self.spans, self.answer_pos = {}, None, None
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in layer_idx]

    def _hook(self, i):
        """Build the hook of layer i (a closure that knows its layer number)."""
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]       # [T, D], batch 1
            s = self.spans
            # sum of the tokens of a sentence = csum[last + 1] - csum[first]: one cumulative sum for all sentences
            csum = torch.cat([h.new_zeros(1, h.size(1), dtype=torch.float), h.float().cumsum(0)])
            mean = (csum[s[:, 1] + 1] - csum[s[:, 0]]) / (s[:, 1] - s[:, 0] + 1)[:, None].float()
            self.out[i] = {"q": h[self.answer_pos].float(), "end": h[s[:, 1]], "mean": mean.to(h.dtype)}
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def attention_per_sentence(rows, spans):
    """Sum, for each sentence, the attention weights of the answer position on its tokens.

    Input:  rows  dict layer -> [16, T] attention row (one per head), spans LongTensor [n, 2].
    Output: [n, 8 * 16] float tensor; columns are (layer 3: heads 0..15, layer 7: heads 0..15, ...).
    """
    feats = []
    for layer in sorted(rows):
        r = rows[layer].float()
        csum = torch.cat([r.new_zeros(r.size(0), 1), r.cumsum(1)], 1)          # [16, T + 1]
        feats.append((csum[:, spans[:, 1] + 1] - csum[:, spans[:, 0]]).T)      # [n, 16]
    return torch.cat(feats, 1)


_TAPS = {}


def get_tap(model):
    """The SentenceTap of this model (created once and kept: hooks cost nothing when not needed)."""
    if id(model) not in _TAPS:
        _TAPS[id(model)] = SentenceTap(text_layers(model), config.SENTENCE_LAYERS)
    return _TAPS[id(model)]


@torch.no_grad()
def forward_sentences(model, ex, features=True, to_cpu=True):
    """ONE forward of a rendered row: verdict and (optionally) the per-sentence features.

    Input:  model     from load_model()
            ex        output of evidence.render.render (input_ids, answer_pos, sent_tok_spans, n_tok, label_ids)
            features  False -> only the verdict (no summaries, no attention rows): used for timing and sufficiency
            to_cpu    True -> tensors moved to the CPU (extraction); False -> left on the GPU (server: no copy)
    Output: dict (tensors on CPU, or on the model's device with to_cpu=False)
            answer_logits [2] float32 log-probabilities of A (Yes) and B (No) over the full vocabulary
            and, with features=True: q [4, D] fp32, h_end [n, 4, D] bf16, h_mean [n, 4, D] bf16,
            att [n, 128] fp16, n_tok [n] int16
    """
    device = next(model.parameters()).device
    ids = torch.tensor([ex["input_ids"]], device=device)
    tap = get_tap(model)
    spans = torch.tensor(ex["sent_tok_spans"], device=device, dtype=torch.long)
    tap.spans, tap.answer_pos, tap.out = spans, ex["answer_pos"], {}
    CAPTURE["on"], CAPTURE["rows"] = features, {}
    if not features:
        tap.spans = spans[:1]                     # the hooks still run; with one span they cost nothing
    try:
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
    finally:
        CAPTURE["on"] = False
    move = (lambda t: t.cpu()) if to_cpu else (lambda t: t)
    out = {"answer_logits": move(logits[ex["label_ids"]])}
    if not features:
        return out
    L = config.SENTENCE_LAYERS
    out["q"] = move(torch.stack([tap.out[i]["q"] for i in L]))                       # [4, D]
    out["h_end"] = move(torch.stack([tap.out[i]["end"] for i in L], 1))              # [n, 4, D]
    out["h_mean"] = move(torch.stack([tap.out[i]["mean"] for i in L], 1))
    rows = {k: v[0] for k, v in CAPTURE["rows"].items()}
    out["att"] = move(attention_per_sentence(rows, spans).half())                     # [n, 128]
    out["n_tok"] = move(torch.tensor(ex["n_tok"], dtype=torch.int16, device=device))
    return out


def p_supported(answer_logits):
    """P(supported) from the two label log-probabilities: softmax over (Yes, No), take Yes.

    Input:  answer_logits (tensor or list of 2 floats).  Output: float in [0, 1].
    """
    return float(torch.as_tensor(answer_logits, dtype=torch.float).softmax(-1)[0])


# ------------------------------------------------------------------ self-test
def selftest(n_rows=10):
    """Checks on real rows (GPU). Prints each check and stops at the first failure.

    1. the hooks do not change the verdict (same P with plain "sdpa" attention and no hooks);
    2. h_end / h_mean equal the direct computation from output_hidden_states (layers below the last one: the last
       entry of hidden_states has the final norm applied);
    3. the captured attention row equals "eager" attention (short prompts).
    """
    from evidence import render
    from evidence.data import io
    model = load_model()
    rows = io.read_jsonl(config.SPLITS / "dev_evid_select.jsonl")[:n_rows]
    preamble = config.preamble_template()
    worst_p, worst_h = 0.0, 0.0
    for row in rows:
        ex = render.render_row(row, preamble)
        out = forward_sentences(model, ex)
        # 1. verdict without hooks and capture
        model.config._attn_implementation = "sdpa"
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "sdpa"
        ids = torch.tensor([ex["input_ids"]], device=model.device)
        with torch.no_grad():
            res = model(input_ids=ids, output_hidden_states=True)
        plain = res.logits[0, -1].float().log_softmax(-1)[ex["label_ids"]].cpu()
        worst_p = max(worst_p, abs(p_supported(plain) - p_supported(out["answer_logits"])))
        # 2. summaries against the full hidden states
        hs = res.hidden_states
        s = ex["sent_tok_spans"]
        for j, layer in enumerate(config.SENTENCE_LAYERS[:-1]):
            h = hs[layer + 1][0].float()
            end = torch.stack([h[b] for a, b in s]).cpu()
            mean = torch.stack([h[a:b + 1].mean(0) for a, b in s]).cpu()
            worst_h = max(worst_h, (end - out["h_end"][:, j].float()).abs().max().item() / end.abs().max().item(),
                          (mean - out["h_mean"][:, j].float()).abs().max().item() / mean.abs().max().item())
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "sdpa_capture"
        del res, hs
    print(f"1. verdict, hooks vs plain sdpa: max |dP| = {worst_p:.5f} (want < 0.005)")
    print(f"2. sentence summaries vs hidden_states: max relative diff = {worst_h:.5f} (want < 0.01, bf16)")
    assert worst_p < 0.005 and worst_h < 0.01

    # 3. attention row vs eager, on a short prompt
    row = min(rows, key=lambda r: len(r["document"]))
    ex = render.render_row(row, preamble)
    out = forward_sentences(model, ex)
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    with torch.no_grad():
        res = model(input_ids=torch.tensor([ex["input_ids"]], device=model.device), output_attentions=True)
    att = [a for a in res.attentions if a is not None]
    rows_eager = {layer: att[i][0, :, -1].float() for i, layer in enumerate(config.FULL_ATTENTION_LAYERS)}
    eager = attention_per_sentence(rows_eager, torch.tensor(ex["sent_tok_spans"], device=model.device)).cpu()
    diff = (eager - out["att"].float()).abs().max().item()
    corr = torch.corrcoef(torch.stack([eager.flatten(), out["att"].float().flatten()]))[0, 1].item()
    # eager attention keeps its weights in bf16, so small differences are expected
    print(f"3. attention per sentence vs eager: max diff = {diff:.5f} (want < 0.01), correlation {corr:.5f} (> 0.999)")
    assert len(att) == len(config.FULL_ATTENTION_LAYERS) and diff < 0.01 and corr > 0.999
    print("self-test passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--rows", type=int, default=10)
    a = p.parse_args()
    if a.selftest:
        selftest(a.rows)
