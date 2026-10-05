#!/usr/bin/env python3
"""Convert Winnow-12B-BF16.gguf into a standard safetensors checkpoint that vLLM can load.

vLLM 0.28 has no GGUF loader, and transformers cannot parse this GGUF's config. But the BF16 GGUF
holds only BF16 and F32 tensors, so it is a lossless copy of the model: llama.cpp's Gemma 4
converter only renames tensors (Gemma 4 has no norm shift and no permutation), so the inverse
is a rename too. This script:

  1. reads every tensor of the GGUF (memory-mapped, one shard at a time),
  2. renames it to the Hugging Face / vLLM Gemma 4 name (table below; layer norms F32 -> bf16 is exact),
  3. writes bf16 safetensors shards (<= 4 GB) plus model.safetensors.index.json,
  4. writes a text-only gemma4_text config.json and copies the tokenizer and chat template.

It refuses to finish unless every GGUF tensor is accounted for (except rope_freqs, which is derived
from the config) and every expected tensor of the architecture is present with the right shape.
Vision and audio towers are not part of the language GGUF (they are in mmproj-*.gguf) and are not used.

    python winnow-12b-bf16/convert.py            # about 3 minutes, writes winnow-12b-bf16/model/ (24 GB)
"""
import argparse
import glob
import json
from pathlib import Path
import re
import shutil

import gguf
import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SNAPSHOT_GLOB = str(Path.home() / ".cache/huggingface/hub/models--EldanRing--Winnow-12B/snapshots/*/")

# GGUF block-tensor suffix -> Hugging Face name inside model.layers.N.
LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "layer_output_scale.weight": "layer_scalar",   # a buffer in HF/vLLM, stored as a 1-element tensor
}
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}
SKIPPED = {"rope_freqs.weight"}   # llama.cpp-only helper for proportional RoPE; derived from the config


def hf_name(gguf_name):
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if match and match.group(2) in LAYER_MAP:
        return f"model.layers.{match.group(1)}.{LAYER_MAP[match.group(2)]}"
    return None


def to_tensor(t):
    """GGUF tensor -> bf16 torch tensor in the usual [out, in] layout (GGUF stores dims reversed)."""
    shape = tuple(int(x) for x in t.shape[::-1])
    data = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(data).view(np.uint16).reshape(shape)
        return torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16)   # bit-exact
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return torch.from_numpy(np.ascontiguousarray(data).reshape(shape)).to(torch.bfloat16)
    raise SystemExit(f"{t.name}: unexpected type {t.tensor_type.name}; this converter only handles BF16 and F32")


def expected_shapes(cfg):
    """Shapes the vLLM Gemma 4 text model needs, from the config, to verify the conversion."""
    h, n = cfg["hidden_size"], cfg["num_hidden_layers"]
    heads, ff = cfg["num_attention_heads"], cfg["intermediate_size"]
    shapes = {"model.embed_tokens.weight": (cfg["vocab_size"], h), "model.norm.weight": (h,)}
    for i in range(n):
        full = cfg["layer_types"][i] == "full_attention"
        hd = cfg["global_head_dim"] if full else cfg["head_dim"]
        kv = cfg["num_global_key_value_heads"] if full else cfg["num_key_value_heads"]
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
            p + "pre_feedforward_layernorm.weight": (h,), p + "post_feedforward_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (heads * hd, h), p + "self_attn.k_proj.weight": (kv * hd, h),
            p + "self_attn.o_proj.weight": (h, heads * hd),
            p + "self_attn.q_norm.weight": (hd,), p + "self_attn.k_norm.weight": (hd,),
            p + "mlp.gate_proj.weight": (ff, h), p + "mlp.up_proj.weight": (ff, h), p + "mlp.down_proj.weight": (h, ff),
            p + "layer_scalar": (1,),
        })
        if not (full and cfg["attention_k_eq_v"]):   # global layers share K and V: no v_proj
            shapes[p + "self_attn.v_proj.weight"] = (kv * hd, h)
    return shapes


def text_config(repo_config):
    """Plain gemma4_text config for vLLM, from the repo's 'unified' multimodal config."""
    t = dict(repo_config["text_config"])
    per_layer = t.pop("per_layer_config", {})
    full = {int(i): c for i, c in per_layer.items()}
    heads = {(c["head_dim"], c["num_key_value_heads"]) for c in full.values()}
    assert len(heads) == 1, "global layers are expected to share one head size and KV head count"
    (global_head_dim, global_kv), = heads
    for key in ("use_bidirectional_attention", "vocab_size_per_layer_input", "moe_intermediate_size",
                "num_experts", "top_k_experts"):
        t.pop(key, None)
    t.update({"model_type": "gemma4_text", "global_head_dim": global_head_dim,
              "num_global_key_value_heads": global_kv, "hidden_size_per_layer_input": 0})
    t["architectures"] = ["Gemma4ForCausalLM"]
    t["eos_token_id"] = repo_config.get("eos_token_id", t.get("eos_token_id"))
    t["dtype"] = "bfloat16"
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", default=None, help="Hugging Face snapshot directory of EldanRing/Winnow-12B")
    p.add_argument("--out", default=str(HERE / "model"))
    p.add_argument("--shard-gb", type=float, default=4.0)
    args = p.parse_args()

    snapshot = Path(args.snapshot or sorted(glob.glob(SNAPSHOT_GLOB))[0])
    source = snapshot / "gguf/Winnow-12B-BF16.gguf"
    out = Path(args.out)
    if (out / "model.safetensors.index.json").exists():
        raise SystemExit(f"{out} already holds a converted model; delete it to convert again")
    repo_config = json.loads((snapshot / "config.json").read_text())
    cfg = text_config(repo_config)

    reader = gguf.GGUFReader(str(source))
    arch = reader.fields["general.architecture"].contents()
    assert arch == "gemma4", f"unexpected architecture {arch}"
    rows = int([t for t in reader.tensors if t.name == "token_embd.weight"][0].shape[1])
    if rows != cfg["vocab_size"]:
        print(f"vocab_size {cfg['vocab_size']} -> {rows} (rows in token_embd)")
        cfg["vocab_size"] = rows
    expected = expected_shapes(cfg)

    out.mkdir(parents=True, exist_ok=True)
    shard, shard_bytes, shards, index, seen = {}, 0, [], {}, set()

    def flush():
        nonlocal shard, shard_bytes
        if shard:
            shards.append(shard)
            shard, shard_bytes = {}, 0

    plan = []
    for t in reader.tensors:
        if t.name in SKIPPED:
            continue
        name = hf_name(t.name)
        if name is None:
            raise SystemExit(f"No mapping for GGUF tensor {t.name}")
        shape = tuple(int(x) for x in t.shape[::-1])
        if name not in expected:
            raise SystemExit(f"{t.name} -> {name} is not a tensor of this architecture")
        if expected[name] != shape:
            raise SystemExit(f"{t.name} -> {name}: shape {shape}, expected {expected[name]}")
        plan.append((t, name))
        seen.add(name)
    missing = sorted(set(expected) - seen)
    if missing:
        raise SystemExit(f"Missing tensors after mapping: {missing[:10]} ({len(missing)} total)")
    print(f"mapping verified: {len(plan)} tensors, none missing, none unexpected")

    # Write shards one at a time so only a few GB are in RAM.
    groups, current, size = [], [], 0
    for t, name in plan:
        nbytes = int(np.prod([int(x) for x in t.shape])) * 2
        if current and size + nbytes > args.shard_gb * 1e9:
            groups.append(current)
            current, size = [], 0
        current.append((t, name))
        size += nbytes
    groups.append(current)
    total = 0
    for i, group in enumerate(groups, 1):
        filename = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        tensors = {name: to_tensor(t).contiguous() for t, name in group}
        save_file(tensors, str(out / filename), metadata={"format": "pt"})
        for name, tensor in tensors.items():
            index[name] = filename
            total += tensor.numel() * 2
        print(f"wrote {filename}: {len(tensors)} tensors", flush=True)
        del tensors
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        src = snapshot / name
        if src.exists():
            shutil.copyfile(src.resolve(), out / name)   # resolve the Hugging Face cache symlink
    print(f"done: {total / 1e9:.1f} GB in {len(groups)} shards at {out}")


if __name__ == "__main__":
    main()
