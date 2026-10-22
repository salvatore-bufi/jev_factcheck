I'd fine-tune it with LoRA on your fact-checking data, keeping the model's exact prompt format and training only on the answer letter. The other options are cheaper variants of this or different model families. I'm answering from the JPT-9B model card and general practice, and I haven't trained anything here.

**1. Decide what the data is allowed to do**
- If your dataset is LLM-AggreFact, do not train on it. Your repo's `benchmark_notes.md` records that its README permits evaluation and excludes pretraining or fine-tuning use. It also has no train split, and training on dev while reporting on test would make your numbers meaningless.
- If it's your own data, split it by document, not by row. Otherwise the same document appears in train and validation and inflates the scores.
- Check for overlap between your training documents and the LLM-AggreFact test documents before you start.
- JPT-9B's weights are CC BY-NC 4.0, so anything fine-tuned from it is non-commercial too.

**2. How JPT-9B was trained, and what to copy**
- It is a Qwen3.5-9B with a merged LoRA (rank 16, all attention and MLP layers, learning rate 5e-5, one epoch over about 49k typed questions).
- The loss is multi-class Brier over the option labels, computed on the `llm2jev` chat prompt with thinking off.
- To continue it, keep that format exactly: the same system and user turns, the `Answer:` prefill, options A/B, and the loss computed only at the answer position over the option-letter logits. A different prompt would move the model off its learned behaviour.

**3. Options, cheapest first**

| Option | What you change | Cost | Typical gain |
|---|---|---|---|
| **A. Question tuning** | Rewrite the questions, the combination rule, the threshold on a held-out part of your data. No training. | Minutes | Small, but it's what the repo already does |
| **B. Calibration only** | Fit a threshold or a small logistic layer over the three output probabilities. | Minutes | Small and cheap. Your repo already tried a version of this and it didn't beat the frozen checker |
| **C. LoRA / QLoRA fine-tune** | Train adapters on your labelled (document, claim) pairs in JPT's format, then merge them. | Hours to a day | Usually the biggest gain on in-domain data |
| **D. Full fine-tune** | Update all weights. | Needs 8+ large GPUs | Rarely worth it over LoRA |
| **E. Smaller classifier** | Fine-tune an encoder like DeBERTa-v3-large, as FactCG and MiniCheck did. | Cheap | Needs good data, but a strong baseline for the cost. The published FactCG-DeBERTa-L gets 75.6 |

**4. How I would do option C**
1. **Format the data:** One row becomes one question: the state is `{"document", "claim"}`, and the question is the support question from `packs/claim_support.json`. The label is "true" or "false". Optionally add the unsupported-detail and contradiction questions as extra rows.
2. **Loss:** Compute logits at the answer position, restrict them to the option letters, and train with cross-entropy or Brier on those letters. This matches how the model is read at inference. Plain next-token SFT on the letter approximates it.
3. **Setup:** Use LoRA rank 16–32 on all projections, a learning rate around 1e-5 to 5e-5, one to two epochs, and gradient checkpointing. Cap training sequences at 4–8k tokens, since 20k-token documents are very costly. Chunk or truncate long ones, and say so when you report results.
4. **Avoid forgetting:** Mix in some general typed-decision data (choice, score, noul questions), or at least keep the learning rate low. The original JPT training data isn't public, so a small replay set of your own is the substitute.
5. **Balance the data:** Sample evenly per source and label, because the metric is per-source balanced accuracy.
6. **Tools:** Hugging Face PEFT with TRL is the simplest. JPT itself was trained with ms-swift (its `args.json` shows that). Axolotl, LLaMA-Factory and Unsloth also work. A custom loss at the answer position needs a small custom trainer in most of them.
7. **Evaluate:** Merge the adapter into the weights, point `jpt-9b/benchmark.py` at the merged folder, and run the benchmark once. Pick checkpoints and the threshold on a validation split of your own data only.

**5. Hardware**
- **Your 24 GB laptop GPU:** Full-precision 9B LoRA won't fit with long documents. QLoRA (4-bit base weights) can fit, with short sequences and small batches, and will be slow.
- **The 45 GB L40S:** Bf16 LoRA is realistic for sequences up to a few thousand tokens. As a rough guess, 50k examples of about 1.5k tokens each would take around a day. I haven't measured this.
- **8 large GPUs:** Full runs like the original, which used 8 GPUs, are comfortable there.

**6. Other things worth considering**
- **Synthetic training data:** If you lack labelled data, MiniCheck-style and FactCG-style synthetic data (claims generated from documents, with negatives made by corrupting a fact) was how published checkers were trained. It could extend a smaller dataset.
- **Don't overfit the benchmark:** If you iterate on LLM-AggreFact numbers, keep one split you look at once. Your repo's protocol is a good model for that.

If you tell me what the dataset contains, how many examples it has and its labels, I can suggest a concrete configuration.

**Memory**
- **Base weights:** JPT-9B in bf16 takes about 18–19 GB.
- **LoRA adapter:** The adapter and its optimizer states are small, around 1–2 GB at rank 16–32.
- **Activations:** With gradient checkpointing and batch size 1, sequences up to about 8k tokens should fit, with room to spare.
- **Longer sequences:** Documents of 16k tokens or more get risky. Truncate or chunk them for training.
- **Larger batches:** Reach them with gradient accumulation, not a bigger per-step batch.

**Speed**
- I'd expect roughly 1,000 tokens/s of training throughput per L40S. This is a rough estimate, not a measurement.
- That is about 5–6 hours for 20M training tokens, or about 20 hours for 75M. Count tokens, not rows: 50k examples of 1.5k tokens is 75M.

**Your second GPU**
- The readout says "Device 1", so the server seems to have at least two L40S cards.
- You can train with both using data parallelism, which would roughly halve the time. LoRA only exchanges small gradients, so PCIe without NVLink is fine for it.
- Each card holds its own full copy of the model, which fits.

**What doesn't fit**
- A full fine-tune of all 9B weights (about 100+ GB with optimizer states) needs many more GPUs.
- It also doesn't need to be on one L40S: QLoRA (4-bit base weights) would also fit in 45 GB, but it isn't needed there. It is for your 24 GB laptop GPU.

If you meant the CPU instead, a CPU can't train a 9B model in any practical time. I'd only use it for preparing the data.