# StartLux-Decision-v1.1-4B-Instant

[startlux-models/StartLux-Decision-v1.1-4B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant):
the 4B member of the StartLux Decision v1.1 Instant family (Qwen3.5-4B architecture with a vision tower). It answers
every question with one read-out of the option letters and never thinks; its full-attention layers read the prompt
bidirectionally. Licences: weights **CC BY-NC 4.0**; inference package
([StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision)) Apache-2.0.

## How the benchmark runs it

Exactly as [models/startlux-9b-instant](../startlux-9b-instant/README.md): the same package (`startlux-decide` at
commit `a99977d`, vLLM 0.31.0 with the bidirectional plugin), the same prompt and the same read-out. The adapter is a copy
of the 9B's with three differences:

- `REPO` and `revision` (`416088f`) of the 4B weights;
- default ports 8093 (decision server) and 8102 (vLLM), so a 9B and a 4B server can run at the same time;
- it also finds `startlux-decide` in `models/startlux-9b-instant/.venv`, so one environment serves both.

The read-out temperatures come from the 4B's own `decision_config.json`: noul 1.2662, choice 1.1103, score 1.1344.

What is stored: as for the 9B (`probs`; `logits` = `T * log(p)` of the read-out, so answers can be re-tempered;
`prompt_tokens`).

## Setup (once)

```sh
bash models/startlux-4b-instant/setup.sh      # or nothing, if models/startlux-9b-instant/setup.sh has run
```

## Run

```sh
hf download startlux-models/StartLux-Decision-v1.1-4B-Instant --revision 416088f77cfce09633906a3675c7cd2e5dabcf10
python benchmark.py --model startlux-4b-instant --gpu 0 --limit 32 --no-profile --tag smoke
```

Options: [config/models/startlux-4b-instant.json](../../config/models/startlux-4b-instant.json), same meaning as the 9B's.
About 9 GB of bf16 weights; a 24 GB GPU is enough.
