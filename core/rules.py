"""Decision rules: from the stored per-question answers to one score and one prediction per row.

An answer is stored as {"type", "keys", "probs": {key: p}, "logits": [...] or null, "temperature": T}:
`logits` are the raw (untempered) option logits in `keys` order when the model exposes them, and
`probs` = softmax(logits / T) as the model's own default produces them. A rule picks
values from answers and combines them:

    {"combine": "min" | "mean" | "product" | "single",
     "inputs": [{"question": "q1", "take": "yes"}, {"question": "q2", "take": "no"}, ...],
     "threshold": 0.30}

`take`: "yes" (P(true) of a noul), "no" (1 - P(true)), "option:KEY" (a choice/score option's
probability), "not_option:KEY", or "score" (a score question's expected level scaled to 0..1).
The prediction is "supported" when the combined value is strictly above the threshold.
"""
import math

COMBINE = {
    "min": min,
    "mean": lambda xs: sum(xs) / len(xs),
    "product": lambda xs: math.prod(xs),
    "single": lambda xs: xs[0],
}


def softmax(logits, temperature=1.0):
    peak = max(logits)
    weights = [math.exp((x - peak) / temperature) if math.isfinite(x) else 0.0 for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def probs_of(answer, temperature=None):
    """The answer's option probabilities, re-tempered from its logits when a temperature is given."""
    if temperature is not None:
        if not answer.get("logits"):
            raise ValueError("this model stores no logits, so its answers cannot be re-tempered")
        return dict(zip(answer["keys"], softmax(answer["logits"], temperature)))
    return answer["probs"]


def take(answer, how, temperature=None):
    probs = probs_of(answer, temperature)
    if how == "yes":
        return probs["true"]
    if how == "no":
        return 1.0 - probs["true"]
    if how.startswith("option:"):
        return probs[how.split(":", 1)[1]]
    if how.startswith("not_option:"):
        return 1.0 - probs[how.split(":", 1)[1]]
    if how == "score":
        keys = answer["keys"]
        return sum(int(k) * probs[k] for k in keys) / (len(keys) - 1)
    raise ValueError(f"unknown take {how!r}")


def score(answers, rule, temperature=None):
    values = [take(answers[i["question"]], i["take"], temperature) for i in rule["inputs"]]
    return COMBINE[rule["combine"]](values)


def single_question(question, take_how="yes", threshold=0.5):
    return {"combine": "single", "inputs": [{"question": question, "take": take_how}], "threshold": threshold}
