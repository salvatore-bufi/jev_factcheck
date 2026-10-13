"""How a dataset row becomes the state every model sees (identical for all models)."""

DEFAULT_MAX_DOC_CHARS = 80000


def state_for(row, cap=DEFAULT_MAX_DOC_CHARS):
    """{"document", "claim"}; a document over `cap` characters keeps its first 75% and last 25%."""
    document = row["doc"]
    if cap and len(document) > cap:
        head = 3 * cap // 4
        document = document[:head] + "\n[Middle of long document omitted]\n" + document[-(cap - head):]
    return {"document": document, "claim": row["claim"]}


def truncated(row, cap=DEFAULT_MAX_DOC_CHARS):
    return bool(cap) and len(row["doc"]) > cap
