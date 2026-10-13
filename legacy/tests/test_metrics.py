"""Run with python test_metrics.py."""

from math import isclose

from metrics import document_cluster_bootstrap, metrics, tune_threshold


def check_metrics():
    # Source A is perfect; larger source B predicts only unsupported.
    rows = ([{"dataset": "A", "label": y} for y in [0, 1]]
            + [{"dataset": "B", "label": y} for y in [0, 1, 1, 1]])
    result = metrics(rows, [0.2, 0.8, 0.1, 0.1, 0.1, 0.1], 0.5)
    assert result["macro_balanced_accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.625
    assert result["accuracy"] == 0.5
    assert result["by_source"]["A"]["roc_auc"] == 1.0
    assert result["by_source"]["B"]["roc_auc"] == 0.5

    binary_rows = [{"dataset": "A", "label": y} for y in [0, 1]]
    assert metrics(binary_rows, [0.5, 0.5], 0.5)["supported_recall"] == 0.0
    assert tune_threshold(binary_rows, [0.2, 0.8])["threshold"] == 0.5
    assert tune_threshold(binary_rows, [0.7, 0.8])["threshold"] == 0.7

    # .49 and .51 tie and are equally near .5; the larger threshold wins.
    tie_rows = [{"dataset": "A", "label": y} for y in [1, 0, 1, 0]]
    tied = tune_threshold(tie_rows, [0.5, 0.51, 0.9, 0.1])
    assert tied["threshold"] == 0.51
    assert isclose(tied["macro_balanced_accuracy"], 0.75)

    # Whitespace variants are one document, containing both classes; every
    # resample must therefore be valid, with perfect primary/wrong baseline.
    clustered_rows = [{"dataset": "A", "label": 0, "doc": "One  document"},
                      {"dataset": "A", "label": 1, "doc": "One\ndocument"}]
    intervals = document_cluster_bootstrap(clustered_rows, [0.1, 0.9], [0.9, 0.1], 0.5, repeats=20)
    assert intervals == {"primary": [1.0, 1.0], "baseline": [0.0, 0.0],
                         "paired_delta": [1.0, 1.0], "valid_repeats": 20, "repeats": 20}

    one_class = [{"dataset": "A", "label": 1}]
    assert metrics(one_class, [0.8])["macro_balanced_accuracy"] is None
    for invalid_rows, invalid_scores in [(rows, [0.1]), (binary_rows, [0.2, float("nan")]), ([], [])]:
        try:
            metrics(invalid_rows, invalid_scores)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scoring input accepted")
    try:
        tune_threshold(one_class, [0.8])
    except ValueError:
        pass
    else:
        raise AssertionError("Threshold tuning accepted a missing class")


if __name__ == "__main__":
    check_metrics()
    print("Metric checks passed")
