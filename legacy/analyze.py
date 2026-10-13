#!/usr/bin/env python3
"""Development selection and frozen, aggregate-only test reporting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench import ROOT, MODEL, PRICE, read_json, write_json, read_rows, candidate_pack, digest, verify_freeze, make_state, extract_scores
from metrics import metrics, tune_threshold, document_cluster_bootstrap


def joined(phase, paths):
    rows = read_rows(ROOT / f'data/{phase}.jsonl')
    row_map = {r['id']: r for r in rows}
    predictions = {r['id']: {} for r in rows}
    pack_paths = list(ROOT.glob('question*.json')) + list((ROOT / 'results').glob('*_pack.json'))
    packs = {digest(read_json(p)): read_json(p) for p in pack_paths}
    for path in paths:
        for r in read_rows(path):
            if r['id'] in predictions:
                if r['model'] != MODEL or r['input_hash'] != digest(make_state(row_map[r['id']])) or r['pack_hash'] not in packs:
                    raise RuntimeError('Stale or unknown prediction cache')
                if extract_scores(r['answers'], packs[r['pack_hash']]) != r['scores']:
                    raise RuntimeError('Cached scores disagree with answers')
                if any(k in predictions[r['id']] and predictions[r['id']][k] != v for k, v in r['scores'].items()):
                    raise RuntimeError('Conflicting duplicate scores')
                predictions[r['id']].update(r['scores'])
    if any(not v for v in predictions.values()):
        raise ValueError('Missing predictions; finish inference before analyzing')
    return rows, predictions


def tune(pack_path, paths, output):
    pack = read_json(pack_path)
    rows, predictions = joined('tune', paths)
    candidates = [k for k in list(pack['extractions']) + list(pack.get('combinations', {}))
                  if k not in pack.get('selection', {}).get('diagnostic_only', [])]
    results = []
    for name in candidates:
        scores = [predictions[r['id']][name] for r in rows]
        tuned = tune_threshold(rows, scores)
        results.append({'candidate': name, 'questions': len(candidate_pack(pack, [name])['questions']),
                        **tuned, 'at_half': metrics(rows, scores, .5)})
    results.sort(key=lambda x: (-x['macro_balanced_accuracy'], x['questions'], x['candidate']))
    write_json(output, results)
    for r in results:
        print(f"{r['candidate']:28s} threshold={r['threshold']:.2f} macro_BA={r['macro_balanced_accuracy']:.4f} at_.5={r['at_half']['macro_balanced_accuracy']:.4f}")


def shortlist(pack_path, ranking_path):
    path = ROOT / 'results/shortlist.json'
    if path.exists():
        raise RuntimeError('Shortlist already fixed')
    ranking = read_json(ranking_path)[:3]
    pack = read_json(pack_path)
    candidates = [r['candidate'] for r in ranking]
    short = {'created_at': datetime.now(timezone.utc).isoformat(), 'ranking': ranking,
             'pack': candidate_pack(pack, candidates + ['support_simple'])}
    write_json(path, short)
    write_json(ROOT / 'results/selection_pack.json', short['pack'])
    print('Locked candidates:', ', '.join(candidates))


def freeze(tune_paths):
    target = ROOT / 'results/frozen.json'
    if target.exists() or (ROOT / 'data/test.parquet').exists():
        raise RuntimeError('Freeze already exists or test already downloaded')
    short = read_json(ROOT / 'results/shortlist.json')
    rows, predictions = joined('selection', [ROOT / 'results/selection_predictions.jsonl'])
    results = []
    for trial in short['ranking']:
        name = trial['candidate']
        scores = [predictions[r['id']][name] for r in rows]
        results.append({'candidate': name, 'threshold': trial['threshold'], 'questions': trial['questions'],
                        'tune_macro_balanced_accuracy': trial['macro_balanced_accuracy'],
                        **metrics(rows, scores, trial['threshold'])})
    results.sort(key=lambda x: (-x['macro_balanced_accuracy'], x['questions'], -x['tune_macro_balanced_accuracy'], x['candidate']))
    winner = results[0]
    name = winner['candidate']
    tune_rows, tune_predictions = joined('tune', tune_paths)
    combined = tune_rows + rows
    all_predictions = {**tune_predictions, **predictions}
    fitted = tune_threshold(combined, [all_predictions[r['id']][name] for r in combined])
    baseline_fitted = tune_threshold(combined, [all_predictions[r['id']]['support_simple'] for r in combined])
    pack = candidate_pack(short['pack'], [name, 'support_simple'])
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'model': MODEL,
              'dataset': 'lytang/LLM-AggreFact', 'revision': read_json(ROOT / 'data/hub_metadata.json')['sha'],
              'candidate': name, 'threshold': fitted['threshold'], 'rule': 'score > threshold', 'pack': pack,
              'preprocessing': 'document <=80000 chars unchanged; otherwise first60000+omission marker+last20000; entire claim',
              'baseline': {'candidate': 'support_simple', 'threshold': .5,
                           'dev_tuned_threshold': baseline_fitted['threshold']},
              'selection_results': results, 'combined_dev_fit': fitted,
              'code_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in ['bench.py', 'metrics.py', 'analyze.py', 'fetch_data.py']},
              'data_hashes': read_json(ROOT / 'results/provenance.json')['development_files'],
              'test_accessed_before_freeze': False}
    write_json(ROOT / 'results/test_pack.json', pack)
    with target.open('x') as handle:
        json.dump(frozen, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'candidate': name, 'selection_macro_BA': winner['macro_balanced_accuracy'],
                      'tune_threshold': winner['threshold'], 'final_threshold': fitted['threshold'],
                      'frozen_at': frozen['frozen_at']}, indent=2))


def test_report():
    frozen = verify_freeze()
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / 'data/test.parquet').to_pylist()
    predictions = {p['id']: p for p in read_rows(ROOT / 'results/test_predictions.jsonl')}
    if set(predictions) != {f'test:{i}' for i in range(len(rows))}:
        raise RuntimeError('Test incomplete; no headline metric until full coverage')
    for i, row in enumerate(rows):
        p = predictions[f'test:{i}']
        if p['model'] != frozen['model'] or p['pack_hash'] != digest(frozen['pack']) or p['input_hash'] != digest(make_state(row)):
            raise RuntimeError('Test prediction provenance mismatch')
        if extract_scores(p['answers'], frozen['pack']) != p['scores']:
            raise RuntimeError('Test cached scores disagree with answers')
    candidate = frozen['candidate']
    scores = [predictions[f'test:{i}']['scores'][candidate] for i in range(len(rows))]
    baseline = [predictions[f'test:{i}']['scores']['support_simple'] for i in range(len(rows))]
    result = {'reported_at': datetime.now(timezone.utc).isoformat(), 'frozen_at': frozen['frozen_at'],
              'candidate': candidate, 'threshold': frozen['threshold'],
              'selected': metrics(rows, scores, frozen['threshold']), 'baseline': metrics(rows, baseline, .5),
              'calibrated_baseline': metrics(rows, baseline, frozen['baseline']['dev_tuned_threshold']),
              'truncated_documents': sum(p.get('document_truncated', False) for p in predictions.values()),
              'paced_latency_seconds': {str(q): float(np.percentile([p['seconds'] for p in predictions.values()], q)) for q in [50, 90, 95, 99]},
              'http_latency_seconds': {str(q): float(np.percentile([p['request_seconds'] for p in predictions.values()], q)) for q in [50, 90, 95, 99]},
              'usage': {k: sum(p['usage'].get(k, 0) for p in predictions.values()) for k in ['input_tokens', 'output_tokens']},
              'retries': sum(p['attempts'] - 1 for p in predictions.values())}
    result['estimated_input_cost_usd'] = result['usage']['input_tokens'] * PRICE
    result['document_cluster_bootstrap_95ci'] = document_cluster_bootstrap(rows, scores, baseline, frozen['threshold'])
    result['versus_calibrated_baseline_95ci'] = document_cluster_bootstrap(
        rows, scores, baseline, frozen['threshold'], frozen['baseline']['dev_tuned_threshold'])
    development_pairs = {digest([r['doc'], r['claim']]) for phase in ['tune', 'selection']
                         for r in read_rows(ROOT / f'data/{phase}.jsonl')}
    result['exact_pairs_also_in_observed_development'] = sum(digest([r['doc'], r['claim']]) in development_pairs for r in rows)
    write_json(ROOT / 'results/test_metrics.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['tune', 'shortlist', 'freeze', 'test-report'])
    parser.add_argument('--pack', default='question_candidates.json')
    parser.add_argument('--predictions', nargs='+', default=['results/tune_v1_predictions.jsonl'])
    parser.add_argument('--output', default='results/tune_v1_metrics.json')
    args = parser.parse_args()
    if args.command == 'tune':
        tune(args.pack, args.predictions, args.output)
    elif args.command == 'shortlist':
        shortlist(args.pack, args.output)
    elif args.command == 'freeze':
        freeze(args.predictions)
    else:
        test_report()
