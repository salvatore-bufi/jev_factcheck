"""Fixed diagnostic sample for the original, unchanged checker; no tuning."""
import argparse
from datetime import datetime, timezone
import hashlib

import pyarrow.parquet as pq

from bench import ROOT, read_json, read_rows, write_json, write_rows, digest, verify_freeze, run
from metrics import metrics, document_cluster_bootstrap

SOURCES = ('Lfqa', 'RAGTruth')


def prepare():
    frozen = verify_freeze()
    if (ROOT / 'results/pilot_plan.json').exists():
        raise RuntimeError('Pilot sample already locked')
    rows = pq.read_table(ROOT / 'data/test.parquet').to_pylist()
    sampled = []
    for source in SOURCES:
        for label in [0, 1]:
            bucket = [dict(r, id=f'test:{i}') for i, r in enumerate(rows)
                      if r['dataset'] == source and r['label'] == label]
            bucket.sort(key=lambda r: digest(['pilot-20260928-fixed', r['id']]))
            sampled.extend(bucket[:200])
    sampled.sort(key=lambda r: int(r['id'].split(':')[1]))
    ids = {r['id'] for r in sampled}
    previous = [r for r in read_rows(ROOT / 'results/test_predictions.jsonl') if r['id'] in ids]
    write_rows(ROOT / 'data/pilot.jsonl', sampled)
    write_rows(ROOT / 'results/pilot_predictions.jsonl', [dict(r, reused=True) for r in previous])
    write_json(ROOT / 'results/pilot_plan.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'sources': SOURCES,
        'rows_per_source_label': 200, 'rows': len(sampled), 'reused_predictions': len(previous),
        'seed_rule': "sort sha256(['pilot-20260928-fixed', row_id]) within source/label",
        'purpose': 'Estimate two unfinished sources; no candidate, threshold or prompt changes allowed.',
        'checker': frozen['candidate'], 'threshold': frozen['threshold'],
        'frozen_sha256': hashlib.sha256((ROOT / 'results/frozen.json').read_bytes()).hexdigest(),
        'sample_sha256': hashlib.sha256((ROOT / 'data/pilot.jsonl').read_bytes()).hexdigest(),
        'sample_ids': sorted(ids), 'scope': 'Diagnostic sample; not an official full benchmark score.'})
    print(f'Locked {len(sampled)} rows; {len(previous)} predictions reused; {len(sampled)-len(previous)} API calls needed.')


def check_plan():
    frozen = verify_freeze()
    plan = read_json(ROOT / 'results/pilot_plan.json')
    assert plan['frozen_sha256'] == hashlib.sha256((ROOT / 'results/frozen.json').read_bytes()).hexdigest()
    assert plan['sample_sha256'] == hashlib.sha256((ROOT / 'data/pilot.jsonl').read_bytes()).hexdigest()
    return frozen


def report():
    frozen = check_plan()
    all_rows = pq.read_table(ROOT / 'data/test.parquet').to_pylist()
    original = read_rows(ROOT / 'results/test_predictions.jsonl')
    predictions = {r['id']: r for r in original if all_rows[int(r['id'].split(':')[1])]['dataset'] not in SOURCES}
    pilot = {r['id']: r for r in read_rows(ROOT / 'results/pilot_predictions.jsonl')}
    assert set(pilot) == set(read_json(ROOT / 'results/pilot_plan.json')['sample_ids'])
    predictions.update(pilot)
    ids = sorted(predictions, key=lambda x: int(x.split(':')[1]))
    rows = [all_rows[int(i.split(':')[1])] for i in ids]
    scores = [predictions[i]['scores'][frozen['candidate']] for i in ids]
    baseline = [predictions[i]['scores']['support_simple'] for i in ids]
    result = {'status': 'Nine fully evaluated sources plus two fixed stratified samples; estimated eleven-source score.',
              'n': len(rows), 'selected': metrics(rows, scores, frozen['threshold']),
              'baseline': metrics(rows, baseline, .5),
              'calibrated_baseline': metrics(rows, baseline, frozen['baseline']['dev_tuned_threshold']),
              'ci95': document_cluster_bootstrap(rows, scores, baseline, frozen['threshold']),
              'sampled_sources': list(SOURCES), 'no_tuning_on_pilot': True}
    write_json(ROOT / 'results/pilot_metrics.json', result)
    print({k: result[k]['macro_balanced_accuracy'] for k in ['selected', 'baseline', 'calibrated_baseline']})
    print('95% interval:', result['ci95']['primary'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'report'])
    command = p.parse_args().command
    if command == 'prepare':
        prepare()
    elif command == 'run':
        check_plan()
        run('pilot', ROOT / 'results/test_pack.json', ROOT / 'results/pilot_predictions.jsonl', workers=24, rps=18)
    else:
        report()
