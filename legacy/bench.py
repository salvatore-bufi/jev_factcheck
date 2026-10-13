#!/usr/bin/env python3
"""Small, resumable Jev experiment. Never loads test data without a verified freeze."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import requests

ROOT = Path(__file__).resolve().parent
MODEL = 'jev-1.13.0'
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
PRICE = 0.042 / 1_000_000


def make_state(row):
    document = row['doc']
    if len(document) > 80000:
        document = document[:60000] + '\n[Middle of long document omitted]\n' + document[-20000:]
    return {'document': document, 'claim': row['claim']}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def prepare():
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / 'data/dev.parquet').to_pylist()
    pools = {'tune': {}, 'selection': {}}
    for i, row in enumerate(rows):
        row['id'] = f'dev:{i}'
        # The same normalized document cannot cross internal development pools.
        group = digest(' '.join(row['doc'].split()))
        phase = 'tune' if int(group[:8], 16) % 10 < 6 else 'selection'
        row['group'] = group
        row.pop('contamination_identifier', None)
        pools[phase].setdefault((row['dataset'], row['label']), []).append(row)
    counts = {}
    for phase, buckets in pools.items():
        selected = []
        for key, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda row: digest(['jev-check-20260928', row['id']]))
            selected.extend(bucket[:110])
        selected.sort(key=lambda row: row['id'])
        write_rows(ROOT / f'data/{phase}.jsonl', selected)
        counts[phase] = {f'{key[0]}:{key[1]}': min(110, len(v)) for key, v in sorted(buckets.items())}
    assert not ({r['group'] for r in read_rows(ROOT / 'data/tune.jsonl')} &
                {r['group'] for r in read_rows(ROOT / 'data/selection.jsonl')})
    write_json(ROOT / 'data/development_counts.json', counts)
    print(json.dumps(counts, indent=2))


def extract_scores(answers, pack):
    scores = {}
    for name, spec in pack['extractions'].items():
        answer = answers[spec['answer_key']]
        value = answer[spec['field']]
        if 'option' in spec:
            value = value[spec['option']]
        value = float(value) / spec.get('divisor', 1)
        if spec.get('invert'):
            value = 1 - value
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid probability')
        scores[name] = value
    for name, spec in pack.get('combinations', {}).items():
        values = [scores[k] for k in spec['inputs']]
        scores[name] = {'mean': lambda: sum(values) / len(values),
                        'min': lambda: min(values), 'product': lambda: math.prod(values)}[spec['operation']]()
    return scores


def candidate_pack(pack, candidates):
    needed = set(candidates)
    for name in list(needed):
        if name in pack.get('combinations', {}):
            needed.update(pack['combinations'][name]['inputs'])
    extractions = {k: v for k, v in pack['extractions'].items() if k in needed}
    question_ids = {v['answer_key'] for v in extractions.values()}
    return {'questions': {k: v for k, v in pack['questions'].items() if k in question_ids},
            'extractions': extractions,
            'combinations': {k: v for k, v in pack.get('combinations', {}).items() if k in needed}}


def load_key():
    if os.environ.get('JEV_API_KEY'):
        return os.environ['JEV_API_KEY']
    for path in [ROOT / '.env', ROOT.parent / 'uncertainty/.env']:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.startswith('JEV_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('JEV_API_KEY not configured')


class Client:
    def __init__(self, rps=16):
        self.key = load_key()
        self.lock = threading.Lock()
        self.next_start = 0
        self.rps = rps
        self.reserved_cost = 0
        self.spent = sum(row.get('usage', {}).get('input_tokens', 0) * PRICE
                         for p in (ROOT / 'results').glob('*predictions.jsonl') for row in read_rows(p))
        self.local = threading.local()

    def ask(self, row, pack):
        state = make_state(row)
        payload = {'model': MODEL, 'state': state, 'questions': pack['questions']}
        # UTF-8 bytes conservatively bound ordinary text token count; reserve retries too.
        reserve = (len(json.dumps(payload, ensure_ascii=False).encode()) + 4096) * PRICE * 5
        with self.lock:
            if self.spent + self.reserved_cost + reserve > 10:
                raise RuntimeError('Experiment cost ceiling reached')
            self.reserved_cost += reserve
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        started = time.monotonic()
        request_seconds = 0.0
        try:
            for attempt in range(5):
                with self.lock:
                    delay = max(0, self.next_start - time.monotonic())
                    self.next_start = max(time.monotonic(), self.next_start) + 1 / self.rps
                time.sleep(delay)
                request_started = time.monotonic()
                try:
                    response = self.local.session.post(ENDPOINT, headers={
                        'Authorization': f'Bearer {self.key}'}, json=payload, timeout=(15, 90), allow_redirects=False)
                except requests.RequestException:
                    request_seconds += time.monotonic() - request_started
                    if attempt == 4:
                        raise RuntimeError('Jev connection failed after five attempts') from None
                    time.sleep(2 ** attempt)
                    continue
                request_seconds += time.monotonic() - request_started
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 4:
                        raise RuntimeError(f'Jev HTTP {response.status_code} after five attempts')
                    wait = response.headers.get('Retry-After', '')
                    time.sleep(min(60, max(2 ** attempt, float(wait) if wait.replace('.', '', 1).isdigit() else 0)))
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f'Jev HTTP {response.status_code}')
                try:
                    body = response.json()
                    if body['model'] != MODEL:
                        raise ValueError('Unexpected model')
                    for name, question in pack['questions'].items():
                        if body['answers'][name]['type'] != question['type']:
                            raise ValueError('Unexpected answer type')
                    scores = extract_scores(body['answers'], pack)
                    tokens = body['usage']['input_tokens']
                    if not isinstance(tokens, int) or tokens < 0:
                        raise ValueError('Invalid usage')
                except (KeyError, TypeError, ValueError):
                    raise RuntimeError('Invalid Jev response schema') from None
                with self.lock:
                    self.spent += tokens * PRICE
                return {'id': row['id'], 'input_hash': digest(state), 'pack_hash': digest(pack),
                        'model': body['model'], 'answers': body['answers'], 'scores': scores,
                        'usage': body['usage'], 'seconds': time.monotonic() - started,
                        'attempts': attempt + 1, 'document_truncated': len(row['doc']) > 80000,
                        'request_seconds': request_seconds}
        finally:
            with self.lock:
                self.reserved_cost -= reserve


def verify_freeze():
    frozen = read_json(ROOT / 'results/frozen.json')
    for filename, expected in frozen['code_hashes'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen file changed: {filename}')
    return frozen


def run(phase, pack_path, output, limit=None, workers=16, rps=16):
    if phase == 'selection':
        short = read_json(ROOT / 'results/shortlist.json')
        if digest(read_json(pack_path)) != digest(short['pack']):
            raise RuntimeError('Selection pack differs from locked shortlist')
    if phase == 'test':
        frozen = verify_freeze()
        if digest(read_json(pack_path)) != digest(frozen['pack']):
            raise RuntimeError('Test pack differs from frozen configuration')
        import pyarrow.parquet as pq
        rows = pq.read_table(ROOT / 'data/test.parquet').to_pylist()
        for i, row in enumerate(rows):
            row['id'] = f'test:{i}'
    else:
        rows = read_rows(ROOT / f'data/{phase}.jsonl')
    if limit:
        rows = rows[:limit]
    pack = read_json(pack_path)
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    cached = {r['id']: r for r in read_rows(output)} if output.exists() else {}
    if any(r['pack_hash'] != digest(pack) or r['model'] != MODEL for r in cached.values()):
        raise RuntimeError('Cache pack/model mismatch')
    for row in rows:
        old = cached.get(row['id'])
        if old and old['input_hash'] != digest(make_state(row)):
            raise RuntimeError('Cache input mismatch')
    todo = [row for row in rows if row['id'] not in cached]
    client = Client(rps)
    print(json.dumps({'phase': phase, 'rows': len(rows), 'cached': len(cached), 'pending': len(todo),
                      'questions': list(pack['questions'])}), flush=True)
    failures = 0
    started = time.monotonic()
    with output.open('a') as handle, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(todo)
        pending = {}
        for _ in range(workers):
            row = next(iterator, None)
            if row is not None:
                pending[pool.submit(client.ask, row, pack)] = row
        while pending:
            completed, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                row = pending.pop(future)
                try:
                    result = future.result()
                except RuntimeError as error:
                    failures += 1
                    with output.with_name(output.stem.replace('predictions', 'errors') + '.jsonl').open('a') as errors:
                        errors.write(json.dumps({'id': row['id'], 'error': str(error)}) + '\n')
                    print(json.dumps({'error': str(error), 'failures': failures}), flush=True)
                    if failures >= 10:
                        for task in pending:
                            task.cancel()
                        raise RuntimeError('Stopped after ten failed records; successful responses remain cached')
                else:
                    handle.write(json.dumps(result) + '\n')
                    handle.flush()
                    cached[row['id']] = result
                    if len(cached) % (1000 if phase == 'test' else 100) == 0 or len(cached) == len(rows):
                        print(json.dumps({'completed': len(cached), 'total': len(rows),
                                          'seconds': round(time.monotonic() - started, 1),
                                          'estimated_total_cost_usd': round(client.spent, 5)}), flush=True)
                row = next(iterator, None)
                if row is not None:
                    pending[pool.submit(client.ask, row, pack)] = row
    if failures:
        raise RuntimeError(f'{failures} records failed; rerun to resume')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--phase', choices=['tune', 'selection', 'test'], default='tune')
    parser.add_argument('--pack', default='question_candidates.json')
    parser.add_argument('--output', default='results/tune_v1_predictions.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--rps', type=float, default=16)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        run(args.phase, args.pack, args.output, args.limit, args.workers, args.rps)
