#!/usr/bin/env python3
"""Fetch a single pinned split; test access requires the frozen protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import os

import requests

from bench import ROOT, read_json, write_json, verify_freeze


def fetch(split):
    revision = read_json(ROOT / 'data/hub_metadata.json')['sha']
    if split == 'test':
        frozen = verify_freeze()
        if frozen['revision'] != revision:
            raise RuntimeError('Dataset revision mismatch')
    path = ROOT / f'data/{split}.parquet'
    if path.exists():
        print('Split already downloaded:', split)
        return
    if split == 'test':
        write_json(ROOT / 'results/test_access.json', {
            'download_started_at': datetime.now(timezone.utc).isoformat(),
            'frozen_at': frozen['frozen_at'], 'revision': revision,
            'frozen_config_sha256': hashlib.sha256((ROOT / 'results/frozen.json').read_bytes()).hexdigest()})
    url = f'https://huggingface.co/datasets/lytang/LLM-AggreFact/resolve/{revision}/data/{split}-00000-of-00001.parquet'
    headers = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=(30, 180))
        if response.status_code != 200:
            raise RuntimeError(f'Hugging Face HTTP {response.status_code}')
        tmp = path.with_suffix('.part')
        with tmp.open('wb') as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        tmp.replace(path)
    except requests.RequestException:
        raise RuntimeError('Dataset download connection failed; retry command') from None
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(ROOT / f'results/{split}_download.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(), 'revision': revision,
        'sha256': fingerprint, 'bytes': path.stat().st_size})
    print(f'Downloaded {split}: {path.stat().st_size} bytes; sha256={fingerprint}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['dev', 'test'])
    fetch(parser.parse_args().split)
