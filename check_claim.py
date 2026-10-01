#!/usr/bin/env python3
"""Check a claim against a local document using the frozen Jev configuration."""
import argparse
import json
from pathlib import Path

from bench import Client, verify_freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, required=True)
    claim = parser.add_mutually_exclusive_group(required=True)
    claim.add_argument('--claim')
    claim.add_argument('--claim-file', type=Path)
    args = parser.parse_args(argv)
    frozen = verify_freeze()
    text = args.claim if args.claim is not None else args.claim_file.read_text(encoding='utf-8')
    row = {'id': 'local:0', 'doc': args.document.read_text(encoding='utf-8'), 'claim': text}
    if not row['doc'].strip() or not text.strip():
        parser.error('Document and claim must both contain text')
    result = Client().ask(row, frozen['pack'])
    candidate, threshold = frozen['candidate'], frozen['threshold']
    score = result['scores'][candidate]
    inputs = frozen['pack'].get('combinations', {}).get(candidate, {}).get('inputs', [candidate])
    print(json.dumps({
        'supported': score > threshold,
        'score': score,
        'threshold': threshold,
        'components': {name: result['scores'][name] for name in inputs},
        'model': result['model'],
        'usage': result['usage'],
        'document_truncated': result['document_truncated'],
    }, indent=2))


if __name__ == '__main__':
    main()
