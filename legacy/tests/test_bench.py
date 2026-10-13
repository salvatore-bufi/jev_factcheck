"""Offline harness checks: python test_bench.py."""
from copy import deepcopy
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

import bench


def raises(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def check_extraction():
    pack = bench.read_json(Path(__file__).with_name('question_candidates.json'))
    answers = {
        'support_simple': {'noul': 0.8},
        'support_strict': {'noul': 0.7},
        'unsupported_detail': {'noul': 0.25},
        'support_binary': {'probabilities': {'supported': 0.6}},
        'support_ternary': {'probabilities': {'supported': 0.5}},
        'support_score': {'score': 2.1, 'probabilities': {'3': 0.4}},
        'contradicted_detail': {'noul': 0.1},
    }
    scores = bench.extract_scores(answers, pack)
    assert scores['no_unsupported_detail'] == 0.75
    assert scores['no_contradicted_detail'] == 0.9
    assert isclose(scores['support_score_normalized'], 0.7)
    assert scores['support_score_complete'] == 0.4
    assert isclose(scores['mean_support'], 0.6)
    assert scores['min_support_checks'] == 0.75
    assert isclose(scores['product_support_checks'], 0.54)
    for bad in [float('nan'), float('inf'), -0.01, 1.01]:
        invalid = deepcopy(answers)
        invalid['support_simple']['noul'] = bad
        raises(ValueError, bench.extract_scores, invalid, pack)

    subpack = bench.candidate_pack(pack, ['mean_support', 'support_simple'])
    assert set(subpack['questions']) == {
        'support_simple', 'support_strict', 'support_binary', 'support_ternary'}
    assert set(subpack['combinations']) == {'mean_support'}
    smaller = bench.extract_scores({key: answers[key] for key in subpack['questions']}, subpack)
    assert all(isclose(value, scores[key]) for key, value in smaller.items())


def check_client():
    pack = bench.candidate_pack(
        bench.read_json(Path(__file__).with_name('question_candidates.json')), ['support_simple'])
    row = {'id': 'synthetic:0', 'doc': 'A fictional document.', 'claim': 'A fictional claim.',
           'label': 1, 'dataset': 'excluded', 'contamination_identifier': 'excluded'}
    body = {'model': bench.MODEL, 'answers': {'support_simple': {'type': 'noul', 'noul': 0.75}},
            'usage': {'input_tokens': 100, 'output_tokens': 7}}

    def response(status=200, data=None):
        return Mock(status_code=status, headers={}, json=Mock(return_value=data or body))

    with TemporaryDirectory() as directory, \
            patch.object(bench, 'ROOT', Path(directory)), \
            patch.object(bench, 'load_key', return_value='synthetic-key'), \
            patch.object(bench.time, 'sleep'), \
            patch.object(bench.requests, 'Session') as session_factory:
        session = session_factory.return_value
        session.post.side_effect = [response(429), response(503), response()]
        client = bench.Client()
        result = client.ask(row, pack)
        assert result['attempts'] == 3
        assert result['scores'] == {'support_simple': 0.75}
        assert isclose(client.spent, 100 * bench.PRICE)
        assert client.reserved_cost == 0
        request = session.post.call_args.kwargs
        assert set(request['json']['state']) == {'document', 'claim'}
        assert request['allow_redirects'] is False
        assert session.post.call_args.args[0] == bench.ENDPOINT

        for failure, expected_calls in [(response(401), 1), (response(529), 5),
                                         (requests.Timeout('synthetic secret'), 5)]:
            session.post.reset_mock()
            session.post.side_effect = [failure] * expected_calls
            client = bench.Client()
            raises(RuntimeError, client.ask, row, pack)
            assert session.post.call_count == expected_calls
            assert client.reserved_cost == 0

        for mutation in [('model', 'unexpected-model'), ('answers', {}),
                         ('usage', {'input_tokens': -1})]:
            malformed = deepcopy(body)
            malformed[mutation[0]] = mutation[1]
            session.post.side_effect = [response(data=malformed)]
            client = bench.Client()
            raises(RuntimeError, client.ask, row, pack)
            assert client.reserved_cost == 0


def check_preprocessing():
    # These are synthetic strings, never benchmark examples.
    short = {'doc': 'a' * 80000, 'claim': 'x'}
    assert bench.make_state(short)['document'] == short['doc']
    long = {'doc': 'a' * 60000 + 'b' * 100 + 'c' * 20000, 'claim': 'x'}
    processed = bench.make_state(long)
    assert processed['document'].startswith('a' * 60000)
    assert processed['document'].endswith('c' * 20000)
    assert '[Middle of long document omitted]' in processed['document']
    assert 'b' not in processed['document']
    assert processed['claim'] == 'x'


if __name__ == '__main__':
    check_extraction()
    check_client()
    check_preprocessing()
    print('Harness checks passed; no network or benchmark data accessed')
