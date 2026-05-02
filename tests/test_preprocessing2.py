import ast
import csv
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / 'preprocessing.ipynb'
if not NOTEBOOK_PATH.exists():
    pytest.skip(f'preprocessing.ipynb not found at {NOTEBOOK_PATH}', allow_module_level=True)

TARGET_FUNCTIONS = {
    'clean_col',
    'robust_load_csv',
    'normalize_text',
    'make_test_cases',
    'recommend_popularity',
    'recommend_cooccurrence',
    'evaluate_model',
}


def load_notebook_functions():
    """Load only the target function definitions from the notebook."""
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {
        '__builtins__': __builtins__,
        'csv': csv,
        're': re,
        'time': time,
        'random': random,
        'unicodedata': unicodedata,
        'Counter': Counter,
        'defaultdict': defaultdict,
        'np': np,
        'pd': pd,
    }

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        fn_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in TARGET_FUNCTIONS]
        if fn_nodes:
            module = ast.Module(body=fn_nodes, type_ignores=[])
            exec(compile(module, filename=str(NOTEBOOK_PATH), mode='exec'), namespace)

    missing = TARGET_FUNCTIONS - set(namespace)
    if missing:
        raise RuntimeError(f'Missing functions in notebook: {sorted(missing)}')
    return namespace


@pytest.fixture(scope='module')
def nb_ns():
    return load_notebook_functions()


def test_clean_col_strips_outer_quotes_and_spaces(nb_ns):
    clean_col = nb_ns['clean_col']
    assert clean_col('  "artistname"  ') == 'artistname'
    assert clean_col(' playlistname ') == 'playlistname'
    assert clean_col('"a,b"') == 'a,b'


def test_robust_load_csv_drops_malformed_rows_and_preserves_good_rows(nb_ns, tmp_path, capsys):
    robust_load_csv = nb_ns['robust_load_csv']

    csv_text = '\n'.join([
        '"user_id","artistname","trackname","playlistname"',
        '1,Taylor Swift,Lover,Pop Mix',
        '2,Adele,Hello',  # malformed: only 3 columns
        '3,Daft Punk,One More Time,Dance',
        '4,"AC/DC","Back In Black","Rock"',
    ])
    csv_path = tmp_path / 'sample.csv'
    csv_path.write_text(csv_text, encoding='utf-8')

    df = robust_load_csv(str(csv_path))
    out = capsys.readouterr().out

    assert list(df.columns) == ['user_id', 'artistname', 'trackname', 'playlistname']
    assert df.shape == (3, 4)
    assert 'Malformed rows dropped: 1' in out
    assert df.iloc[2]['artistname'] == 'AC/DC'


def test_normalize_text_handles_accents_symbols_whitespace_and_nan(nb_ns):
    normalize_text = nb_ns['normalize_text']

    assert normalize_text('  Beyoncé & JAY-Z!!! ') == 'beyonce and jay z'
    assert normalize_text('Panic! At The Disco') == 'panic at the disco'
    assert normalize_text('Crème   brûlée') == 'creme brulee'
    assert normalize_text(np.nan) == ''


def test_make_test_cases_builds_valid_masked_cases(nb_ns):
    make_test_cases = nb_ns['make_test_cases']

    test_dict = {
        'p1': ['s1', 's2', 's3', 's4', 's5'],
        'p2': ['a', 'b', 'c'],
        'p3': ['x', 'y'],  # too short, should be skipped
    }

    random.seed(7)
    cases = make_test_cases(test_dict, min_input_size=2, max_hide=3)

    keys = {c['playlist_key'] for c in cases}
    assert keys == {'p1', 'p2'}

    for case in cases:
        observed = case['observed']
        hidden = case['hidden']
        original = test_dict[case['playlist_key']]

        assert len(observed) >= 2
        assert len(hidden) >= 1
        assert set(observed).isdisjoint(hidden)
        assert set(observed) | hidden == set(original)


def test_recommend_popularity_excludes_seen_and_respects_order(nb_ns):
    recommend_popularity = nb_ns['recommend_popularity']
    nb_ns['popular_ranking'] = ['s1', 's2', 's3', 's4', 's5']

    recs = recommend_popularity(['s2', 's4'], topk=3)
    assert recs == ['s1', 's3', 's5']


def test_recommend_cooccurrence_scores_candidates_and_backfills(nb_ns):
    recommend_cooccurrence = nb_ns['recommend_cooccurrence']

    nb_ns['train_song_popularity'] = Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1, 'f': 1})
    nb_ns['popular_ranking'] = ['a', 'b', 'c', 'd', 'e', 'f']
    nb_ns['song_to_playlists'] = defaultdict(set, {
        'a': {'p1', 'p2'},
        'b': {'p1'},
        'x': set(),
    })
    nb_ns['playlist_song_set'] = {
        'p1': {'a', 'b', 'c'},
        'p2': {'a', 'd'},
    }

    recs = recommend_cooccurrence(['a'], topk=5)
    assert recs[:3] == ['b', 'c', 'd']
    assert 'a' not in recs
    assert len(recs) == 5
    assert recs[3:] == ['e', 'f']


def test_recommend_cooccurrence_falls_back_to_popularity_when_no_signal(nb_ns):
    recommend_cooccurrence = nb_ns['recommend_cooccurrence']

    nb_ns['popular_ranking'] = ['p1', 'p2', 'p3']
    nb_ns['song_to_playlists'] = defaultdict(set)
    nb_ns['playlist_song_set'] = {}
    nb_ns['train_song_popularity'] = Counter({'p1': 3, 'p2': 2, 'p3': 1})

    recs = recommend_cooccurrence(['unknown', 'p2'], topk=2)
    assert recs == ['p1', 'p3']


def test_evaluate_model_computes_hitrate_recall_and_mrr(nb_ns):
    evaluate_model = nb_ns['evaluate_model']

    test_cases = [
        {'observed': ['a'], 'hidden': {'x', 'y'}},
        {'observed': ['b'], 'hidden': {'z'}},
    ]

    def recommender(observed, topk=3):
        if observed == ['a']:
            return ['q', 'x', 'w']
        return ['z', 't', 'u']

    metrics = evaluate_model(test_cases, recommender, topk=3)

    assert metrics['num_test_cases'] == 2
    assert metrics['HitRate@3'] == pytest.approx(1.0)
    assert metrics['Recall@3'] == pytest.approx((1/2 + 1.0) / 2)
    assert metrics['MRR@3'] == pytest.approx((1/2 + 1.0) / 2)
