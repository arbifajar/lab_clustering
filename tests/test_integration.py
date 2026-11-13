from src.benchmark import run_benchmark
from src.data import ROOT, baskets_subset, load_audit, load_baskets, nested_indices
from src.mining import ALGORITHMS, min_count_for
from src.rules import make_rules


def test_sample_nested():
    a = nested_indices(100, 42, .25)
    b = nested_indices(100, 42, .50)
    assert len(a) == 25 and len(b) == 50
    assert set(a).issubset(set(b))


def test_real_source_counts_and_slide_18_rules():
    _, baskets = load_baskets()
    audit = load_audit()
    assert (len(baskets), audit['baris_awal'], audit['baris_valid'], audit['jumlah_produk_unik']) == (19960, 541909, 530104, 3922)
    expected = {(25, 3): 150, (25, 5): 33, (25, 8): 4,
                (50, 3): 141, (50, 5): 33, (50, 8): 4,
                (100, 3): 142, (100, 5): 36, (100, 8): 4}
    for (fraction, support), count in expected.items():
        subset = baskets_subset(baskets, 42, fraction / 100)
        itemsets = ALGORITHMS['Eclat'](subset, min_count_for(len(subset), support/100))
        assert len(itemsets) == count
        if (fraction, support) == (100, 3):
            assert itemsets[('22697', '22698')] == 632
            assert len(make_rules(itemsets, len(subset), .30, pair_only=True)) == 18


def test_run_benchmark_small_real_data():
    _, baskets = load_baskets()
    result = run_benchmark(tuple(baskets[:200]), fractions=(50, 100), supports=(5, 8), repeats=2, measure_memory=False)
    assert len(result.summary) == 2*2*3
    assert len(result.durations) == 2*2*3*2
    assert set(result.summary['Kesetaraan output']) == {'COCOK'}
    assert len(result.counts_by_scenario) == 4
