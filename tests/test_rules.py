from src.rules import make_rules


def test_two_directions_support_conf_lift():
    counts = {('a',): 3, ('b',): 2, ('a','b'): 2}
    df = make_rules(counts, 4, min_confidence=.0)
    assert len(df) == 2
    ra = df.query("Antecedent == 'a'").iloc[0]
    rb = df.query("Antecedent == 'b'").iloc[0]
    assert abs(ra['Support (%)'] - 50) < 1e-9
    assert abs(ra['Confidence (%)'] - 200/3) < 1e-9
    assert abs(rb['Confidence (%)'] - 100) < 1e-9
    assert abs(ra['Lift'] - 4/3) < 1e-9
    assert rb['Lift'] == ra['Lift']
