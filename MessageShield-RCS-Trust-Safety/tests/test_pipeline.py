from src.generate_data import generate_messages
from src.features import add_graph_features


def test_data_and_graph_features():
    df = generate_messages(500, seed=7)
    out = add_graph_features(df)
    assert len(out) == 500
    assert {"is_spam", "sender_out_degree", "fanout_ratio"}.issubset(out.columns)
    assert out.is_spam.isin([0, 1]).all()
