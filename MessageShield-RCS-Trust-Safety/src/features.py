from __future__ import annotations
import networkx as nx
import pandas as pd


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build sender→receiver graph signals that approximate messaging-abuse topology."""
    edges = df.groupby(["sender_id", "receiver_id"]).size().reset_index(name="edge_messages")
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges[["sender_id", "receiver_id", "edge_messages"]].itertuples(index=False, name=None))
    out_degree = dict(g.out_degree())
    weighted_out = dict(g.out_degree(weight="weight"))
    pagerank = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=100)
    out = df.copy()
    out["sender_out_degree"] = out["sender_id"].map(out_degree).fillna(0)
    out["sender_weighted_out"] = out["sender_id"].map(weighted_out).fillna(0)
    out["sender_pagerank"] = out["sender_id"].map(pagerank).fillna(0)
    out["fanout_ratio"] = out["sender_out_degree"] / out["sender_weighted_out"].clip(lower=1)
    return out
