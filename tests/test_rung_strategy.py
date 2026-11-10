"""Forward and backward checks for every strategy preset."""

import torch

from model.rung_combined import RUNG_combined
from model.rung_strategy import RUNGStrategy
from model.strategy_presets import PRESETS, build_strategy_model


def _graph(n=12, d=8, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) > 0.7).float()
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    X = torch.randn(n, d, generator=g)
    y = torch.randint(0, c, (n,), generator=g)
    return A, X, y


def test_every_preset_forward_and_backward():
    A, X, y = _graph()
    n, d = X.shape
    c = int(y.max()) + 1
    train = torch.arange(4)
    for sid, preset in PRESETS.items():
        if preset.get("kind") == "baselines":
            continue
        cfg = dict(preset)
        cfg["strategy_id"] = sid
        cfg["prop_step"] = 2
        model = build_strategy_model(d, c, cfg)
        if hasattr(model, "set_graph_stats"):
            model.set_graph_stats(A, y, train)
        model.train()
        out = model(A, X)
        assert out.shape == (n, c), sid
        assert torch.isfinite(out).all(), sid
        out.sum().backward()


def test_edge_only_matches_dense():
    A, X, _ = _graph()
    d = X.shape[1]
    dense = RUNGStrategy(
        d, 3, hidden_dims=[16], distance="euclidean", gamma_mode="fixed",
        penalty="mcp", gamma=1.0, prop_step=3, dropout=0.0, strategy_id="dense",
    )
    sparse = RUNGStrategy(
        d, 3, hidden_dims=[16], distance="euclidean", gamma_mode="fixed",
        penalty="mcp", gamma=1.0, prop_step=3, dropout=0.0, edge_only=True,
        strategy_id="edge",
    )
    sparse.load_state_dict(dense.state_dict())
    dense.eval()
    sparse.eval()
    with torch.no_grad():
        a = dense(A, X)
        b = sparse(A, X)
    gap = (a - b).abs().max().item()
    assert gap < 1e-4, gap


def test_s06_matches_rung_combined():
    A, X, _ = _graph()
    d = X.shape[1]
    combined = RUNG_combined(d, 3, [16], prop_step=3, dropout=0.0)
    cfg = dict(PRESETS["S06"])
    cfg.update(strategy_id="S06", prop_step=3, dropout=0.0)
    strategy = build_strategy_model(d, 3, cfg, hidden_dims=[16])
    strategy.mlp.load_state_dict(combined.mlp.state_dict())
    combined.eval()
    strategy.eval()
    with torch.no_grad():
        a = combined(A, X)
        b = strategy(A, X)
    gap = (a - b).abs().max().item()
    assert gap < 1e-4, gap


if __name__ == "__main__":
    test_every_preset_forward_and_backward()
    test_edge_only_matches_dense()
    test_s06_matches_rung_combined()
    print("strategy presets ok")
