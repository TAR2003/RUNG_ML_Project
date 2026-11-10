"""Training loop for RUNGStrategy presets.

Supports clean cross-entropy, a clean/perturbed consistency term, and the
existing curriculum adversarial trainer. Sets ``model.current_epoch`` so a
preset can leave the distance attached for a short warmup.
"""

import torch
import torch.nn.functional as F
import tqdm

from utils import accuracy


def _perturb_edges(A, ratio):
    """Swap a fraction of undirected edges for the same number of non-edges."""
    n = A.shape[0]
    upper = torch.triu(torch.ones(n, n, device=A.device, dtype=torch.bool), diagonal=1)
    edges = ((A > 0) & upper).nonzero(as_tuple=False)
    non = ((A <= 0) & upper).nonzero(as_tuple=False)
    if edges.numel() == 0 or non.numel() == 0:
        return A
    n_flip = max(1, int(ratio * edges.shape[0]))
    n_flip = min(n_flip, edges.shape[0], non.shape[0])
    take_e = torch.randperm(edges.shape[0], device=A.device)[:n_flip]
    take_n = torch.randperm(non.shape[0], device=A.device)[:n_flip]
    out = A.clone()
    er, ec = edges[take_e, 0], edges[take_e, 1]
    nr, nc = non[take_n, 0], non[take_n, 1]
    out[er, ec] = 0
    out[ec, er] = 0
    out[nr, nc] = 1
    out[nc, nr] = 1
    return out


def _optimizer(model, lr, weight_decay):
    if hasattr(model, "build_optimizer"):
        return model.build_optimizer(lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _report_bias(model, A, X):
    from utils.metrics import compute_estimation_bias

    attacked = _perturb_edges(A, 0.05)
    device = A.device
    bias_total, bias_mean = compute_estimation_bias(model, A, X, attacked, device)
    print(f"  estimation bias total={float(bias_total):.4f} mean={float(bias_mean):.4f}")


def _fit_adversarial(model, A, X, y, train_idx, val_idx, test_idx, lr, weight_decay, max_epoch):
    from experiments.run_ablation import pgd_attack
    from train_eval_data.adversarial_trainer import AdversarialTrainer, CurriculumSchedule

    trainer = AdversarialTrainer(
        model=model,
        attack_fn=pgd_attack,
        curriculum=CurriculumSchedule(),
        alpha=0.7,
        attack_freq=5,
        train_pgd_steps=20,
        lr=lr,
        weight_decay=weight_decay,
        patience=100,
    )
    trainer.train(
        A, X, y, train_idx, val_idx, test_idx,
        epochs=max_epoch,
        device=str(A.device),
    )


def fit_strategy(
    model,
    A,
    X,
    y,
    train_idx,
    val_idx,
    test_idx=None,
    lr=5e-2,
    weight_decay=5e-4,
    max_epoch=300,
    patience=100,
    grad_clip=1.0,
    log_every=50,
    **kwargs,
):
    device = next(model.parameters()).device
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    if hasattr(model, "set_graph_stats"):
        model.set_graph_stats(A, y, train_idx)

    mode = getattr(model, "train_mode", "clean")
    print(
        f"\nFit {getattr(model, 'strategy_id', model.__class__.__name__)} | "
        f"mode={mode} lr={lr} epochs={max_epoch}"
    )
    if mode == "adversarial":
        if test_idx is None:
            raise ValueError("Adversarial strategy training needs test_idx.")
        _fit_adversarial(
            model, A, X, y, train_idx, val_idx, test_idx.to(device),
            lr, weight_decay, max_epoch,
        )
        if getattr(model, "report_bias", False):
            model.eval()
            with torch.no_grad():
                _report_bias(model, A, X)
        return

    optimizer = _optimizer(model, lr, weight_decay)
    best_val = -1.0
    best_epoch = 0
    best_state = None
    weight = float(getattr(model, "consistency_weight", 0.1))

    for epoch in tqdm.trange(max_epoch, desc="fit_strategy"):
        model.current_epoch = epoch
        if hasattr(model, "experts"):
            for expert in model.experts.values():
                expert.current_epoch = epoch
        model.train()
        optimizer.zero_grad()
        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        if mode == "consistency":
            perturbed = _perturb_edges(A, 0.05)
            other = model(perturbed, X)
            loss = loss + weight * F.kl_div(
                F.log_softmax(other[train_idx], dim=-1),
                F.softmax(logits[train_idx].detach(), dim=-1),
                reduction="batchmean",
            )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(A, X)[val_idx], y[val_idx]).item()
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log_every and (epoch + 1) % log_every == 0:
            print(f"\n  epoch {epoch + 1} loss={float(loss.detach()):.4f} val={val_acc:.4f}")
            if hasattr(model, "log_stats"):
                model.log_stats()
        if epoch - best_epoch > patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (best val={best_val:.4f} at epoch {best_epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _ = model(A, X)
    if hasattr(model, "log_stats"):
        model.log_stats()
    if getattr(model, "report_bias", False):
        _report_bias(model, A, X)
    print(f"\nTraining done. Best val acc: {best_val:.4f} (epoch {best_epoch + 1})")
