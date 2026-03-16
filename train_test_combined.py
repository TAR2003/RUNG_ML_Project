#!/usr/bin/env python
"""Train/attack RUNG_combined and emit canonical clean+attack logs."""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.config.get_model import get_model_default
from gb.attack.gd import proj_grad_descent
from gb.metric import margin
from gb.pert import edge_diff_matrix
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy


ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]


def train_clean(
    model,
    A,
    X,
    y,
    train_idx,
    val_idx,
    test_idx,
    max_epoch=300,
    lr=0.05,
    weight_decay=5e-4,
    early_stopping_patience=100,
    device="cpu",
):
    """Train one split and return clean test accuracy."""
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epoch + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(A, X)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 != 0:
            continue

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(A, X)[val_idx], y[val_idx]).item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        clean_acc = accuracy(model(A, X)[test_idx], y[test_idx]).item()
    return clean_acc


def attack_pgd(model, A, X, y, test_idx, budget=0.1, n_epochs=10, lr_attack=0.01, device="cpu"):
    """Run PGD attack and return attacked test accuracy."""
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)
    test_idx = test_idx.to(device)

    def loss_fn(flip):
        A_pert = A + (flip * (1.0 - 2.0 * A))
        out = model(A_pert, X)
        return margin(out[test_idx], y[test_idx]).mean()

    def grad_fn(flip):
        loss = loss_fn(flip)
        return torch.autograd.grad(loss, flip, create_graph=True)[0]

    budget_edge_num = int(budget * A.count_nonzero().item() // 2)
    try:
        edge_pert, _ = proj_grad_descent(
            flip_shape_or_init=A.shape,
            symmetric=True,
            device=A.device,
            budget=budget_edge_num,
            grad_fn=grad_fn,
            loss_fn=loss_fn,
            iterations=n_epochs,
            base_lr=lr_attack,
            grad_clip=1.0,
            progress=False,
        )
    except Exception:
        with torch.no_grad():
            return accuracy(model(A, X)[test_idx], y[test_idx]).item()

    A_attacked = A + edge_diff_matrix(edge_pert.long(), A) if edge_pert.numel() > 0 else A
    model.eval()
    with torch.no_grad():
        return accuracy(model(A_attacked, X)[test_idx], y[test_idx]).item()


def _fmt_stats(values):
    return f"{np.mean(values)}±{np.std(values)}: {values}"


def _run_one_dataset(args, dataset):
    os.makedirs(f"log/{dataset}/clean", exist_ok=True)
    os.makedirs(f"log/{dataset}/attack", exist_ok=True)

    clean_log_path = f"log/{dataset}/clean/RUNG_combined_MCP_{args.percentile_q}.log"
    attack_log_path = f"log/{dataset}/attack/RUNG_combined_normMCP_gamma{args.percentile_q}.log"

    clean_fh = open(clean_log_path, "w", buffering=1)
    attack_fh = open(attack_log_path, "w", buffering=1)

    A, X, y = get_dataset(dataset)
    splits = get_splits(y)
    device = torch.device(args.device)
    A = A.to(device)
    X = X.to(device)
    y = y.to(device)

    model_params = {
        "percentile_q": args.percentile_q,
        "use_layerwise_q": args.use_layerwise_q,
        "percentile_q_late": args.percentile_q_late,
    }

    print(f"Dataset={dataset}: training {len(splits)} split(s)")
    split_clean = []
    trained_models = []
    split_test_idx = []

    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        model, _ = get_model_default(dataset, "RUNG_combined", custom_model_params=model_params)
        model = copy.deepcopy(model)
        torch.manual_seed(split_id)
        clean_acc = train_clean(
            model,
            A,
            X,
            y,
            train_idx,
            val_idx,
            test_idx,
            max_epoch=args.max_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        split_clean.append(clean_acc)
        trained_models.append(model)
        split_test_idx.append(test_idx)

    clean_fh.write(f"model RUNG_combined done, clean acc: {_fmt_stats(split_clean)}\n")
    clean_fh.close()

    if args.skip_attack:
        attack_fh.close()
        return clean_log_path, attack_log_path

    for budget in ATTACK_BUDGETS:
        attack_fh.write(f"Budget: {budget}\n")
        attack_fh.write("Model:RUNG_combined\n")
        split_attack = []
        for i, model in enumerate(trained_models):
            attacked_acc = attack_pgd(
                model,
                A,
                X,
                y,
                split_test_idx[i],
                budget=budget,
                n_epochs=args.attack_epochs,
                lr_attack=args.attack_lr,
                device=device,
            )
            split_attack.append(attacked_acc)
            attack_fh.write(f"{split_clean[i]} {attacked_acc}\n")
        attack_fh.write(f"Clean: {_fmt_stats(split_clean)}\n")
        attack_fh.write(f"Attacked: {_fmt_stats(split_attack)}\n")

    attack_fh.close()
    return clean_log_path, attack_log_path


def main():
    parser = argparse.ArgumentParser(description="Train/attack RUNG_combined with canonical logs")
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--percentile_q", type=float, default=0.75)
    parser.add_argument("--use_layerwise_q", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--percentile_q_late", type=float, default=0.65)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--attack_epochs", type=int, default=10)
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument("--skip_attack", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for dataset in args.datasets:
        clean_log, attack_log = _run_one_dataset(args, dataset)
        print(f"Wrote clean log: {clean_log}")
        print(f"Wrote attack log: {attack_log}")


if __name__ == "__main__":
    main()
