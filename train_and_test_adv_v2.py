#!/usr/bin/env python
"""Train/attack RUNG_percentile_adv_v2 and emit canonical clean+attack logs."""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.config.get_model import get_model_default
from experiments.run_ablation import pgd_attack
from gb.attack.gd import proj_grad_descent
from gb.metric import margin
from gb.pert import edge_diff_matrix
from train_eval_data.fit_percentile_adv_v2 import fit_percentile_adv_v2
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy


ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]


def attack_pgd_v2(model, A, X, y, test_idx, budget=0.1, n_steps=200, device="cpu"):
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
            iterations=n_steps,
            base_lr=0.01,
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


def main():
    parser = argparse.ArgumentParser(description="Train/attack RUNG_percentile_adv_v2 with canonical logs")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--percentile_q", type=float, default=0.75)
    parser.add_argument("--use_layerwise_q", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--percentile_q_late", type=float, default=0.65)
    parser.add_argument("--max_epoch", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--train_pgd_steps", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--attack_freq", type=int, default=3)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--attack_steps", type=int, default=200)
    parser.add_argument("--skip_attack", action="store_true")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(f"log/{args.dataset}/clean", exist_ok=True)
    os.makedirs(f"log/{args.dataset}/attack", exist_ok=True)

    clean_log_path = f"log/{args.dataset}/clean/RUNG_percentile_adv_v2_MCP_{args.percentile_q}.log"
    attack_log_path = f"log/{args.dataset}/attack/RUNG_percentile_adv_v2_normMCP_gamma{args.percentile_q}.log"
    clean_fh = open(clean_log_path, "w", buffering=1)
    attack_fh = open(attack_log_path, "w", buffering=1)

    A, X, y = get_dataset(args.dataset)
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

    split_clean = []
    trained_models = []
    split_test_idx = []

    print(f"Dataset={args.dataset}: adversarial v2 training on {len(splits)} split(s)")
    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)

        model, _ = get_model_default(
            args.dataset,
            "RUNG_percentile_gamma",
            custom_model_params=model_params,
        )
        model = copy.deepcopy(model)
        torch.manual_seed(split_id)

        _, clean_acc = fit_percentile_adv_v2(
            model,
            A,
            X,
            y,
            train_idx,
            val_idx,
            test_idx,
            attack_fn=pgd_attack,
            alpha=args.alpha,
            train_pgd_steps=args.train_pgd_steps,
            max_epoch=args.max_epoch,
            warmup_epochs=args.warmup_epochs,
            attack_freq=args.attack_freq,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
        )

        split_clean.append(clean_acc)
        trained_models.append(model)
        split_test_idx.append(test_idx)

    clean_fh.write(f"model RUNG_percentile_adv_v2 done, clean acc: {_fmt_stats(split_clean)}\n")
    clean_fh.close()

    if args.skip_attack:
        attack_fh.close()
        print(f"Wrote clean log: {clean_log_path}")
        return

    for budget in ATTACK_BUDGETS:
        attack_fh.write(f"Budget: {budget}\n")
        attack_fh.write("Model:RUNG_percentile_adv_v2\n")
        split_attack = []
        for i, model in enumerate(trained_models):
            attacked_acc = attack_pgd_v2(
                model,
                A,
                X,
                y,
                split_test_idx[i],
                budget=budget,
                n_steps=args.attack_steps,
                device=device,
            )
            split_attack.append(attacked_acc)
            attack_fh.write(f"{split_clean[i]} {attacked_acc}\n")

        attack_fh.write(f"Clean: {_fmt_stats(split_clean)}\n")
        attack_fh.write(f"Attacked: {_fmt_stats(split_attack)}\n")

    attack_fh.close()
    print(f"Wrote clean log: {clean_log_path}")
    print(f"Wrote attack log: {attack_log_path}")


if __name__ == "__main__":
    main()
