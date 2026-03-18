#!/usr/bin/env python
"""Train/attack RUNG_combined and emit canonical clean+attack logs."""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.config.get_model import get_model_default
from gb.attack.gd import proj_grad_descent
from gb.metric import margin
from gb.pert import edge_diff_matrix
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy, get_log_identifier


# Default budget array (if not provided via CLI)
DEFAULT_ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


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
    split_id=0,
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

    # Progress bar for epoch training
    pbar = trange(1, max_epoch + 1, desc=f"  [Split {split_id}] Training", unit="epoch", leave=False)
    
    for epoch in pbar:
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

        pbar.set_postfix({"best_val_acc": f"{best_val_acc:.4f}"})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            pbar.close()
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        clean_acc = accuracy(model(A, X)[test_idx], y[test_idx]).item()
    return clean_acc


def attack_pgd(model, A, X, y, test_idx, budget=0.1, n_epochs=10, lr_attack=0.01, device="cpu", budget_id=0, total_budgets=6):
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
            progress=True,
            desc=f"    [Budget {budget_id+1}/{total_budgets}, b={budget:.2f}] Attack"
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

    # Generate model-specific log identifier for RUNG_combined
    # Ensure args has model attribute for the identifier function
    if not hasattr(args, 'model'):
        args.model = 'RUNG_combined'
    
    log_identifier = get_log_identifier(args.model, args)
    clean_log_path = f"log/{dataset}/clean/{log_identifier}.log"
    attack_log_path = f"log/{dataset}/attack/{log_identifier}.log"

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

    print(f"\n█ RUNG_combined — Dataset: {dataset}")
    print(f"  Training {len(splits)} split(s)...")
    split_clean = []
    trained_models = []
    split_test_idx = []

    # Progress bar for splits (training phase)
    for split_id, (train_idx, val_idx, test_idx) in enumerate(tqdm(splits, desc="  Training splits", unit="split", leave=True)):
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
            split_id=split_id,
        )
        split_clean.append(clean_acc)
        trained_models.append(model)
        split_test_idx.append(test_idx)

    clean_fh.write(f"model RUNG_combined done, clean acc: {_fmt_stats(split_clean)}\n")
    clean_fh.close()

    if args.skip_attack:
        attack_fh.close()
        print(f"  ✓ Training complete. Skipped attack phase.\n")
        return clean_log_path, attack_log_path

    print(f"\n  Attacking with {len(args.budgets)} budget(s)...")
    
    # Progress bar for budgets (attack phase)
    for budget_id, budget in enumerate(tqdm(args.budgets, desc="  Attacking budgets", unit="budget", leave=True)):
        attack_fh.write(f"Budget: {budget}\n")
        attack_fh.write("Model:RUNG_combined\n")
        split_attack = []
        
        # Progress bar for splits during attack
        for split_idx, model in enumerate(tqdm(trained_models, desc=f"    Budget {budget_id+1}/{len(args.budgets)} (b={budget:.2f}): Attacks", 
                                                unit="model", leave=False)):
            attacked_acc = attack_pgd(
                model,
                A,
                X,
                y,
                split_test_idx[split_idx],
                budget=budget,
                n_epochs=args.attack_epochs,
                lr_attack=args.attack_lr,
                device=device,
                budget_id=budget_id,
                total_budgets=len(args.budgets),
            )
            split_attack.append(attacked_acc)
            attack_fh.write(f"{split_clean[split_idx]} {attacked_acc}\n")
        
        attack_fh.write(f"Clean: {_fmt_stats(split_clean)}\n")
        attack_fh.write(f"Attacked: {_fmt_stats(split_attack)}\n")

    attack_fh.close()
    print(f"  ✓ Attack complete.\n")
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
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=DEFAULT_ATTACK_BUDGETS,
        help="Attack budgets (fraction of edges). Centrally controlled from run_all.py."
    )
    parser.add_argument("--skip_attack", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  RUNG_combined: Train & Attack")
    print("="*70)
    
    all_results = []
    for dataset in args.datasets:
        clean_log, attack_log = _run_one_dataset(args, dataset)
        all_results.append((dataset, clean_log, attack_log))

    print("="*70)
    print("  Summary")
    print("="*70)
    for dataset, clean_log, attack_log in all_results:
        print(f"  {dataset:12} → clean: {clean_log}")
        if not args.skip_attack:
            print(f"  {' ':12} → attack: {attack_log}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
