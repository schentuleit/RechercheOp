from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from build_decision_dataset import load_decision_dataset
from model import VRPTWAttentionModel


@dataclass
class SplitSummary:
    train_examples: int
    val_examples: int
    train_seeds: List[int]
    val_seeds: List[int]


class DecisionDataset(Dataset):
    """Dataset PyTorch pour les exemples de décisions oracle."""

    def __init__(self, examples: Sequence[Dict[str, object]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "node_features": torch.tensor(ex["node_features"], dtype=torch.float32),
            "action_features": torch.tensor(ex["action_features"], dtype=torch.float32),
            "current_node": torch.tensor(ex["current_node"], dtype=torch.long),
            "dynamic_global": torch.tensor(ex["dynamic_global"], dtype=torch.float32),
            "feasible_mask": torch.tensor(ex["feasible_mask"], dtype=torch.float32),
            "target_next_node": torch.tensor(ex["target_next_node"], dtype=torch.long),
        }


def collate_decisions(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # node_features, action_features, feasible_mask ont des tailles variables selon n
    # → padding à la taille max du batch
    max_nodes = max(x["node_features"].shape[0] for x in batch)

    def pad_2d(t: torch.Tensor) -> torch.Tensor:
        """Pad (N, d) → (max_nodes, d) avec des zéros."""
        pad = max_nodes - t.shape[0]
        if pad == 0:
            return t
        return torch.cat([t, torch.zeros(pad, t.shape[1], dtype=t.dtype)], dim=0)

    def pad_1d(t: torch.Tensor) -> torch.Tensor:
        """Pad (N,) → (max_nodes,) avec des zéros (nœuds paddés = infaisables)."""
        pad = max_nodes - t.shape[0]
        if pad == 0:
            return t
        return torch.cat([t, torch.zeros(pad, dtype=t.dtype)], dim=0)

    return {
        "node_features":    torch.stack([pad_2d(x["node_features"]) for x in batch], dim=0),
        "action_features":  torch.stack([pad_2d(x["action_features"]) for x in batch], dim=0),
        "current_node":     torch.stack([x["current_node"] for x in batch], dim=0),
        "dynamic_global":   torch.stack([x["dynamic_global"] for x in batch], dim=0),
        "feasible_mask":    torch.stack([pad_1d(x["feasible_mask"]) for x in batch], dim=0),
        "target_next_node": torch.stack([x["target_next_node"] for x in batch], dim=0),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_seed(
    dataset: Sequence[Dict[str, object]],
    val_ratio: float = 0.2,
    split_seed: int = 123,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], SplitSummary]:
    """Sépare train/val par seeds d'instances pour éviter la fuite de données."""
    unique_seeds = sorted({int(ex["seed"]) for ex in dataset})
    if len(unique_seeds) < 2:
        raise ValueError("Il faut au moins 2 seeds distinctes pour faire un split train/val.")

    rng = random.Random(split_seed)
    shuffled = unique_seeds[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)

    val_seeds = sorted(shuffled[:n_val])
    train_seeds = sorted(shuffled[n_val:])

    train_examples = [ex for ex in dataset if int(ex["seed"]) in train_seeds]
    val_examples = [ex for ex in dataset if int(ex["seed"]) in val_seeds]

    summary = SplitSummary(
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )
    return train_examples, val_examples, summary


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        node_features = batch["node_features"].to(device)
        action_features = batch["action_features"].to(device)
        current_node = batch["current_node"].to(device)
        dynamic_global = batch["dynamic_global"].to(device)
        feasible_mask = batch["feasible_mask"].to(device)
        targets = batch["target_next_node"].to(device)

        logits = model(
            node_features=node_features,
            action_features=action_features,
            current_node=current_node,
            dynamic_global=dynamic_global,
            feasible_mask=feasible_mask,
        )

        loss = loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.size(0))

    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        node_features = batch["node_features"].to(device)
        action_features = batch["action_features"].to(device)
        current_node = batch["current_node"].to(device)
        dynamic_global = batch["dynamic_global"].to(device)
        feasible_mask = batch["feasible_mask"].to(device)
        targets = batch["target_next_node"].to(device)

        optimizer.zero_grad()
        logits = model(
            node_features=node_features,
            action_features=action_features,
            current_node=current_node,
            dynamic_global=dynamic_global,
            feasible_mask=feasible_mask,
        )
        loss = loss_fn(logits, targets)
        loss.backward()

        # Clipping du gradient — indispensable avec les Transformers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((preds == targets).sum().item())
        total_count += int(targets.size(0))

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement Attention Model VRPTW")
    parser.add_argument("--data", type=str, default="decision_dataset_n10.npy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LR initial — le scheduler cosine descend jusqu'à lr/100")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Dimension des embeddings (attention + FFN)")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Nombre de têtes d'attention")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Nombre de couches Transformer dans l'encodeur")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--save_dir", type=str, default="artifacts_train_small")
    parser.add_argument("--output", type=str, default=None,
                        help="Nom du fichier .pt de sortie (ex: best_model_n20.pt). "
                             "Par défaut : best_model.pt dans save_dir.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Chargement dataset: {args.data}")
    dataset = load_decision_dataset(args.data)
    print(f"Nb exemples total: {len(dataset)}")
    if not dataset:
        raise ValueError("Dataset vide.")

    train_examples, val_examples, split_summary = split_by_seed(
        dataset=dataset,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    print("\n=== Split par seed ===")
    print(f"Train seeds : {split_summary.train_seeds[:5]}... ({len(split_summary.train_seeds)} seeds)")
    print(f"Val seeds   : {split_summary.val_seeds[:5]}... ({len(split_summary.val_seeds)} seeds)")
    print(f"Train ex    : {split_summary.train_examples}")
    print(f"Val ex      : {split_summary.val_examples}")

    train_ds = DecisionDataset(train_examples)
    val_ds = DecisionDataset(val_examples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_decisions,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
    )

    sample = dataset[0]
    node_dim = int(np.asarray(sample["node_features"]).shape[1])
    action_dim = int(np.asarray(sample["action_features"]).shape[1])
    global_dim = int(np.asarray(sample["dynamic_global"]).shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice    : {device}")
    print(f"node_dim  : {node_dim}  action_dim: {action_dim}  global_dim: {global_dim}")
    print(f"d_model   : {args.d_model}  n_heads: {args.n_heads}  n_layers: {args.n_layers}")

    model = VRPTWAttentionModel(
        node_dim=node_dim,
        action_dim=action_dim,
        global_dim=global_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing : LR descend de lr jusqu'à lr/100 sur toute la durée
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100,
    )

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_model_path = os.path.join(args.save_dir, args.output if args.output else "best_model.pt")
    history_path = os.path.join(args.save_dir, "history.json")
    config_path = os.path.join(args.save_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\n{'Epoch':>5} | {'train_loss':>10} | {'train_acc':>9} | {'val_loss':>8} | {'val_acc':>8} | {'lr':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": current_lr,
        }
        history.append(row)

        print(
            f"{epoch:5d} | "
            f"{row['train_loss']:10.4f} | "
            f"{row['train_acc']:9.2%} | "
            f"{row['val_loss']:8.4f} | "
            f"{row['val_acc']:8.2%} | "
            f"{current_lr:8.2e}"
        )

        # Sauvegarde history après chaque epoch (monitoring en temps réel)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {
                        "node_dim": node_dim,
                        "action_dim": action_dim,
                        "global_dim": global_dim,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                    },
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )

    print("\n=== Fin entraînement ===")
    print(f"Meilleure val_acc : {best_val_acc:.2%}")
    print(f"Modèle sauvegardé : {best_model_path}")
    print(f"Historique        : {history_path}")
    print(f"Config            : {config_path}")


if __name__ == "__main__":
    main()
