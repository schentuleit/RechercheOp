"""
train_reinforce.py — Entraînement REINFORCE (policy gradient) pour VRPTW
=========================================================================
Approche Kool et al. 2019 avec deux améliorations clés vs la v1 :

  [A] Baseline EMA (Exponential Moving Average) au lieu du rollout glouton
      → l'avantage devient positif dès que l'épisode est meilleur que la
        moyenne récente, même en début d'entraînement (modèle aléatoire).

  [B] Bonus d'entropie : loss += -entropy_coef × H(π)
      → pénalise les distributions trop piquées, force l'exploration et
        évite l'effondrement prématuré vers une politique déterministe.

Usage :
    # Depuis zéro (recommandé — vrai Kool)
    python train_reinforce.py --n 10 --epochs 200 --batch_size 128

    # Avec resume
    python train_reinforce.py --resume artifacts_train_small/best_model_reinforce.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from preprocess import generate_instance
from model import VRPTWAttentionModel
from decoder import (
    DecodeResult,
    RouteState,
    _travel_cost,
    close_route,
    feasible_clients,
    is_feasible_next_client,
    repair_unserved_clients,
)
from build_decision_dataset import compute_static_node_features
from inference_decoder import build_inference_tensors, load_trained_model


# ── Décodage stochastique avec gradient + entropie ────────────────────────

def decode_stochastic(
    instance: dict,
    model: VRPTWAttentionModel,
    device: torch.device,
    unserved_penalty: float = 300.0,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Décode une instance de façon stochastique avec suivi de gradient.

    Retourne
    --------
    effective_cost : coût total + pénalité clients non servis
    log_prob_sum   : Σ log π(aₜ|sₜ)  — avec gradient (pour REINFORCE)
    entropy_sum    : Σ H(π(·|sₜ))    — avec gradient (pour bonus entropie)
    """
    n = int(instance["n"])
    n_vehicles = int(instance["n_vehicles"])

    # Encodage statique — une seule fois par instance
    node_features_np = compute_static_node_features(instance)
    nf = torch.tensor(node_features_np, dtype=torch.float32, device=device)
    h = model.encode(nf.unsqueeze(0))   # (1, N, d_model)

    served: set = set()
    routes: List[List[int]] = []
    arrivals_by_route: List[List[float]] = []
    loads_by_route: List[float] = []
    log_prob_sum: Optional[torch.Tensor] = None
    entropy_sum:  Optional[torch.Tensor] = None

    for v_idx in range(n_vehicles):
        if len(served) == n:
            break

        state = RouteState(
            route=[0], current_node=0,
            current_time=0.0, current_load=0.0, arrivals=[0.0],
        )

        for step in range(n + 1):
            feas = feasible_clients(
                instance, state.current_node,
                state.current_time, state.current_load, served,
            )
            if not feas:
                break

            tensors = build_inference_tensors(
                instance=instance,
                current_node=state.current_node,
                current_time=state.current_time,
                current_load=state.current_load,
                served_clients=served,
                candidate_clients=feas,
                vehicle_idx=v_idx,
                step_in_vehicle=step,
            )

            af = tensors["action_features"].to(device).unsqueeze(0)
            dg = tensors["dynamic_global"].to(device).unsqueeze(0)
            fm = tensors["feasible_mask"].to(device).unsqueeze(0)
            cn = torch.tensor([state.current_node], device=device)

            logits = model.decode(h, af, cn, dg, fm).squeeze(0)

            feas_idx = torch.tensor(feas, dtype=torch.long, device=device)
            feas_logits = logits[feas_idx]
            probs = torch.softmax(feas_logits, dim=0)

            dist_cat = Categorical(probs)
            sampled_idx = dist_cat.sample()

            # [A] log-prob de l'action choisie
            lp = dist_cat.log_prob(sampled_idx)
            log_prob_sum = lp if log_prob_sum is None else log_prob_sum + lp

            # [B] entropie de la distribution à ce pas
            ent = dist_cat.entropy()
            entropy_sum = ent if entropy_sum is None else entropy_sum + ent

            chosen = feas[sampled_idx.item()]
            ok, info = is_feasible_next_client(
                instance, state.current_node, state.current_time,
                state.current_load, chosen, served,
            )
            if not ok:
                break

            state.route.append(chosen)
            state.current_node = chosen
            state.current_time = float(info["service_start"])
            state.current_load = float(info["next_load"])
            state.arrivals.append(float(info["service_start"]))
            served.add(chosen)

        close_route(instance, state)
        if len(state.route) > 2:
            routes.append(state.route)
            arrivals_by_route.append(state.arrivals)
            loads_by_route.append(state.current_load)

    unserved = sorted(set(range(1, n + 1)) - served)
    total_cost = sum(
        _travel_cost(instance, i, j)
        for r in routes for i, j in zip(r[:-1], r[1:])
    )

    result = DecodeResult(
        routes=routes,
        arrivals_by_route=arrivals_by_route,
        loads_by_route=loads_by_route,
        total_cost=total_cost,
        served_clients=sorted(served),
        unserved_clients=unserved,
        feasible=len(unserved) == 0,
        violations=[],
    )
    if unserved:
        result = repair_unserved_clients(instance, result)

    effective_cost = result.total_cost + unserved_penalty * len(result.unserved_clients)

    zero = torch.zeros(1, device=device).squeeze()
    if log_prob_sum is None:
        log_prob_sum = zero
    if entropy_sum is None:
        entropy_sum = zero

    return effective_cost, log_prob_sum, entropy_sum


# ── Boucle d'entraînement ──────────────────────────────────────────────────

def train_reinforce(
    n: int = 10,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-4,
    seed_max: int = 9999,
    unserved_penalty: float = 300.0,
    ema_alpha: float = 0.95,       # [A] coefficient EMA baseline
    entropy_coef: float = 0.01,    # [B] poids du bonus d'entropie
    checkpoint_path: str = "artifacts_train_small/best_model_reinforce.pt",
    resume: Optional[str] = None,
    device_str: Optional[str] = None,
) -> None:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device        : {device}")
    print(f"Batch size    : {batch_size}")
    print(f"EMA alpha     : {ema_alpha}  (baseline)")
    print(f"Entropy coef  : {entropy_coef}")

    out_dir = Path("artifacts_train_small")
    out_dir.mkdir(exist_ok=True)

    model_kwargs = {
        "node_dim": 7, "action_dim": 5, "global_dim": 6,
        "d_model": 128, "n_heads": 8, "n_layers": 3, "dropout": 0.1,
    }

    if resume and Path(resume).exists():
        print(f"Resume depuis : {resume}")
        model = load_trained_model(resume)
    else:
        print("Nouveau modele (depuis zero)")
        model = VRPTWAttentionModel(**model_kwargs)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "avg_cost": [], "baseline": [], "avg_entropy": [],
    }

    best_cost = float("inf")
    ema_baseline: Optional[float] = None   # [A] initialisée au premier batch
    rng = random.Random(42)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametres    : {n_params:,}")
    print(f"\n{'Epoch':>6} | {'Loss':>9} | {'Cost':>8} | {'Baseline':>9} | {'Entropy':>8} | {'t(s)':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        seeds = [rng.randint(0, seed_max) for _ in range(batch_size)]
        instances = [generate_instance(n=n, seed=s) for s in seeds]

        # ── Collecter les coûts et log-probs du batch ─────────────────────────
        costs = []
        log_probs = []
        entropies = []

        for instance in instances:
            cost, lp, ent = decode_stochastic(instance, model, device, unserved_penalty)
            costs.append(cost)
            log_probs.append(lp)
            entropies.append(ent)

        mean_cost = sum(costs) / batch_size

        # [A] Mise à jour EMA baseline
        if ema_baseline is None:
            ema_baseline = mean_cost
        else:
            ema_baseline = ema_alpha * ema_baseline + (1 - ema_alpha) * mean_cost

        # ── Calcul de la loss ─────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss: Optional[torch.Tensor] = None
        mean_entropy = 0.0

        for cost, lp, ent in zip(costs, log_probs, entropies):
            # Avantage : positif si cet épisode est meilleur que la moyenne récente
            advantage = ema_baseline - cost

            # REINFORCE + bonus entropie
            loss_i = (-lp * advantage - entropy_coef * ent) / batch_size

            total_loss = loss_i if total_loss is None else total_loss + loss_i
            mean_entropy += ent.item()

        mean_entropy /= batch_size

        if total_loss is not None and total_loss.requires_grad:
            total_loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        avg_loss = total_loss.item() * batch_size if total_loss is not None else 0.0

        history["train_loss"].append(avg_loss)
        history["avg_cost"].append(mean_cost)
        history["baseline"].append(ema_baseline)
        history["avg_entropy"].append(mean_entropy)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {avg_loss:>+9.2f} | {mean_cost:>8.1f} | "
              f"{ema_baseline:>9.1f} | {mean_entropy:>8.3f} | {elapsed:>6.1f}")

        # Sauvegarde si meilleur coût moyen
        if mean_cost < best_cost:
            best_cost = mean_cost
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "best_cost": best_cost,
                "epoch": epoch,
            }, checkpoint_path)

        hist_name = Path(checkpoint_path).stem + "_history.json"
        with open(out_dir / hist_name, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\n=== Fin entrainement REINFORCE ===")
    print(f"Meilleur cout moyen  : {best_cost:.1f}")
    print(f"Modele sauvegarde    : {checkpoint_path}")


# ── Point d'entrée ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE VRPTW — baseline EMA + bonus entropie (Kool 2019)"
    )
    parser.add_argument("--n",               type=int,   default=10)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--unserved_penalty",type=float, default=300.0)
    parser.add_argument("--ema_alpha",       type=float, default=0.95,
                        help="Coefficient EMA pour la baseline (0.9-0.99)")
    parser.add_argument("--entropy_coef",    type=float, default=0.01,
                        help="Poids du bonus entropie (0.005-0.05)")
    parser.add_argument("--output",          type=str,
                        default="artifacts_train_small/best_model_reinforce.pt")
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--device",          type=str,   default=None)
    args = parser.parse_args()

    train_reinforce(
        n=args.n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unserved_penalty=args.unserved_penalty,
        ema_alpha=args.ema_alpha,
        entropy_coef=args.entropy_coef,
        checkpoint_path=args.output,
        resume=args.resume,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
