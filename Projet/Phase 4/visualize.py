"""
visualize.py — Visualisation des routes VRPTW
==============================================
Génère une figure côte à côte :
    Gauche  : solution heuristique (nearest-feasible)
    Droite  : solution AM + réparation

Usage :
    python visualize.py --seed 10042 --save routes.png
    python visualize.py --seed 10042          # affiche sans sauvegarder
    python visualize.py --n 20 --seed 10005   # instance n=20
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from preprocess import generate_instance
from decoder import (
    decode_vrptw_with_repair,
    verify_solution,
    nearest_feasible_scorer,
    DecodeResult,
)
from inference_decoder import load_trained_model, TorchModelScorer


# ── Palette de couleurs ─────────────────────────────────────────────────────
ROUTE_COLORS = [
    "#2196F3",  # bleu
    "#4CAF50",  # vert
    "#FF9800",  # orange
    "#9C27B0",  # violet
    "#F44336",  # rouge
    "#00BCD4",  # cyan
    "#8BC34A",  # vert clair
    "#FF5722",  # rouge-orange
    "#607D8B",  # gris-bleu
    "#E91E63",  # rose
]
DEPOT_COLOR  = "#D32F2F"
CLIENT_COLOR = "#ECEFF1"
EDGE_ALPHA   = 0.85


# ── Utilitaires ─────────────────────────────────────────────────────────────

def _route_cost(instance: dict, route: List[int]) -> float:
    return sum(
        float(instance["dist"][i, j])
        for i, j in zip(route[:-1], route[1:])
    )


def _draw_solution(
    ax: plt.Axes,
    instance: dict,
    result: DecodeResult,
    title: str,
) -> None:
    """Dessine une solution VRPTW sur un axe matplotlib."""
    coords = instance["coords"]
    n = int(instance["n"])
    tw = instance["time_windows"]
    demands = instance["demands"]
    Q = float(instance["capacity"])

    # ── Fond et grille légère ────────────────────────────────────────────────
    ax.set_facecolor("#F8F9FA")
    ax.grid(color="white", linewidth=1.0, zorder=0)

    # ── Tracer les routes ────────────────────────────────────────────────────
    used_vehicles = 0
    for r_idx, route in enumerate(result.routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
        used_vehicles += 1

        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            xi, yi = coords[i]
            xj, yj = coords[j]
            dx, dy = xj - xi, yj - yi

            ax.annotate(
                "",
                xy=(xj, yj),
                xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.8,
                    alpha=EDGE_ALPHA,
                    shrinkA=7,
                    shrinkB=7,
                ),
                zorder=3,
            )

    # ── Clients ──────────────────────────────────────────────────────────────
    # Couleur du nœud = couleur de la route qui le visite
    node_color = {}
    for r_idx, route in enumerate(result.routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
        for node in route[1:-1]:
            node_color[node] = color

    for i in range(1, n + 1):
        x, y = coords[i]
        a_i, b_i = tw[i]
        color = node_color.get(i, "#BDBDBD")  # gris si non servi

        served = i in result.served_clients
        edge_c = color if served else "#F44336"
        lw = 2.0 if served else 2.5
        marker = "o" if served else "X"
        size = 120 if served else 140

        ax.scatter(x, y, s=size, c=CLIENT_COLOR, edgecolors=edge_c,
                   linewidths=lw, marker=marker, zorder=5)

        # Numéro du client
        ax.annotate(
            str(i),
            xy=(x, y),
            fontsize=7.5,
            ha="center", va="center",
            fontweight="bold",
            color="#37474F",
            zorder=6,
        )

        # Fenêtre de temps en petit
        ax.annotate(
            f"[{int(a_i)},{int(b_i)}]",
            xy=(x, y + 3.5),
            fontsize=5.5,
            ha="center", va="bottom",
            color="#607D8B",
            zorder=6,
        )

    # ── Dépôt ────────────────────────────────────────────────────────────────
    xd, yd = coords[0]
    ax.scatter(xd, yd, s=280, c=DEPOT_COLOR, marker="*",
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.annotate("Dépôt", xy=(xd, yd), xytext=(xd + 2.5, yd + 3),
                fontsize=8, color=DEPOT_COLOR, fontweight="bold", zorder=7)

    # ── Vérification ─────────────────────────────────────────────────────────
    checks = verify_solution(instance, result)
    feasible_str = "[OK] Faisable" if checks["feasible"] else "[KO] Infaisable"
    feasible_color = "#2E7D32" if checks["feasible"] else "#C62828"

    # ── Légende des véhicules ────────────────────────────────────────────────
    patches = []
    for r_idx, route in enumerate(result.routes):
        cost = _route_cost(instance, route)
        label = f"Véhicule {r_idx + 1}  ({len(route)-2} clients, {cost:.0f}km)"
        patches.append(
            mpatches.Patch(color=ROUTE_COLORS[r_idx % len(ROUTE_COLORS)], label=label)
        )
    if result.unserved_clients:
        patches.append(
            mpatches.Patch(color="#F44336",
                           label=f"Non servis : {result.unserved_clients}")
        )
    ax.legend(
        handles=patches,
        loc="lower left",
        fontsize=7,
        framealpha=0.9,
        edgecolor="#CFD8DC",
    )

    # ── Titre et annotations ─────────────────────────────────────────────────
    n_served = len(result.served_clients)
    ax.set_title(
        f"{title}\n"
        f"Coût total : {result.total_cost:.1f}   "
        f"Clients servis : {n_served}/{n}   "
        f"{feasible_str}",
        fontsize=10,
        fontweight="bold",
        color=feasible_color if not checks["feasible"] else "#212121",
        pad=10,
    )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-8, 112)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.tick_params(labelsize=8)

    # Info capacité
    ax.text(
        0.99, 0.01,
        f"Q={Q:.0f}  |  {used_vehicles} véhicules",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right", va="bottom",
        color="#78909C",
    )


# ── Figure principale ────────────────────────────────────────────────────────

def visualize(
    instance: dict,
    result_heuristic: DecodeResult,
    result_model: DecodeResult,
    oracle_cost: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 7.5),
        facecolor="white",
    )
    fig.subplots_adjust(wspace=0.08)

    _draw_solution(axes[0], instance, result_heuristic, "Heuristique (nearest-feasible + réparation)")
    _draw_solution(axes[1], instance, result_model,     "Attention Model + réparation")

    # Titre principal
    n = instance["n"]
    seed = instance.get("seed", "?")
    oracle_str = f"  |  Oracle : {oracle_cost:.1f}" if oracle_cost else ""
    gap_str = ""
    if oracle_cost and result_model.feasible:
        gap = (result_model.total_cost - oracle_cost) / oracle_cost * 100
        gap_str = f"  (gap +{gap:.1f}%)"

    fig.suptitle(
        f"VRPTW — n={n} clients, seed={seed}{oracle_str}{gap_str}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Figure sauvegardée : {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    import os
    from oracle import resoudre_instance

    parser = argparse.ArgumentParser(description="Visualisation routes VRPTW")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join("artifacts_train_small", "best_model.pt"))
    parser.add_argument("--save", type=str, default=None,
                        help="Chemin de sauvegarde PNG (ex: routes.png)")
    parser.add_argument("--no_oracle", action="store_true",
                        help="Ne pas appeler OR-Tools (plus rapide)")
    args = parser.parse_args()

    print(f"Instance n={args.n}, seed={args.seed}")
    instance = generate_instance(n=args.n, seed=args.seed)

    # Heuristique
    print("Heuristique...")
    result_heuristic = decode_vrptw_with_repair(instance, scorer=nearest_feasible_scorer)
    print(f"  cout={result_heuristic.total_cost:.1f}  "
          f"servis={len(result_heuristic.served_clients)}/{args.n}  "
          f"faisable={result_heuristic.feasible}")

    # Modèle AM
    print("Chargement modèle AM...")
    model = load_trained_model(args.checkpoint)
    scorer = TorchModelScorer(model=model)
    print("AM...")
    result_model = decode_vrptw_with_repair(instance, scorer=scorer)
    print(f"  cout={result_model.total_cost:.1f}  "
          f"servis={len(result_model.served_clients)}/{args.n}  "
          f"faisable={result_model.feasible}")

    # Oracle (optionnel)
    oracle_cost = None
    if not args.no_oracle:
        print("Oracle OR-Tools...")
        sol = resoudre_instance(instance, time_limit_s=10.0)
        if sol and sol["faisable"]:
            oracle_cost = float(sol["cout"])
            print(f"  cout oracle={oracle_cost:.1f}")

    visualize(
        instance=instance,
        result_heuristic=result_heuristic,
        result_model=result_model,
        oracle_cost=oracle_cost,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
