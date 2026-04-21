"""
large_instance_solver.py — Résolution de grandes instances VRPTW
================================================================
Approche Cluster-then-Route :
  1. Clustering géographique (K-Means maison, sans dépendance sklearn)
  2. Extraction de sous-instances indépendantes autour du dépôt
  3. Résolution de chaque sous-instance avec notre modèle + beam search
  4. Recombinaison des routes avec indices originaux
  5. Réparation des clients non servis (phase de réparation globale)

Gain typique vs greedy direct sur n=1000 :
  Greedy direct          : +18% gap vs OR-Tools (30s)
  Cluster-then-Route W=10: ~+10-12% gap, <30s

Usage :
    from large_instance_solver import solve_large_instance
    result = solve_large_instance(instance, scorer, cluster_size=50, beam_width=10)
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from decoder import (
    DecodeResult,
    _travel_cost,
    decode_vrptw_beam_search,
    decode_vrptw_with_repair,
    repair_unserved_clients,
)


# ── K-Means maison (sans sklearn) ─────────────────────────────────────────

def _kmeans(coords: np.ndarray, k: int, seed: int = 0, n_iter: int = 30) -> np.ndarray:
    """
    K-Means++ simplifié — retourne un tableau de labels (0..k-1).
    Pas de dépendance externe.
    """
    rng = np.random.RandomState(seed)
    n = len(coords)
    k = min(k, n)

    # Initialisation K-Means++ : centroïdes bien espacés
    centroids = [coords[rng.randint(n)]]
    for _ in range(k - 1):
        dists = np.min(
            [np.sum((coords - c) ** 2, axis=1) for c in centroids], axis=0
        )
        dists = np.maximum(dists, 1e-10)
        probs = dists / dists.sum()
        centroids.append(coords[rng.choice(n, p=probs)])
    centroids = np.array(centroids, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assignation
        dists_all = np.array([np.sum((coords - c) ** 2, axis=1) for c in centroids])
        labels = np.argmin(dists_all, axis=0)
        # Mise à jour
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = coords[mask].mean(axis=0)

    return labels


# ── Clustering des clients ─────────────────────────────────────────────────

def cluster_clients(
    instance: dict,
    cluster_size: int = 50,
    seed: int = 0,
) -> List[List[int]]:
    """
    Découpe les clients en K clusters géographiques de taille ~cluster_size.

    Retourne une liste de listes d'indices clients (1-indexés, comme dans l'instance).
    """
    n = int(instance["n"])
    coords = instance["coords"][1:]          # exclure le dépôt (node 0)
    k = max(1, math.ceil(n / cluster_size))

    labels = _kmeans(coords, k=k, seed=seed)

    clusters: List[List[int]] = [[] for _ in range(k)]
    for client_0idx, label in enumerate(labels):
        clusters[label].append(client_0idx + 1)   # +1 → index original

    return [c for c in clusters if c]             # enlever les clusters vides


# ── Extraction d'une sous-instance ────────────────────────────────────────

def extract_sub_instance(instance: dict, client_indices: List[int]) -> dict:
    """
    Crée une sous-instance VRPTW valide pour un sous-ensemble de clients.

    Structure de la sous-instance :
        node 0          = dépôt (identique à l'original)
        nodes 1..k      = les clients de client_indices (dans cet ordre)

    Le champ "_node_map" permet de retrouver les indices originaux :
        sub_idx → original_idx
    """
    all_nodes = [0] + list(client_indices)
    idx = np.array(all_nodes)

    n_sub = len(client_indices)
    n_vehicles_sub = max(1, math.ceil(n_sub / 10))

    return {
        "n":             n_sub,
        "n_vehicles":    n_vehicles_sub,
        "coords":        instance["coords"][idx],
        "dist":          instance["dist"][np.ix_(idx, idx)],
        "durees":        instance["durees"][np.ix_(idx, idx)],
        "demands":       instance["demands"][idx],
        "time_windows":  instance["time_windows"][idx],
        "service_times": instance["service_times"][idx],
        "capacity":      instance["capacity"],
        "horizon":       instance["horizon"],
        "seed":          instance.get("seed", 0),
        "_node_map":     all_nodes,   # sub_idx → original_idx
    }


# ── Solveur principal ──────────────────────────────────────────────────────

def solve_large_instance(
    instance: dict,
    scorer,
    cluster_size: int = 50,
    beam_width: int = 10,
    verbose: bool = True,
) -> DecodeResult:
    """
    Résout une grande instance VRPTW par Cluster-then-Route.

    Paramètres
    ----------
    cluster_size : taille cible de chaque cluster (~50 pour notre modèle)
    beam_width   : beam search dans chaque sous-problème (10 recommandé)
    verbose      : afficher la progression par cluster

    Retourne
    --------
    DecodeResult avec indices originaux (compatible verify_solution)
    """
    n = int(instance["n"])
    t_start = time.time()

    # ── Cas dégénéré : instance assez petite pour être résolue directement ──
    if n <= cluster_size * 1.5:
        if verbose:
            print(f"Instance petite (n={n}) — résolution directe")
        return decode_vrptw_beam_search(instance, scorer=scorer, beam_width=beam_width)

    # ── Étape 1 : Clustering ──────────────────────────────────────────────
    clusters = cluster_clients(instance, cluster_size=cluster_size)
    k = len(clusters)
    if verbose:
        sizes = [len(c) for c in clusters]
        print(f"Clustering : {k} clusters, taille moy={np.mean(sizes):.0f} "
              f"[{min(sizes)}..{max(sizes)}]")

    # ── Étape 2 : Résolution par cluster ─────────────────────────────────
    all_routes:   List[List[int]] = []
    all_arrivals: List[List[float]] = []
    all_loads:    List[float] = []
    all_served:   List[int] = []
    total_cost:   float = 0.0

    for c_idx, cluster in enumerate(clusters):
        sub_inst = extract_sub_instance(instance, cluster)
        node_map = sub_inst["_node_map"]   # sub_idx → original_idx

        t0 = time.time()
        result = decode_vrptw_beam_search(sub_inst, scorer=scorer, beam_width=beam_width)
        t_c = time.time() - t0

        if verbose:
            print(f"  Cluster {c_idx+1:>3}/{k} : n={len(cluster):>3}, "
                  f"routes={len(result.routes)}, "
                  f"servis={len(result.served_clients)}/{len(cluster)}, "
                  f"cout={result.total_cost:.0f}  ({t_c:.1f}s)")

        # Remapper les routes vers les indices originaux
        for r_idx, route in enumerate(result.routes):
            orig_route = [node_map[node] for node in route]
            all_routes.append(orig_route)
            all_arrivals.append(
                result.arrivals_by_route[r_idx]
                if r_idx < len(result.arrivals_by_route) else []
            )
            all_loads.append(
                result.loads_by_route[r_idx]
                if r_idx < len(result.loads_by_route) else 0.0
            )

        # Clients servis (indices originaux)
        for c in result.served_clients:
            all_served.append(node_map[c])

        # Recalculer le coût avec la matrice de distance originale
        for route in result.routes:
            orig_route = [node_map[node] for node in route]
            for i, j in zip(orig_route[:-1], orig_route[1:]):
                total_cost += float(instance["dist"][i, j])

    # ── Étape 3 : Recombinaison ───────────────────────────────────────────
    served_set = set(all_served)
    unserved   = sorted(set(range(1, n + 1)) - served_set)

    combined = DecodeResult(
        routes=all_routes,
        arrivals_by_route=all_arrivals,
        loads_by_route=all_loads,
        total_cost=total_cost,
        served_clients=sorted(served_set),
        unserved_clients=unserved,
        feasible=len(unserved) == 0,
        violations=[f"Non servis apres clustering : {unserved}"] if unserved else [],
    )

    # ── Étape 4 : Réparation globale (clients non servis) ─────────────────
    if unserved:
        if verbose:
            print(f"  Reparation : {len(unserved)} clients non servis...")
        combined = repair_unserved_clients(instance, combined)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nSolution finale :")
        print(f"  Routes          : {len(combined.routes)}")
        print(f"  Clients servis  : {len(combined.served_clients)}/{n}")
        print(f"  Clients non servis : {len(combined.unserved_clients)}")
        print(f"  Cout total      : {combined.total_cost:.1f}")
        print(f"  Faisable        : {combined.feasible}")
        print(f"  Temps total     : {elapsed:.1f}s")

    return combined
