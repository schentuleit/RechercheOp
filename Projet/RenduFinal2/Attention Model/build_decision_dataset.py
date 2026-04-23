"""
build_decision_dataset.py — Dataset séquentiel de décisions oracle pour VRPTW
=============================================================================

Objectif
--------
Construire un nouveau dataset supervisé pour l'approche :
    instance VRPTW -> suite de décisions "prochain nœud"

Le dataset est généré à partir de :
    1. preprocess.py  -> génération d'instances
    2. oracle.py      -> résolution de référence avec OR-Tools

On n'utilise pas l'ancien dataset d'arêtes comme source d'entraînement.
Ici, chaque exemple représente un état partiel de construction de tournée,
et la cible est le prochain nœud choisi par l'oracle :
    - un client j dans {1, ..., n}
    - ou 0 pour signifier "retour dépôt"

Philosophie
-----------
Le futur modèle apprendra à scorer les actions faisables, puis decoder.py
imposera les contraintes à l'inférence.

Chaque exemple contient notamment :
    - les node_features statiques de l'instance
    - le nœud courant
    - le temps courant
    - la charge courante
    - les clients déjà servis (mask)
    - les actions faisables à cet instant (feasible_mask)
    - la cible oracle (target_next_node)

Format
------
Le dataset retourné est une liste de dictionnaires Python, puis sauvegardé
en .npy (dtype=object), comme dans oracle.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import math
import time
import numpy as np

from preprocess import generate_instance, preprocess
from oracle import resoudre_instance
from decoder import is_feasible_next_client


@dataclass
class ReplayState:
    """État courant lors du rejeu d'une tournée oracle."""

    current_node: int
    current_time: float
    current_load: float
    served_global: Set[int]
    served_in_route: List[int]


def _service_time(instance: dict, node: int) -> float:
    return float(instance["service_times"][node])


def _travel_time(instance: dict, i: int, j: int) -> float:
    return float(instance["durees"][i, j])


def _travel_cost(instance: dict, i: int, j: int) -> float:
    return float(instance["dist"][i, j])


def _time_window(instance: dict, node: int) -> Tuple[float, float]:
    a_i, b_i = instance["time_windows"][node]
    return float(a_i), float(b_i)


def _route_with_depots(route_clients: Sequence[int]) -> List[int]:
    return [0] + list(route_clients) + [0]


def compute_static_node_features(instance: dict) -> np.ndarray:
    """
    Features statiques des nœuds.

    Base : preprocess(instance) -> shape (N, 6)
    Ajout : flag dépôt -> shape (N, 7)
    """
    base = preprocess(instance).astype(np.float32)
    n_nodes = base.shape[0]
    depot_flag = np.zeros((n_nodes, 1), dtype=np.float32)
    depot_flag[0, 0] = 1.0
    return np.concatenate([base, depot_flag], axis=1)


def compute_feasible_action_mask(instance: dict, state: ReplayState) -> np.ndarray:
    """
    Construit un masque booléen des actions faisables sur l'espace 0..n.

    Convention
    ----------
    - action 0 : retour dépôt
    - actions 1..n : visiter le client correspondant

    Retour dépôt (0) est autorisé seulement si la route a déjà au moins un client
    et si le retour au dépôt reste dans l'horizon.
    """
    n = int(instance["n"])
    horizon = float(instance["horizon"])
    feasible = np.zeros(n + 1, dtype=np.float32)

    # Action 0 = retour dépôt
    if state.current_node != 0 or len(state.served_in_route) > 0:
        return_time = (
            state.current_time
            + _service_time(instance, state.current_node)
            + _travel_time(instance, state.current_node, 0)
        )
        if return_time <= horizon + 1e-9:
            feasible[0] = 1.0

    # Actions clients
    for client in range(1, n + 1):
        ok, _ = is_feasible_next_client(
            instance=instance,
            current_node=state.current_node,
            current_time=state.current_time,
            current_load=state.current_load,
            next_client=client,
            served_clients=state.served_global,
        )
        if ok:
            feasible[client] = 1.0

    return feasible


def advance_state_with_target(instance: dict, state: ReplayState, target_next_node: int) -> ReplayState:
    """
    Avance l'état courant en appliquant l'action oracle cible.

    target_next_node = 0 signifie fermeture de la route par retour dépôt.
    """
    if target_next_node == 0:
        next_time = (
            state.current_time
            + _service_time(instance, state.current_node)
            + _travel_time(instance, state.current_node, 0)
        )
        return ReplayState(
            current_node=0,
            current_time=float(next_time),
            current_load=float(state.current_load),
            served_global=set(state.served_global),
            served_in_route=list(state.served_in_route),
        )

    ok, info = is_feasible_next_client(
        instance=instance,
        current_node=state.current_node,
        current_time=state.current_time,
        current_load=state.current_load,
        next_client=target_next_node,
        served_clients=state.served_global,
    )
    if not ok:
        raise ValueError(
            f"Action oracle infaisable pendant le rejeu: "
            f"{state.current_node} -> {target_next_node}"
        )

    next_served_global = set(state.served_global)
    next_served_global.add(target_next_node)

    next_served_in_route = list(state.served_in_route)
    next_served_in_route.append(target_next_node)

    return ReplayState(
        current_node=int(target_next_node),
        current_time=float(info["service_start"]),
        current_load=float(info["next_load"]),
        served_global=next_served_global,
        served_in_route=next_served_in_route,
    )


def build_decision_example(
    instance: dict,
    node_features: np.ndarray,
    state: ReplayState,
    target_next_node: int,
    vehicle_id: int,
    step_in_vehicle: int,
    route_id: int,
) -> Dict[str, object]:
    """
    Construit un exemple d'entraînement pour une décision oracle.
    """
    n = int(instance["n"])
    Q = float(instance["capacity"])
    T = float(instance["horizon"])
    feasible_mask = compute_feasible_action_mask(instance, state)

    if feasible_mask[target_next_node] != 1.0:
        raise ValueError(
            f"Cible oracle {target_next_node} non marquée faisable. "
            f"Noeud courant={state.current_node}, temps={state.current_time:.2f}"
        )

    served_mask = np.zeros(n + 1, dtype=np.float32)
    for c in state.served_global:
        served_mask[c] = 1.0

    remaining_mask = 1.0 - served_mask
    remaining_mask[0] = 0.0

    current_node_one_hot = np.zeros(n + 1, dtype=np.float32)
    current_node_one_hot[state.current_node] = 1.0

    dynamic_global = np.array(
        [
            state.current_time / T,
            state.current_load / Q,
            len(state.served_global) / max(1, n),
            len(state.served_in_route) / max(1, n),
            vehicle_id / max(1, int(instance["n_vehicles"])),
            step_in_vehicle / max(1, n + 1),
        ],
        dtype=np.float32,
    )

    # Features par action (0..n).
    # [0] distance normalisée         — coût de déplacement
    # [1] slack temporel normalisé    — urgence de la fenêtre de temps
    #     slack = max(0, b_j - service_start_j) avec service_start depuis l'état courant
    #     Remplace travel_time/T qui était redondant avec [0] (dist * vitesse = cst).
    # [2] ouverture TW absolue        — contrainte statique
    # [3] fermeture TW absolue        — contrainte statique
    # [4] demande normalisée          — contrainte de capacité
    action_features = np.zeros((n + 1, 5), dtype=np.float32)
    dist_max = float(instance["dist"].max()) + 1e-6
    for node in range(n + 1):
        tt = _travel_time(instance, state.current_node, node)
        a_i, b_i = _time_window(instance, node)

        action_features[node, 0] = _travel_cost(instance, state.current_node, node) / dist_max

        # Slack : temps disponible APRÈS début de service si on part maintenant
        arrival = state.current_time + _service_time(instance, state.current_node) + tt
        service_start = max(arrival, a_i)
        slack = max(0.0, b_i - service_start)
        action_features[node, 1] = slack / T

        action_features[node, 2] = a_i / T
        action_features[node, 3] = b_i / T
        if node == 0:
            action_features[node, 4] = 0.0
        else:
            action_features[node, 4] = float(instance["demands"][node]) / Q

    return {
        "node_features": node_features.astype(np.float32),          # (N, 7)
        "action_features": action_features.astype(np.float32),      # (N, 5)
        "current_node": int(state.current_node),
        "current_time": float(state.current_time),
        "current_load": float(state.current_load),
        "current_node_one_hot": current_node_one_hot,
        "served_mask": served_mask,
        "remaining_mask": remaining_mask,
        "feasible_mask": feasible_mask,
        "target_next_node": int(target_next_node),
        "vehicle_id": int(vehicle_id),
        "step_in_vehicle": int(step_in_vehicle),
        "route_id": int(route_id),
        "seed": int(instance.get("seed", -1)),
        "n": int(instance["n"]),
        "n_vehicles": int(instance["n_vehicles"]),
        "dynamic_global": dynamic_global,
    }


def extract_decision_examples_from_solution(
    instance: dict,
    solution: dict,
    include_return_to_depot: bool = True,
) -> List[Dict[str, object]]:
    """
    Transforme une solution oracle en exemples de décisions séquentielles.

    Notes
    -----
    - On rejoue les routes dans l'ordre renvoyé par OR-Tools.
    - Les clients déjà servis par les véhicules précédents restent marqués
      dans served_global.
    - Une décision cible vaut 0 lorsque l'oracle ferme la route.
    """
    node_features = compute_static_node_features(instance)
    examples: List[Dict[str, object]] = []
    served_global: Set[int] = set()

    oracle_routes = solution.get("tournees", [])
    route_id = 0

    for vehicle_id, route_clients in enumerate(oracle_routes, start=1):
        if not route_clients:
            continue

        route_id += 1
        state = ReplayState(
            current_node=0,
            current_time=0.0,
            current_load=0.0,
            served_global=set(served_global),
            served_in_route=[],
        )

        augmented_route = _route_with_depots(route_clients)
        # Décisions : 0->c1, c1->c2, ..., ck->0
        transitions = list(zip(augmented_route[:-1], augmented_route[1:]))

        step_in_vehicle = 0
        for expected_current, target_next_node in transitions:
            if state.current_node != expected_current:
                raise ValueError(
                    f"Incohérence de rejeu: courant={state.current_node}, attendu={expected_current}"
                )

            if target_next_node == 0 and not include_return_to_depot:
                state = advance_state_with_target(instance, state, target_next_node)
                continue

            ex = build_decision_example(
                instance=instance,
                node_features=node_features,
                state=state,
                target_next_node=int(target_next_node),
                vehicle_id=vehicle_id,
                step_in_vehicle=step_in_vehicle,
                route_id=route_id,
            )
            examples.append(ex)

            state = advance_state_with_target(instance, state, int(target_next_node))
            step_in_vehicle += 1

        served_global.update(route_clients)

    n = int(instance["n"])
    if len(served_global) != n:
        raise ValueError(
            f"Solution oracle incomplète pendant l'extraction: {len(served_global)}/{n} clients servis."
        )

    return examples


def build_decision_dataset(
    n: int,
    n_instances: int = 200,
    seed_offset: int = 0,
    time_limit_s: float = 5.0,
    include_return_to_depot: bool = False,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """
    Génère un dataset séquentiel de décisions oracle pour petites instances.

    Retour
    ------
    Liste de dictionnaires. Chaque élément = un état + une action cible.
    """
    dataset: List[Dict[str, object]] = []
    n_ok = 0
    n_skipped = 0
    total_examples = 0
    t0 = time.time()

    # Même règle que preprocess.py / oracle.py
    n_vehicles = max(3, math.ceil(n / 10))

    for idx in range(n_instances):
        seed = seed_offset + idx
        instance = generate_instance(n=n, n_vehicles=n_vehicles, seed=seed)
        solution = resoudre_instance(instance, time_limit_s=time_limit_s)

        if solution is None or not solution.get("faisable", False):
            n_skipped += 1
            if verbose and (idx + 1) % 25 == 0:
                print(f"[{idx+1}/{n_instances}] {n_ok} OK  {n_skipped} ignorées  {total_examples} décisions")
            continue

        try:
            examples = extract_decision_examples_from_solution(
                instance=instance,
                solution=solution,
                include_return_to_depot=include_return_to_depot,
            )
        except Exception as exc:
            n_skipped += 1
            if verbose:
                print(f"[seed={seed}] extraction échouée: {exc}")
            continue

        dataset.extend(examples)
        total_examples += len(examples)
        n_ok += 1

        if verbose and (idx + 1) % 25 == 0:
            elapsed = max(1e-6, time.time() - t0)
            rate = total_examples / elapsed
            print(
                f"[{idx+1:4d}/{n_instances}] {n_ok} OK  {n_skipped} ignorées  "
                f"{total_examples} décisions  {rate:.1f} dec/s"
            )

    if verbose:
        elapsed = time.time() - t0
        print("\n=== Dataset décisions oracle ===")
        print(f"Instances valides   : {n_ok}/{n_instances}")
        print(f"Instances ignorées  : {n_skipped}")
        print(f"Nb de décisions     : {total_examples}")
        print(f"Temps total         : {elapsed:.1f}s")
        if total_examples > 0:
            targets = np.array([ex["target_next_node"] for ex in dataset], dtype=np.int64)
            depot_rate = 100.0 * np.mean(targets == 0)
            print(f"Taux retour dépôt   : {depot_rate:.1f}%")

    return dataset


def summarize_decision_dataset(dataset: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Résumé simple du dataset pour debug rapide dans un notebook."""
    if not dataset:
        return {
            "n_examples": 0,
            "n_instances": 0,
            "avg_feasible_actions": 0.0,
            "depot_target_rate": 0.0,
        }

    seeds = sorted({int(ex["seed"]) for ex in dataset})
    feasible_counts = [int(np.sum(ex["feasible_mask"])) for ex in dataset]
    targets = np.array([int(ex["target_next_node"]) for ex in dataset], dtype=np.int64)

    return {
        "n_examples": len(dataset),
        "n_instances": len(seeds),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "avg_feasible_actions": float(np.mean(feasible_counts)),
        "min_feasible_actions": int(np.min(feasible_counts)),
        "max_feasible_actions": int(np.max(feasible_counts)),
        "depot_target_rate": float(np.mean(targets == 0)),
    }


def save_decision_dataset(dataset: Sequence[Dict[str, object]], path: str) -> None:
    """Sauvegarde le dataset au format .npy (dtype=object)."""
    np.save(path, np.array(list(dataset), dtype=object), allow_pickle=True)
    print(f"Dataset décisions sauvegardé : {path}  ({len(dataset)} exemples)")


def load_decision_dataset(path: str) -> List[Dict[str, object]]:
    """Recharge un dataset de décisions sauvegardé en .npy."""
    data = np.load(path, allow_pickle=True)
    return list(data)


if __name__ == "__main__":
    # Convention de séparation des seeds :
    #   train  : seed_offset=0,     n_instances=1000  → seeds 0..999
    #   val    : seed_offset=5000,  n_instances=200   → seeds 5000..5199
    #   test   : seed_offset=10000, n_instances=100   → seeds 10000..10099
    dataset = build_decision_dataset(
        n=10,
        n_instances=1000,
        seed_offset=0,
        time_limit_s=5.0,
        include_return_to_depot=False,
        verbose=True,
    )
    save_decision_dataset(dataset, "decision_dataset_n10.npy")
    print("\nRésumé:")
    print(summarize_decision_dataset(dataset))

    if dataset:
        ex0 = dataset[0]
        print("\nPremier exemple:")
        print(f"seed={ex0['seed']}, current_node={ex0['current_node']}, target={ex0['target_next_node']}")
        print(f"feasible_mask={ex0['feasible_mask']}")
