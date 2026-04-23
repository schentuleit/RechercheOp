"""
decoder.py — Décodeur faisable VRPTW pour petites instances
===========================================================

Objectif
--------
Construire des tournées de la forme :
    dépôt -> client(s) -> dépôt
à partir d'une instance VRPTW générée par preprocess.py.

Le décodeur n'autorise que des insertions faisables, ce qui garantit :
    - C1 : chaque client est servi au plus une fois
    - C3 : capacité véhicule respectée
    - C4 : fenêtres temporelles respectées
    - C5 : cohérence temporelle respectée par propagation explicite des temps
    - C6 : chaque véhicule part du dépôt au plus une fois

La contrainte C2 (conservation des flux) est satisfaite par construction,
car chaque route est représentée comme une séquence ordonnée unique.

Ce fichier est conçu pour :
    1. tester le pipeline sans deep learning,
    2. servir de base à un futur décodeur guidé par un modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import math
import numpy as np


ClientId = int
Route = List[int]
# Les deux derniers kwargs optionnels (vehicle_idx, step_in_vehicle) permettent
# aux scoreurs avancés de récupérer le contexte de route sans casser les scoreurs simples.
ScoreFunction = Callable[..., Dict[int, float]]


@dataclass
class RouteState:
    """État courant d'une route en construction."""

    route: Route
    current_node: int
    current_time: float
    current_load: float
    arrivals: List[float]
    feasible: bool = True


@dataclass
class DecodeResult:
    """Résultat complet du décodage VRPTW."""

    routes: List[Route]
    arrivals_by_route: List[List[float]]
    loads_by_route: List[float]
    total_cost: float
    served_clients: List[int]
    unserved_clients: List[int]
    feasible: bool
    violations: List[str]


# ---------------------------------------------------------------------------
# Utilitaires de temps et faisabilité
# ---------------------------------------------------------------------------

def _travel_time(instance: dict, i: int, j: int) -> float:
    """Retourne la durée de trajet entre i et j."""
    return float(instance["durees"][i, j])


def _travel_cost(instance: dict, i: int, j: int) -> float:
    """Retourne le coût utilisé pour l'objectif (distance euclidienne ici)."""
    return float(instance["dist"][i, j])


def _service_time(instance: dict, i: int) -> float:
    return float(instance["service_times"][i])


def _time_window(instance: dict, i: int) -> Tuple[float, float]:
    a_i, b_i = instance["time_windows"][i]
    return float(a_i), float(b_i)


def _demand(instance: dict, i: int) -> float:
    return float(instance["demands"][i])


def compute_arrival_after_move(instance: dict, current_node: int, current_time: float, next_client: int) -> Tuple[float, float, bool]:
    """
    Calcule l'heure d'arrivée et de début de service au client suivant.

    Retourne
    --------
    arrival_raw   : heure d'arrivée avant attente éventuelle
    service_start : heure réelle de début de service
    feasible      : True si la fenêtre temporelle du client est respectée
    """
    arrival_raw = current_time + _service_time(instance, current_node) + _travel_time(instance, current_node, next_client)
    a_j, b_j = _time_window(instance, next_client)
    service_start = max(arrival_raw, a_j)
    feasible = service_start <= b_j
    return arrival_raw, service_start, feasible


def is_feasible_next_client(
    instance: dict,
    current_node: int,
    current_time: float,
    current_load: float,
    next_client: int,
    served_clients: Set[int],
) -> Tuple[bool, Dict[str, float]]:
    """
    Vérifie si ajouter `next_client` en fin de route est faisable.

    La vérification couvre :
        - client non déjà servi,
        - capacité,
        - fenêtre temporelle du client,
        - possibilité de retour au dépôt avant la fin d'horizon.
    """
    info: Dict[str, float] = {}

    if next_client == 0:
        return False, info
    if next_client in served_clients:
        return False, info

    next_load = current_load + _demand(instance, next_client)
    capacity = float(instance["capacity"])
    if next_load > capacity + 1e-9:
        return False, info

    arrival_raw, service_start, tw_ok = compute_arrival_after_move(
        instance, current_node, current_time, next_client
    )
    if not tw_ok:
        return False, info

    horizon = float(instance["horizon"])
    finish_service = service_start + _service_time(instance, next_client)
    return_to_depot_time = finish_service + _travel_time(instance, next_client, 0)
    if return_to_depot_time > horizon + 1e-9:
        return False, info

    info["arrival_raw"] = arrival_raw
    info["service_start"] = service_start
    info["next_load"] = next_load
    info["return_to_depot_time"] = return_to_depot_time
    return True, info


def feasible_clients(
    instance: dict,
    current_node: int,
    current_time: float,
    current_load: float,
    served_clients: Set[int],
    candidate_clients: Optional[Sequence[int]] = None,
) -> List[int]:
    """Liste des clients encore servis faisables en fin de route."""
    n = int(instance["n"])
    candidates = candidate_clients if candidate_clients is not None else list(range(1, n + 1))
    feas = []
    for client in candidates:
        ok, _ = is_feasible_next_client(
            instance=instance,
            current_node=current_node,
            current_time=current_time,
            current_load=current_load,
            next_client=client,
            served_clients=served_clients,
        )
        if ok:
            feas.append(client)
    return feas


# ---------------------------------------------------------------------------
# Scoreurs
# ---------------------------------------------------------------------------

def nearest_feasible_scorer(
    instance: dict,
    current_node: int,
    current_time: float,
    current_load: float,
    served_clients: Set[int],
    candidate_clients: Sequence[int],
    **_kwargs,
) -> Dict[int, float]:
    """
    Heuristique simple pour tester le décodeur.

    Score élevé = client attractif.
    Ici on favorise les clients proches et urgents.
    """
    scores: Dict[int, float] = {}
    for j in candidate_clients:
        dist = _travel_cost(instance, current_node, j)
        a_j, b_j = _time_window(instance, j)
        slack = max(0.0, b_j - current_time)
        scores[j] = -dist - 0.01 * slack
    return scores


# ---------------------------------------------------------------------------
# Décodeur principal
# ---------------------------------------------------------------------------

def select_next_client(
    instance: dict,
    route_state: RouteState,
    served_clients: Set[int],
    scorer: Optional[ScoreFunction] = None,
    vehicle_idx: int = 0,
    step_in_vehicle: int = 0,
) -> Optional[int]:
    """
    Sélectionne le prochain client parmi les candidats faisables.

    Si aucun client n'est faisable, retourne None pour forcer le retour dépôt.
    """
    feas = feasible_clients(
        instance=instance,
        current_node=route_state.current_node,
        current_time=route_state.current_time,
        current_load=route_state.current_load,
        served_clients=served_clients,
    )
    if not feas:
        return None

    scorer = scorer or nearest_feasible_scorer
    scores = scorer(
        instance,
        route_state.current_node,
        route_state.current_time,
        route_state.current_load,
        served_clients,
        feas,
        vehicle_idx=vehicle_idx,
        step_in_vehicle=step_in_vehicle,
    )
    if not scores:
        return None

    # Choix déterministe : score max, puis plus petit id pour stabilité.
    best_client = max(feas, key=lambda c: (scores.get(c, -math.inf), -c))
    return best_client


def append_client_to_route(
    instance: dict,
    route_state: RouteState,
    next_client: int,
    served_clients: Set[int],
) -> bool:
    """
    Ajoute un client en fin de route si faisable.

    Retourne True si l'ajout a été effectué, False sinon.
    """
    ok, info = is_feasible_next_client(
        instance=instance,
        current_node=route_state.current_node,
        current_time=route_state.current_time,
        current_load=route_state.current_load,
        next_client=next_client,
        served_clients=served_clients,
    )
    if not ok:
        return False

    route_state.route.append(next_client)
    route_state.current_node = next_client
    route_state.current_time = float(info["service_start"])
    route_state.current_load = float(info["next_load"])
    route_state.arrivals.append(float(info["service_start"]))
    served_clients.add(next_client)
    return True


def close_route(instance: dict, route_state: RouteState) -> None:
    """Ferme la route par un retour au dépôt."""
    if route_state.route[-1] == 0:
        return

    return_time = (
        route_state.current_time
        + _service_time(instance, route_state.current_node)
        + _travel_time(instance, route_state.current_node, 0)
    )
    route_state.route.append(0)
    route_state.current_node = 0
    route_state.current_time = return_time
    route_state.arrivals.append(return_time)


def decode_vrptw(
    instance: dict,
    scorer: Optional[ScoreFunction] = None,
    max_steps_per_vehicle: Optional[int] = None,
) -> DecodeResult:
    """
    Construit une solution VRPTW faisable à partir d'une fonction de score.

    Stratégie :
        - on ouvre au plus `n_vehicles` routes,
        - chaque route commence au dépôt,
        - à chaque étape on choisit le meilleur client faisable,
        - s'il n'existe plus de client faisable, on ferme la route,
        - on passe au véhicule suivant.

    Important
    ---------
    Ce décodeur garantit des routes de la forme dépôt -> client(s) -> dépôt.
    """
    n = int(instance["n"])
    n_vehicles = int(instance["n_vehicles"])
    max_steps_per_vehicle = max_steps_per_vehicle or (n + 1)

    served_clients: Set[int] = set()
    routes: List[Route] = []
    arrivals_by_route: List[List[float]] = []
    loads_by_route: List[float] = []
    violations: List[str] = []

    for vehicle_idx in range(n_vehicles):
        if len(served_clients) == n:
            break

        state = RouteState(
            route=[0],
            current_node=0,
            current_time=0.0,
            current_load=0.0,
            arrivals=[0.0],
        )

        steps = 0
        while steps < max_steps_per_vehicle:
            next_client = select_next_client(
                instance=instance,
                route_state=state,
                served_clients=served_clients,
                scorer=scorer,
                vehicle_idx=vehicle_idx,
                step_in_vehicle=steps,
            )
            if next_client is None:
                break

            appended = append_client_to_route(
                instance=instance,
                route_state=state,
                next_client=next_client,
                served_clients=served_clients,
            )
            if not appended:
                violations.append(
                    f"Véhicule {vehicle_idx + 1}: tentative d'ajout infaisable du client {next_client}."
                )
                break
            steps += 1

        close_route(instance, state)

        # On ne conserve que les routes qui servent au moins un client.
        if len(state.route) > 2:
            routes.append(state.route)
            arrivals_by_route.append(state.arrivals)
            loads_by_route.append(state.current_load)

    unserved_clients = sorted(set(range(1, n + 1)) - served_clients)
    if unserved_clients:
        violations.append(f"Clients non servis: {unserved_clients}")

    total_cost = 0.0
    for route in routes:
        for i, j in zip(route[:-1], route[1:]):
            total_cost += _travel_cost(instance, i, j)

    feasible = len(unserved_clients) == 0
    return DecodeResult(
        routes=routes,
        arrivals_by_route=arrivals_by_route,
        loads_by_route=loads_by_route,
        total_cost=total_cost,
        served_clients=sorted(served_clients),
        unserved_clients=unserved_clients,
        feasible=feasible,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Phase de réparation — insertion des clients non servis
# ---------------------------------------------------------------------------

def _recompute_route(
    instance: dict,
    route: Route,
) -> Tuple[List[float], float, bool]:
    """
    Re-propage les temps et recalcule la charge d'une route complète.

    Retourne
    --------
    arrivals : service_start pour chaque nœud (temps d'arrivée au dépôt)
    load     : charge totale de la route
    feasible : True si toutes les fenêtres temporelles sont respectées
    """
    horizon = float(instance["horizon"])
    arrivals: List[float] = [0.0]   # dépôt de départ à t=0
    time = 0.0
    load = 0.0
    current = 0

    for next_node in route[1:]:
        arrival_raw = time + _service_time(instance, current) + _travel_time(instance, current, next_node)
        if next_node != 0:
            a_j, b_j = _time_window(instance, next_node)
            service_start = max(arrival_raw, a_j)
            if service_start > b_j + 1e-9:
                return arrivals, load, False
            time = service_start
            load += _demand(instance, next_node)
        else:
            time = arrival_raw
            if time > horizon + 1e-9:
                return arrivals, load, False
        arrivals.append(time)
        current = next_node

    return arrivals, load, True


def _check_insertion(
    instance: dict,
    route: Route,
    insert_pos: int,
    client: int,
    route_load: float,
) -> Tuple[bool, float]:
    """
    Vérifie si insérer `client` à la position `insert_pos` dans `route` est faisable.

    insert_pos = k signifie que le client est inséré entre route[k-1] et route[k].

    Retourne
    --------
    feasible     : True si l'insertion respecte toutes les contraintes
    cost_delta   : surcoût en distance de l'insertion (si faisable)
    """
    Q = float(instance["capacity"])
    if route_load + _demand(instance, client) > Q + 1e-9:
        return False, 0.0

    new_route = route[:insert_pos] + [client] + route[insert_pos:]
    _, _, feasible = _recompute_route(instance, new_route)
    if not feasible:
        return False, 0.0

    prev_node = route[insert_pos - 1]
    next_node = route[insert_pos]
    cost_delta = (
        _travel_cost(instance, prev_node, client)
        + _travel_cost(instance, client, next_node)
        - _travel_cost(instance, prev_node, next_node)
    )
    return True, cost_delta


def repair_unserved_clients(instance: dict, result: DecodeResult) -> DecodeResult:
    """
    Tente d'insérer les clients non servis dans les routes existantes.

    Stratégie
    ---------
    Pour chaque client non servi (trié par deadline croissante — les plus
    urgents d'abord) : on cherche la position d'insertion dans n'importe
    quelle route existante qui minimise le surcoût de distance tout en
    respectant capacité et fenêtres de temps.

    Si aucune insertion n'est possible pour un client, il reste non servi.

    Retourne
    --------
    Un nouveau DecodeResult avec les routes mises à jour.
    """
    if not result.unserved_clients:
        return result

    # Copie mutable des routes et charges
    routes: List[Route] = [list(r) for r in result.routes]
    loads: List[float] = list(result.loads_by_route)
    served: set = set(result.served_clients)

    # Traiter les clients non servis par deadline croissante (urgence)
    unserved_sorted = sorted(
        result.unserved_clients,
        key=lambda c: _time_window(instance, c)[1],
    )

    for client in unserved_sorted:
        best_delta = float("inf")
        best_route_idx: Optional[int] = None
        best_pos: Optional[int] = None

        for r_idx, route in enumerate(routes):
            # Tenter toutes les positions d'insertion (hors dépôts)
            for pos in range(1, len(route)):   # entre route[pos-1] et route[pos]
                feasible, delta = _check_insertion(
                    instance, route, pos, client, loads[r_idx]
                )
                if feasible and delta < best_delta:
                    best_delta = delta
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx is not None:
            # Insérer le client dans la meilleure position trouvée
            routes[best_route_idx].insert(best_pos, client)
            loads[best_route_idx] += _demand(instance, client)
            served.add(client)

    # Recalculer les temps d'arrivée pour toutes les routes modifiées
    arrivals_by_route: List[List[float]] = []
    for route in routes:
        arrivals, _, _ = _recompute_route(instance, route)
        arrivals_by_route.append(arrivals)

    unserved_clients = sorted(set(range(1, int(instance["n"]) + 1)) - served)

    total_cost = 0.0
    for route in routes:
        for i, j in zip(route[:-1], route[1:]):
            total_cost += _travel_cost(instance, i, j)

    violations: List[str] = []
    if unserved_clients:
        violations.append(f"Clients non servis après réparation : {unserved_clients}")

    return DecodeResult(
        routes=routes,
        arrivals_by_route=arrivals_by_route,
        loads_by_route=loads,
        total_cost=total_cost,
        served_clients=sorted(served),
        unserved_clients=unserved_clients,
        feasible=len(unserved_clients) == 0,
        violations=violations,
    )


def decode_vrptw_with_repair(
    instance: dict,
    scorer: Optional[ScoreFunction] = None,
    max_steps_per_vehicle: Optional[int] = None,
) -> DecodeResult:
    """
    Décodage VRPTW suivi d'une phase de réparation par insertion.

    Garantit les mêmes contraintes que decode_vrptw (C1–C6) sur la solution
    finale, vérifiables via verify_solution.
    """
    result = decode_vrptw(instance, scorer=scorer, max_steps_per_vehicle=max_steps_per_vehicle)
    if result.unserved_clients:
        result = repair_unserved_clients(instance, result)
    return result


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

from dataclasses import field as _field


@dataclass
class _Beam:
    """État interne d'un faisceau pendant le beam search."""
    routes:            List[Route]
    arrivals_by_route: List[List[float]]
    loads_by_route:    List[float]
    current_route:     Route
    current_arrivals:  List[float]
    served:            frozenset          # immuable pour éviter les copies profondes
    current_node:      int
    current_time:      float
    current_load:      float
    vehicle_idx:       int
    step_in_vehicle:   int
    cum_score:         float              # somme des log-scores du modèle (ranking)
    total_cost:        float


def _beam_close_route(beam: _Beam, instance: dict) -> _Beam:
    """Ferme la route courante et prépare le prochain véhicule."""
    route = list(beam.current_route)
    arrivals = list(beam.current_arrivals)

    if route[-1] != 0:
        ret_time = (
            beam.current_time
            + _service_time(instance, beam.current_node)
            + _travel_time(instance, beam.current_node, 0)
        )
        cost_back = _travel_cost(instance, beam.current_node, 0)
        route.append(0)
        arrivals.append(ret_time)
    else:
        cost_back = 0.0

    new_routes = beam.routes + ([route] if len(route) > 2 else [])
    new_arr    = beam.arrivals_by_route + ([arrivals] if len(route) > 2 else [])
    new_loads  = beam.loads_by_route + ([beam.current_load] if len(route) > 2 else [])

    return _Beam(
        routes=new_routes,
        arrivals_by_route=new_arr,
        loads_by_route=new_loads,
        current_route=[0],
        current_arrivals=[0.0],
        served=beam.served,
        current_node=0,
        current_time=0.0,
        current_load=0.0,
        vehicle_idx=beam.vehicle_idx + 1,
        step_in_vehicle=0,
        cum_score=beam.cum_score,
        total_cost=beam.total_cost + cost_back,
    )


def _beam_to_result(beam: _Beam, instance: dict) -> DecodeResult:
    """Convertit un faisceau terminé en DecodeResult."""
    # Fermer la route courante si elle contient des clients
    final = _beam_close_route(beam, instance)
    n = int(instance["n"])
    unserved = sorted(set(range(1, n + 1)) - set(final.served))
    violations = [f"Clients non servis : {unserved}"] if unserved else []
    return DecodeResult(
        routes=final.routes,
        arrivals_by_route=final.arrivals_by_route,
        loads_by_route=final.loads_by_route,
        total_cost=final.total_cost,
        served_clients=sorted(final.served),
        unserved_clients=unserved,
        feasible=len(unserved) == 0,
        violations=violations,
    )


def decode_vrptw_beam_search(
    instance: dict,
    scorer: Optional[ScoreFunction] = None,
    beam_width: int = 10,
) -> DecodeResult:
    """
    Beam search VRPTW : conserve les `beam_width` meilleures solutions partielles
    à chaque étape et retourne celle avec le coût total minimal.

    Contrairement au décodage glouton qui choisit le meilleur client à chaque pas,
    le beam search explore plusieurs chemins en parallèle, ce qui réduit
    significativement le gap d'optimalité sans nécessiter de réentraînement.

    Paramètres
    ----------
    beam_width : nombre de faisceaux conservés (5–20 offre un bon compromis)
    """
    n = int(instance["n"])
    n_vehicles = int(instance["n_vehicles"])
    scorer = scorer or nearest_feasible_scorer

    # Faisceau initial
    initial = _Beam(
        routes=[], arrivals_by_route=[], loads_by_route=[],
        current_route=[0], current_arrivals=[0.0],
        served=frozenset(),
        current_node=0, current_time=0.0, current_load=0.0,
        vehicle_idx=0, step_in_vehicle=0,
        cum_score=0.0, total_cost=0.0,
    )

    active: List[_Beam] = [initial]
    completed: List[_Beam] = []

    while active:
        next_active: List[_Beam] = []

        for beam in active:
            # Tous les clients servis ou plus de véhicules disponibles
            if len(beam.served) == n or beam.vehicle_idx >= n_vehicles:
                completed.append(beam)
                continue

            feas = feasible_clients(
                instance, beam.current_node,
                beam.current_time, beam.current_load,
                set(beam.served),
            )

            if not feas:
                # Fermer la route et passer au véhicule suivant
                new_beam = _beam_close_route(beam, instance)
                if new_beam.vehicle_idx >= n_vehicles:
                    completed.append(new_beam)
                else:
                    next_active.append(new_beam)
                continue

            # Scores du modèle pour les clients faisables
            scores = scorer(
                instance,
                beam.current_node,
                beam.current_time,
                beam.current_load,
                set(beam.served),
                feas,
                vehicle_idx=beam.vehicle_idx,
                step_in_vehicle=beam.step_in_vehicle,
            )

            # Expansion : un nouveau faisceau par client faisable
            for client in feas:
                score = scores.get(client, -math.inf)
                ok, info = is_feasible_next_client(
                    instance, beam.current_node, beam.current_time,
                    beam.current_load, client, set(beam.served),
                )
                if not ok:
                    continue

                move_cost = _travel_cost(instance, beam.current_node, client)
                new_beam = _Beam(
                    routes=beam.routes,
                    arrivals_by_route=beam.arrivals_by_route,
                    loads_by_route=beam.loads_by_route,
                    current_route=beam.current_route + [client],
                    current_arrivals=beam.current_arrivals + [float(info["service_start"])],
                    served=beam.served | {client},
                    current_node=client,
                    current_time=float(info["service_start"]),
                    current_load=float(info["next_load"]),
                    vehicle_idx=beam.vehicle_idx,
                    step_in_vehicle=beam.step_in_vehicle + 1,
                    cum_score=beam.cum_score + score,
                    total_cost=beam.total_cost + move_cost,
                )
                next_active.append(new_beam)

        # Garder les beam_width meilleurs faisceaux par score cumulé
        next_active.sort(key=lambda b: b.cum_score, reverse=True)
        active = next_active[:beam_width]

    # Parmi les faisceaux terminés, choisir le meilleur coût
    if not completed:
        # Fallback : décodage glouton standard
        return decode_vrptw_with_repair(instance, scorer=scorer)

    best = min(completed, key=lambda b: b.total_cost)
    result = _beam_to_result(best, instance)

    # Phase de réparation pour les clients non servis
    if result.unserved_clients:
        result = repair_unserved_clients(instance, result)
    return result


# ---------------------------------------------------------------------------
# Vérification post-décodage
# ---------------------------------------------------------------------------

def verify_solution(instance: dict, result: DecodeResult) -> Dict[str, object]:
    """
    Vérifie explicitement les contraintes C1 à C6 sur une solution décodée.

    Retourne un dictionnaire détaillé avec booléens et messages d'erreur.
    """
    n = int(instance["n"])
    Q = float(instance["capacity"])
    horizon = float(instance["horizon"])

    checks = {
        "C1_unique_service": True,
        "C2_route_flow": True,
        "C3_capacity": True,
        "C4_time_windows": True,
        "C5_time_consistency": True,
        "C6_depot_departure": True,
        "all_routes_depot_to_depot": True,
        "errors": [],
    }

    # C1
    visit_count = {i: 0 for i in range(1, n + 1)}
    for route in result.routes:
        for node in route[1:-1]:
            visit_count[node] += 1
    for node, count in visit_count.items():
        if count != 1:
            checks["C1_unique_service"] = False
            checks["errors"].append(f"C1 violée: client {node} visité {count} fois.")

    # C2 + dépôt/dépôt + C6
    for r_idx, route in enumerate(result.routes, start=1):
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            checks["all_routes_depot_to_depot"] = False
            checks["errors"].append(f"Route {r_idx} invalide: {route}")

        depot_departures = sum(1 for i, j in zip(route[:-1], route[1:]) if i == 0 and j != 0)
        if depot_departures > 1:
            checks["C6_depot_departure"] = False
            checks["errors"].append(
                f"C6 violée: route {r_idx} quitte le dépôt {depot_departures} fois."
            )

        # Par construction d'une séquence, le flux local est cohérent.
        # On garde tout de même un contrôle simple sur les répétitions de dépôt internes.
        if any(node == 0 for node in route[1:-1]):
            checks["C2_route_flow"] = False
            checks["errors"].append(
                f"C2 violée: dépôt interne détecté dans la route {r_idx}: {route}"
            )

    # C3, C4, C5
    for r_idx, route in enumerate(result.routes, start=1):
        time = 0.0
        load = 0.0
        current = 0
        for next_node in route[1:]:
            arrival = time + _service_time(instance, current) + _travel_time(instance, current, next_node)

            if next_node != 0:
                load += _demand(instance, next_node)
                if load > Q + 1e-9:
                    checks["C3_capacity"] = False
                    checks["errors"].append(
                        f"C3 violée: route {r_idx}, charge {load:.2f} > Q={Q:.2f}."
                    )

                a_j, b_j = _time_window(instance, next_node)
                service_start = max(arrival, a_j)
                if not (a_j - 1e-9 <= service_start <= b_j + 1e-9):
                    checks["C4_time_windows"] = False
                    checks["errors"].append(
                        f"C4 violée: route {r_idx}, client {next_node}, arrivée/service={service_start:.2f}, TW=[{a_j:.2f},{b_j:.2f}]."
                    )

                # C5 vérifiée par propagation explicite du temps ; on signale si on sort de l'horizon.
                time = service_start
            else:
                time = arrival

            current = next_node

        if time > horizon + 1e-9:
            checks["C5_time_consistency"] = False
            checks["errors"].append(
                f"C5 violée: route {r_idx}, retour dépôt à t={time:.2f} > horizon={horizon:.2f}."
            )

    checks["feasible"] = all(
        checks[k]
        for k in [
            "C1_unique_service",
            "C2_route_flow",
            "C3_capacity",
            "C4_time_windows",
            "C5_time_consistency",
            "C6_depot_departure",
            "all_routes_depot_to_depot",
        ]
    )
    return checks


# ---------------------------------------------------------------------------
# Démonstration minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from preprocess import generate_instance

    inst = generate_instance(n=10, seed=42)
    result = decode_vrptw(inst)
    checks = verify_solution(inst, result)

    print("=== Solution décodée ===")
    for idx, route in enumerate(result.routes, start=1):
        print(f"Véhicule {idx}: {route}")
    print(f"Coût total: {result.total_cost:.2f}")
    print(f"Clients servis: {result.served_clients}")
    print(f"Clients non servis: {result.unserved_clients}")
    print(f"Faisable: {result.feasible}")
    print("=== Vérification contraintes ===")
    for k, v in checks.items():
        if k != "errors":
            print(f"{k}: {v}")
    if checks["errors"]:
        print("Erreurs:")
        for err in checks["errors"]:
            print(" -", err)
