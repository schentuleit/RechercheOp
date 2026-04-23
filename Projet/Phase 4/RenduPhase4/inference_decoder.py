from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch

from preprocess import generate_instance
from oracle import resoudre_instance
from decoder import (
    decode_vrptw,
    decode_vrptw_with_repair,
    verify_solution,
    nearest_feasible_scorer,
)
from model import VRPTWAttentionModel
from build_decision_dataset import compute_static_node_features


@dataclass
class BenchmarkItem:
    seed: int
    heuristic_feasible: bool
    model_feasible: bool
    heuristic_cost: float
    model_cost: float
    heuristic_served: int
    model_served: int
    oracle_cost: Optional[float]        # coût OR-Tools sur la même instance
    optimality_gap: Optional[float]     # (model_cost - oracle_cost) / oracle_cost


def _travel_cost(instance: dict, i: int, j: int) -> float:
    return float(instance["dist"][i, j])


def _travel_time(instance: dict, i: int, j: int) -> float:
    return float(instance["durees"][i, j])


def _time_window(instance: dict, node: int) -> tuple[float, float]:
    a_i, b_i = instance["time_windows"][node]
    return float(a_i), float(b_i)


def _service_time(instance: dict, node: int) -> float:
    return float(instance["service_times"][node])


def _demand(instance: dict, node: int) -> float:
    return float(instance["demands"][node])


def build_inference_tensors(
    instance: dict,
    current_node: int,
    current_time: float,
    current_load: float,
    served_clients: Set[int],
    candidate_clients: Sequence[int],
    vehicle_idx: int = 0,
    step_in_vehicle: int = 0,
) -> Dict[str, torch.Tensor]:
    """Construit les tenseurs d'inférence au format attendu par model.py.

    Les features dynamic_global sont alignées sur build_decision_dataset.py :
        [0] current_time / T
        [1] current_load / Q
        [2] len(served_global) / n
        [3] len(served_in_route) / n  = step_in_vehicle / n
        [4] vehicle_id / n_vehicles   = (vehicle_idx + 1) / n_vehicles
        [5] step_in_vehicle / (n + 1)
    """
    n = int(instance["n"])
    N = n + 1
    Q = float(instance["capacity"])
    T = float(instance["horizon"])
    n_vehicles = int(instance["n_vehicles"])

    node_features = compute_static_node_features(instance)
    served_mask = np.zeros(N, dtype=np.float32)
    for c in served_clients:
        served_mask[c] = 1.0

    feasible_mask = np.zeros(N, dtype=np.float32)
    for c in candidate_clients:
        feasible_mask[c] = 1.0

    # action_features alignées sur build_decision_dataset.py :
    # [0] distance normalisée
    # [1] slack temporel (remplace travel_time redondant)
    # [2] ouverture TW absolue
    # [3] fermeture TW absolue
    # [4] demande normalisée
    action_features = np.zeros((N, 5), dtype=np.float32)
    dist_max = float(instance["dist"].max()) + 1e-6
    svc_current = _service_time(instance, current_node)
    for node in range(N):
        tt = _travel_time(instance, current_node, node)
        a_i, b_i = _time_window(instance, node)

        action_features[node, 0] = _travel_cost(instance, current_node, node) / dist_max

        arrival = current_time + svc_current + tt
        service_start = max(arrival, a_i)
        slack = max(0.0, b_i - service_start)
        action_features[node, 1] = slack / T

        action_features[node, 2] = a_i / T
        action_features[node, 3] = b_i / T
        action_features[node, 4] = 0.0 if node == 0 else _demand(instance, node) / Q

    dynamic_global = np.array(
        [
            current_time / T,
            current_load / Q,
            len(served_clients) / max(1, n),
            step_in_vehicle / max(1, n),
            (vehicle_idx + 1) / max(1, n_vehicles),
            step_in_vehicle / max(1, n + 1),
        ],
        dtype=np.float32,
    )

    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "action_features": torch.tensor(action_features, dtype=torch.float32),
        "current_node": torch.tensor(current_node, dtype=torch.long),
        "dynamic_global": torch.tensor(dynamic_global, dtype=torch.float32),
        "feasible_mask": torch.tensor(feasible_mask, dtype=torch.float32),
        "served_mask": torch.tensor(served_mask, dtype=torch.float32),
    }


class TorchModelScorer:
    """Adaptateur pour utiliser le modèle PyTorch comme scoreur de decoder.py."""

    def __init__(self, model: VRPTWAttentionModel, device: Optional[str] = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        instance: dict,
        current_node: int,
        current_time: float,
        current_load: float,
        served_clients: Set[int],
        candidate_clients: Sequence[int],
        vehicle_idx: int = 0,
        step_in_vehicle: int = 0,
    ) -> Dict[int, float]:
        if not candidate_clients:
            return {}

        tensors = build_inference_tensors(
            instance=instance,
            current_node=current_node,
            current_time=current_time,
            current_load=current_load,
            served_clients=served_clients,
            candidate_clients=candidate_clients,
            vehicle_idx=vehicle_idx,
            step_in_vehicle=step_in_vehicle,
        )
        logits = self.model(
            node_features=tensors["node_features"].to(self.device),
            action_features=tensors["action_features"].to(self.device),
            current_node=tensors["current_node"].to(self.device),
            dynamic_global=tensors["dynamic_global"].to(self.device),
            feasible_mask=tensors["feasible_mask"].to(self.device),
        )
        logits_np = logits.detach().cpu().numpy()
        return {int(c): float(logits_np[c]) for c in candidate_clients}


def load_trained_model(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> VRPTWAttentionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    kwargs = checkpoint["model_kwargs"]
    model = VRPTWAttentionModel(
        node_dim=kwargs["node_dim"],
        action_dim=kwargs["action_dim"],
        global_dim=kwargs["global_dim"],
        d_model=kwargs.get("d_model", 128),
        n_heads=kwargs.get("n_heads", 8),
        n_layers=kwargs.get("n_layers", 3),
        dropout=kwargs.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def solve_with_heuristic(instance: dict) -> Dict[str, object]:
    result = decode_vrptw(instance, scorer=nearest_feasible_scorer)
    checks = verify_solution(instance, result)
    return {"result": result, "checks": checks}


def solve_with_model(instance: dict, checkpoint_path: str, device: Optional[str] = None) -> Dict[str, object]:
    model = load_trained_model(checkpoint_path=checkpoint_path, device=device)
    scorer = TorchModelScorer(model=model, device=device)
    result = decode_vrptw(instance, scorer=scorer)
    checks = verify_solution(instance, result)
    return {"result": result, "checks": checks}


def benchmark_heuristic_vs_model(
    n: int,
    seeds: Sequence[int],
    checkpoint_path: str,
    device: Optional[str] = None,
    oracle_time_limit: float = 5.0,
) -> Dict[str, object]:
    model = load_trained_model(checkpoint_path=checkpoint_path, device=device)
    scorer = TorchModelScorer(model=model, device=device)

    rows: List[BenchmarkItem] = []
    heuristic_ok = 0
    model_ok = 0
    heuristic_served_total = 0
    model_served_total = 0
    heuristic_costs: List[float] = []
    model_costs: List[float] = []
    gaps: List[float] = []

    for seed in seeds:
        instance = generate_instance(n=n, seed=int(seed))

        # Heuristique (avec réparation pour comparaison équitable)
        heur_result = decode_vrptw_with_repair(instance, scorer=nearest_feasible_scorer)
        heur_checks = verify_solution(instance, heur_result)

        # Modèle AM + réparation
        model_result = decode_vrptw_with_repair(instance, scorer=scorer)
        model_checks = verify_solution(instance, model_result)

        # Oracle OR-Tools (référence de qualité)
        oracle_sol = resoudre_instance(instance, time_limit_s=oracle_time_limit)
        oracle_cost: Optional[float] = float(oracle_sol["cout"]) if oracle_sol and oracle_sol["faisable"] else None

        # Gap d'optimalité (uniquement si modèle et oracle faisables)
        gap: Optional[float] = None
        if model_checks["feasible"] and oracle_cost is not None:
            gap = (float(model_result.total_cost) - oracle_cost) / max(1e-6, oracle_cost)
            gaps.append(gap)

        heuristic_ok += int(heur_checks["feasible"])
        model_ok += int(model_checks["feasible"])
        heuristic_served_total += len(heur_result.served_clients)
        model_served_total += len(model_result.served_clients)

        if heur_checks["feasible"]:
            heuristic_costs.append(float(heur_result.total_cost))
        if model_checks["feasible"]:
            model_costs.append(float(model_result.total_cost))

        rows.append(
            BenchmarkItem(
                seed=int(seed),
                heuristic_feasible=bool(heur_checks["feasible"]),
                model_feasible=bool(model_checks["feasible"]),
                heuristic_cost=float(heur_result.total_cost),
                model_cost=float(model_result.total_cost),
                heuristic_served=len(heur_result.served_clients),
                model_served=len(model_result.served_clients),
                oracle_cost=oracle_cost,
                optimality_gap=gap,
            )
        )

    summary = {
        "n_instances": len(seeds),
        "heuristic_feasible_rate": heuristic_ok / max(1, len(seeds)),
        "model_feasible_rate": model_ok / max(1, len(seeds)),
        "heuristic_avg_served": heuristic_served_total / max(1, len(seeds)),
        "model_avg_served": model_served_total / max(1, len(seeds)),
        "heuristic_avg_cost_on_feasible": float(np.mean(heuristic_costs)) if heuristic_costs else None,
        "model_avg_cost_on_feasible": float(np.mean(model_costs)) if model_costs else None,
        "avg_optimality_gap_pct": float(np.mean(gaps) * 100) if gaps else None,
        "details": [asdict(r) for r in rows],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark heuristique vs modèle sur decoder.py")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("artifacts_train_small", "best_model.pt"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=10000,
                        help="Début des seeds de test — doit être hors des seeds d'entraînement (0..999)")
    parser.add_argument("--n_instances", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_instances))
    summary = benchmark_heuristic_vs_model(
        n=args.n,
        seeds=seeds,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print("=== Benchmark heuristique vs modèle ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2, ensure_ascii=False))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Résultats sauvegardés : {args.save_json}")


if __name__ == "__main__":
    main()
