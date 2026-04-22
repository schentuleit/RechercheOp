"""
generate_benchmark.py — Génération et résolution des 150 instances de benchmark
================================================================================
Génère 30 instances par taille (n=10, 20, 50, 100, 200), seeds 0..29.

Structure produite
------------------
Benchmark/
  instances/
    n10/instance_n10_seed0.json  ...  instance_n10_seed29.json
    n20/ ...
    n50/ ...
    n100/ ...
    n200/ ...
  results/
    oracle_results.json      — coûts OR-Tools par instance
    am_results.json          — coûts Attention Model par instance
    benchmark_summary.csv    — tableau récapitulatif (à compléter par les autres algos)
    benchmark_summary.json   — même données en JSON

Convention seeds
----------------
Seeds 0..29 pour toutes les tailles — continus, faciles à utiliser depuis
n'importe quel algorithme externe.

Pour charger une instance depuis un autre algorithme :
    with open("instances/n50/instance_n50_seed12.json") as f:
        data = json.load(f)
    # data contient : n, n_vehicles, coords, dist, durees, demands,
    #                 capacity, time_windows, service_times, horizon
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Chemins ──────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
PHASE4_DIR = HERE.parent / "Phase 4"
sys.path.insert(0, str(PHASE4_DIR))

from preprocess import generate_instance, sauvegarder_instance
from oracle import resoudre_instance

# ── Paramètres ───────────────────────────────────────────────────────────────
SIZES      = [10, 20, 50, 100, 200]
N_SEEDS    = 30          # seeds 0..29
SEEDS      = list(range(N_SEEDS))

# Temps OR-Tools par taille (en secondes)
ORACLE_TIME: Dict[int, float] = {
    10:  5,
    20:  10,
    50:  30,
    100: 60,
    200: 120,
}

INSTANCES_DIR = HERE / "instances"
RESULTS_DIR   = HERE / "results"


# ── Utilitaires ──────────────────────────────────────────────────────────────

def instance_path(n: int, seed: int) -> Path:
    return INSTANCES_DIR / f"n{n}" / f"instance_n{n}_seed{seed}.json"


def load_instance(n: int, seed: int) -> dict:
    path = instance_path(n, seed)
    cles_numpy = {"coords", "dist", "durees", "demands", "time_windows", "service_times"}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: np.array(v) if k in cles_numpy else v for k, v in data.items()}


# ── Étape 1 : Génération des instances ───────────────────────────────────────

def generate_all_instances(overwrite: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  ETAPE 1 — Generation des 150 instances")
    print("=" * 60)

    total = len(SIZES) * N_SEEDS
    done = 0

    for n in SIZES:
        n_vehicles = max(3, math.ceil(n / 10))
        for seed in SEEDS:
            path = instance_path(n, seed)
            if path.exists() and not overwrite:
                done += 1
                continue

            instance = generate_instance(n=n, n_vehicles=n_vehicles, seed=seed)
            sauvegarder_instance(instance, str(path))
            done += 1

        print(f"  n={n:>3} : {N_SEEDS} instances OK  ({done}/{total})")

    print(f"\n  Total : {done} instances sauvegardees dans instances/")


# ── Étape 2 : Résolution OR-Tools ────────────────────────────────────────────

def solve_all_oracle(overwrite: bool = False) -> Dict[str, object]:
    oracle_path = RESULTS_DIR / "oracle_results.json"

    # Charger résultats existants
    if oracle_path.exists() and not overwrite:
        with open(oracle_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    print("\n" + "=" * 60)
    print("  ETAPE 2 — Resolution OR-Tools (oracle)")
    print("=" * 60)

    total = len(SIZES) * N_SEEDS
    done  = 0
    ok    = 0

    for n in SIZES:
        tl = ORACLE_TIME[n]
        n_ok = 0
        t0 = time.time()

        for seed in SEEDS:
            key = f"n{n}_seed{seed}"
            if key in results and not overwrite:
                done += 1
                if results[key]["feasible"]:
                    n_ok += 1
                    ok   += 1
                continue

            instance = load_instance(n, seed)
            sol = resoudre_instance(instance, time_limit_s=tl)

            if sol and sol["faisable"]:
                results[key] = {
                    "n": n, "seed": seed,
                    "feasible": True,
                    "cost": float(sol["cout"]),
                    "n_clients_servis": int(sol["n_clients_servis"]),
                    "time_limit_s": tl,
                }
                n_ok += 1
                ok   += 1
            else:
                results[key] = {
                    "n": n, "seed": seed,
                    "feasible": False,
                    "cost": None,
                    "n_clients_servis": 0,
                    "time_limit_s": tl,
                }
            done += 1

            # Sauvegarde incrémentale (si interruption)
            with open(oracle_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"  n={n:>3} : {n_ok}/{N_SEEDS} faisables  "
              f"({elapsed:.0f}s, ~{tl}s/instance)")

    print(f"\n  Total oracle : {ok}/{total} solutions faisables")
    return results


# ── Étape 3 : Résolution AM ──────────────────────────────────────────────────

def solve_all_am(overwrite: bool = False) -> Dict[str, object]:
    am_path     = RESULTS_DIR / "am_results.json"
    checkpoint  = PHASE4_DIR / "artifacts_train_small" / "best_model.pt"

    if not checkpoint.exists():
        print("\n  [SKIP] Modele AM non trouve :", checkpoint)
        return {}

    # Import AM
    from decoder import decode_vrptw_beam_search, verify_solution, nearest_feasible_scorer
    from inference_decoder import load_trained_model, TorchModelScorer

    model  = load_trained_model(str(checkpoint))
    scorer = TorchModelScorer(model=model)
    BEAM_WIDTH = 20

    if am_path.exists() and not overwrite:
        with open(am_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    print("\n" + "=" * 60)
    print("  ETAPE 3 — Resolution Attention Model")
    print("=" * 60)

    for n in SIZES:
        n_ok = 0
        t0 = time.time()

        for seed in SEEDS:
            key = f"n{n}_seed{seed}"
            if key in results and not overwrite:
                if results[key]["feasible"]:
                    n_ok += 1
                continue

            instance = load_instance(n, seed)
            result   = decode_vrptw_beam_search(instance, scorer=scorer, beam_width=BEAM_WIDTH)
            checks   = verify_solution(instance, result)

            results[key] = {
                "n": n, "seed": seed,
                "feasible": bool(checks["feasible"]),
                "cost": float(result.total_cost) if checks["feasible"] else None,
                "n_clients_servis": len(result.served_clients),
            }
            if checks["feasible"]:
                n_ok += 1

        with open(am_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"  n={n:>3} : {n_ok}/{N_SEEDS} faisables  ({elapsed:.1f}s)")

    return results


# ── Étape 4 : Tableau récapitulatif ─────────────────────────────────────────

def build_summary(oracle: Dict, am: Dict) -> None:
    print("\n" + "=" * 60)
    print("  ETAPE 4 — Tableau recapitulatif")
    print("=" * 60)

    rows = []
    for n in SIZES:
        for seed in SEEDS:
            key = f"n{n}_seed{seed}"
            orc = oracle.get(key, {})
            am_r = am.get(key, {})

            oracle_cost = orc.get("cost")
            am_cost     = am_r.get("cost")

            gap_am = None
            if oracle_cost and am_cost:
                gap_am = round((am_cost - oracle_cost) / oracle_cost * 100, 2)

            rows.append({
                "n":          n,
                "seed":       seed,
                "instance":   f"instance_n{n}_seed{seed}.json",
                "oracle_cost":   round(oracle_cost, 2) if oracle_cost else None,
                "oracle_feasible": orc.get("feasible", False),
                # Colonnes à remplir par les autres algorithmes
                "nnh_cost":   None,   # A remplir par NNH
                "nnh_gap_pct": None,
                "genetic_cost": None, # A remplir par Algo Génétique
                "genetic_gap_pct": None,
                "sa_cost":    None,   # A remplir par Recuit Simulé
                "sa_gap_pct": None,
                # AM
                "am_cost":    round(am_cost, 2) if am_cost else None,
                "am_feasible": am_r.get("feasible", False),
                "am_gap_pct": gap_am,
            })

    # JSON
    json_path = RESULTS_DIR / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # CSV
    import csv
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Affichage résumé par taille
    print(f"\n  {'n':>5} | {'Oracle OK':>9} | {'AM fais.':>8} | {'Gap AM moy':>10}")
    print("  " + "-" * 42)
    for n in SIZES:
        n_rows = [r for r in rows if r["n"] == n]
        oracle_ok = sum(1 for r in n_rows if r["oracle_feasible"])
        am_ok     = sum(1 for r in n_rows if r["am_feasible"])
        gaps      = [r["am_gap_pct"] for r in n_rows if r["am_gap_pct"] is not None]
        gap_str   = f"{np.mean(gaps):.1f}%" if gaps else "N/A"
        print(f"  {n:>5} | {oracle_ok:>9} | {am_ok:>8} | {gap_str:>10}")

    print(f"\n  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print("\n  Colonnes a remplir par les autres algorithmes :")
    print("    nnh_cost, genetic_cost, sa_cost")
    print("    (les gap_pct se calculent automatiquement si besoin)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Recalculer meme si deja fait")
    parser.add_argument("--skip_oracle", action="store_true",
                        help="Passer l'etape OR-Tools")
    parser.add_argument("--skip_am", action="store_true",
                        help="Passer l'etape AM")
    args = parser.parse_args()

    t_start = time.time()

    generate_all_instances(overwrite=args.overwrite)

    oracle = {}
    if not args.skip_oracle:
        oracle = solve_all_oracle(overwrite=args.overwrite)
    else:
        oracle_path = RESULTS_DIR / "oracle_results.json"
        if oracle_path.exists():
            with open(oracle_path) as f:
                oracle = json.load(f)

    am = {}
    if not args.skip_am:
        am = solve_all_am(overwrite=args.overwrite)
    else:
        am_path = RESULTS_DIR / "am_results.json"
        if am_path.exists():
            with open(am_path) as f:
                am = json.load(f)

    build_summary(oracle, am)

    elapsed = time.time() - t_start
    print(f"\n  Duree totale : {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
