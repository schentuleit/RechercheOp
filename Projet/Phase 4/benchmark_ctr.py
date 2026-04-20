"""
benchmark_ctr.py — Compare Cluster-then-Route : modele n10 vs modele mixte
===========================================================================
Test sur n=1000, 3 seeds, cluster_size=50, beam_width=10
"""
import time
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from preprocess import generate_instance
from large_instance_solver import solve_large_instance
from inference_decoder import load_trained_model, TorchModelScorer

SEEDS        = [42, 123, 999]
N            = 1000
CLUSTER_SIZE = 50
BEAM_WIDTH   = 10

print("Chargement des modeles...")
scorer_n10   = TorchModelScorer(model=load_trained_model("artifacts_train_small/best_model.pt"))
scorer_mixed = TorchModelScorer(model=load_trained_model("artifacts_train_small/best_model_mixed.pt"))
print("OK\n")

results_n10   = []
results_mixed = []

for seed in SEEDS:
    print(f"=== Seed {seed} | n={N} ===")
    inst = generate_instance(n=N, seed=seed)

    t0 = time.time()
    r10 = solve_large_instance(inst, scorer_n10, cluster_size=CLUSTER_SIZE, beam_width=BEAM_WIDTH, verbose=False)
    t10 = time.time() - t0

    t0 = time.time()
    rmx = solve_large_instance(inst, scorer_mixed, cluster_size=CLUSTER_SIZE, beam_width=BEAM_WIDTH, verbose=False)
    tmx = time.time() - t0

    print(f"  Modele n10   : cout={r10.total_cost:.1f}  servis={len(r10.served_clients)}/{N}  t={t10:.1f}s")
    print(f"  Modele mixed : cout={rmx.total_cost:.1f}  servis={len(rmx.served_clients)}/{N}  t={tmx:.1f}s")
    delta = 100 * (rmx.total_cost - r10.total_cost) / r10.total_cost
    print(f"  Delta mixed vs n10 : {delta:+.1f}%\n")

    results_n10.append({"seed": seed, "cost": r10.total_cost, "served": len(r10.served_clients), "time": round(t10, 1)})
    results_mixed.append({"seed": seed, "cost": rmx.total_cost, "served": len(rmx.served_clients), "time": round(tmx, 1)})

# ── Résumé ────────────────────────────────────────────────────────────────────
mean_n10   = np.mean([r["cost"] for r in results_n10])
mean_mixed = np.mean([r["cost"] for r in results_mixed])
delta_mean = 100 * (mean_mixed - mean_n10) / mean_n10

print("=" * 55)
print(f"  Cluster-then-Route n={N} — moyenne sur {len(SEEDS)} seeds")
print(f"  Modele n10   : {mean_n10:.1f}")
print(f"  Modele mixed : {mean_mixed:.1f}")
print(f"  Delta        : {delta_mean:+.1f}%")
print("=" * 55)

with open("artifacts_train_small/benchmark_ctr.json", "w") as f:
    json.dump({"n10": results_n10, "mixed": results_mixed,
               "mean_n10": mean_n10, "mean_mixed": mean_mixed,
               "delta_pct": delta_mean}, f, indent=2)
print("Resultats sauvegardes : artifacts_train_small/benchmark_ctr.json")
