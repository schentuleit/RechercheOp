"""
benchmark_compare.py — Compare modele n10 vs n20 vs mixed (n10+n20+n50)
========================================================================
Test sur 15 seeds par taille, beam W=10
Tailles : n=10, 20, 50, 100, 200
Oracle sur n=10 et n=20 uniquement (rapide)
"""

import json
import time
import numpy as np

from preprocess import generate_instance
from decoder import decode_vrptw_beam_search
from inference_decoder import load_trained_model, TorchModelScorer
from oracle import resoudre_instance

SEEDS      = list(range(5000, 5015))
BEAM_WIDTH = 10
SIZES      = [10, 20, 50, 100, 200]
ORACLE_MAX = 20

print("Chargement des modeles...")
model_n10    = load_trained_model("artifacts_train_small/best_model.pt")
model_n20    = load_trained_model("artifacts_train_small/best_model_n20.pt")
model_mixed  = load_trained_model("artifacts_train_small/best_model_mixed.pt")
scorer_n10   = TorchModelScorer(model=model_n10)
scorer_n20   = TorchModelScorer(model=model_n20)
scorer_mixed = TorchModelScorer(model=model_mixed)
print("OK\n")

results = []

for n in SIZES:
    costs_n10, costs_n20, costs_mixed, costs_oracle = [], [], [], []

    for seed in SEEDS:
        inst = generate_instance(n=n, seed=seed)

        costs_n10.append(decode_vrptw_beam_search(inst, scorer=scorer_n10,   beam_width=BEAM_WIDTH).total_cost)
        costs_n20.append(decode_vrptw_beam_search(inst, scorer=scorer_n20,   beam_width=BEAM_WIDTH).total_cost)
        costs_mixed.append(decode_vrptw_beam_search(inst, scorer=scorer_mixed, beam_width=BEAM_WIDTH).total_cost)

        if n <= ORACLE_MAX:
            sol = resoudre_instance(inst, time_limit_s=10.0)
            if sol and sol["faisable"]:
                costs_oracle.append(float(sol["cout"]))

    mean_n10   = np.mean(costs_n10)
    mean_n20   = np.mean(costs_n20)
    mean_mixed = np.mean(costs_mixed)

    if costs_oracle:
        ref = np.mean(costs_oracle)
        g10  = f"+{100*(mean_n10-ref)/ref:.1f}%"
        g20  = f"+{100*(mean_n20-ref)/ref:.1f}%"
        gmix = f"+{100*(mean_mixed-ref)/ref:.1f}%"
        oracle_str = f"{ref:.1f}"
    else:
        g10 = g20 = gmix = "—"
        oracle_str = "—"

    # Delta mixed vs n10
    delta = 100 * (mean_mixed - mean_n10) / mean_n10
    delta_str = f"{delta:+.1f}%"

    results.append({
        "n": n, "oracle": oracle_str,
        "cost_n10": round(mean_n10, 1),   "gap_n10": g10,
        "cost_n20": round(mean_n20, 1),   "gap_n20": g20,
        "cost_mixed": round(mean_mixed, 1), "gap_mixed": gmix,
        "delta_mixed_vs_n10": delta_str,
    })

    print(f"n={n:>4} | Oracle: {oracle_str:>7} | "
          f"n10: {mean_n10:>7.1f} ({g10}) | "
          f"n20: {mean_n20:>7.1f} ({g20}) | "
          f"mixed: {mean_mixed:>7.1f} ({gmix}) | "
          f"Delta: {delta_str}")

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"{'n':>5} | {'Oracle':>7} | {'n10 gap':>10} | {'n20 gap':>10} | {'mixed gap':>10} | {'mixed vs n10':>13}")
print("-" * 85)
for r in results:
    print(f"{r['n']:>5} | {r['oracle']:>7} | "
          f"{r['gap_n10']:>10} | "
          f"{r['gap_n20']:>10} | "
          f"{r['gap_mixed']:>10} | "
          f"{r['delta_mixed_vs_n10']:>13}")

with open("artifacts_train_small/benchmark_compare.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResultats sauvegardes : artifacts_train_small/benchmark_compare.json")
