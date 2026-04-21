"""Génère le dataset de décisions oracle pour n=50."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from build_decision_dataset import build_decision_dataset, save_decision_dataset, summarize_decision_dataset

print("=== Génération dataset n=50 (1000 instances) ===")
print("Progression toutes les 25 instances...\n")

dataset = build_decision_dataset(
    n=50,
    n_instances=1000,
    seed_offset=0,
    time_limit_s=10.0,   # plus de temps pour n=50
    include_return_to_depot=False,
    verbose=True,
)

save_decision_dataset(dataset, "decision_dataset_n50.npy")

print("\n=== Résumé final ===")
print(summarize_decision_dataset(dataset))
print("\nFichier sauvegardé : decision_dataset_n50.npy")
