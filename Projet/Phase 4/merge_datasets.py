"""
Fusionne plusieurs datasets de décisions en un seul fichier.
Stratégie : n10 × 2 + n20 × 1 + n50 × 1
→ rééquilibre n10 (trop petit) sans surpondérer n50
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

from build_decision_dataset import load_decision_dataset, save_decision_dataset

print("Chargement des datasets...")
d10 = load_decision_dataset("decision_dataset_n10.npy")
d20 = load_decision_dataset("decision_dataset_n20.npy")
d50 = load_decision_dataset("decision_dataset_n50.npy")

print(f"  n=10 : {len(d10):>6} décisions")
print(f"  n=20 : {len(d20):>6} décisions")
print(f"  n=50 : {len(d50):>6} décisions")

# Pondération : n10 × 2 pour rééquilibrer
mixed = d10 * 2 + d20 + d50

import random
random.seed(42)
random.shuffle(mixed)

print(f"\nDataset mixte : {len(mixed)} décisions (n10×2 + n20×1 + n50×1)")
save_decision_dataset(mixed, "decision_dataset_mixed.npy")
print("Sauvegardé : decision_dataset_mixed.npy")
