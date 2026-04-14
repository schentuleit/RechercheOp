"""
statistical_analysis.py — Étude statistique du VRPTW (Algorithme Génétique)
=============================================================================
Lance 30 runs pour n=10, 20, 50 clients et produit :
  - Statistiques descriptives (moyenne, écart-type, variance)
  - Matrice de corrélation
  - Boîtes à moustaches
  - Histogrammes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CHEMINS — adapte si ta structure de dossiers est différente
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers ton generate_instance (Phase 4 / preprocess.py)
# statistical_analysis.py est dans Phase 3/Algo_genetic
# Phase 4 est au même niveau que Phase 3 (sibling)
PHASE4_PATH = os.path.join(BASE_DIR, '..', '..', 'Phase 4')
sys.path.insert(0, PHASE4_PATH)

# Chemin vers ton algo génétique
GA_PATH = BASE_DIR  # le script est dans le même dossier que genetic_algorithm.py
sys.path.insert(0, GA_PATH)

# Chemin vers le recuit (nécessaire car genetic_algorithm l'importe)
RECUIT_PATH = os.path.join(BASE_DIR, '..', 'Algo_Recuit')
sys.path.insert(0, RECUIT_PATH)

from preprocess import generate_instance
from genetic_algorithm import algorithme_genetique
from simulated_annealing import verifier_solution_admissible

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION DE L'ÉTUDE
# ─────────────────────────────────────────────────────────────────
N_CLIENTS_LIST = [10, 20, 50]   # tailles testées
N_RUNS         = 30             # runs par taille
N_VEHICLES     = 5              # véhicules disponibles (adapte si besoin)

# Hyperparamètres de l'algo génétique
GA_PARAMS = {
    "taille_pop"   : 30,
    "n_generations": 100,
    "taux_mutation": 0.3,
}

# Dossier de sortie pour les figures
FIGURES_DIR = os.path.join(BASE_DIR, "figures_stats")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — COLLECTE DES DONNÉES (30 runs × 3 tailles)
# ═══════════════════════════════════════════════════════════════════

def collecter_donnees():
    """
    Lance n_runs fois l'algo génétique pour chaque taille n.
    Retourne un dict :
      results[n] = {"costs": [], "times": [], "n_vehicles": []}
    """
    results = {}

    for n in N_CLIENTS_LIST:
        print(f"\n{'='*55}")
        print(f"  TAILLE n = {n} clients — {N_RUNS} runs en cours...")
        print(f"{'='*55}")

        costs, times, n_vehicles = [], [], []
        succes = 0

        for run in range(N_RUNS):
            try:
                # Génère une instance différente à chaque run (seed = run)
                instance = generate_instance(
                    n         = n,
                    n_vehicles = N_VEHICLES,
                    seed      = run * 100 + n  # seed unique par (run, n)
                )

                t0 = time.time()
                solution, cout, historique = algorithme_genetique(
                    instance,
                    seed=run,
                    **GA_PARAMS
                )
                t1 = time.time()

                duree     = t1 - t0
                nb_routes = len([r for r in solution if len(r) > 0])

                costs.append(cout)
                times.append(duree)
                n_vehicles.append(nb_routes)
                succes += 1

                # Barre de progression maison
                pct  = int((run + 1) / N_RUNS * 30)
                barre = "█" * pct + "░" * (30 - pct)
                print(f"\r  [{barre}] Run {run+1:2d}/{N_RUNS} "
                      f"| coût={cout:8.2f} | t={duree:.1f}s | véhicules={nb_routes}",
                      end="", flush=True)

            except Exception as e:
                print(f"\n   Run {run+1} échoué : {e}")

        print(f"\n   {succes}/{N_RUNS} runs réussis pour n={n}")

        results[n] = {
            "costs"     : costs,
            "times"     : times,
            "n_vehicles": n_vehicles
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — STATISTIQUES DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════

def calculer_statistiques(results):
    """
    Calcule et affiche moyenne, écart-type, variance, min, max, médiane
    pour chaque métrique et chaque taille n.
    Retourne un DataFrame récapitulatif.
    """
    lignes = []

    for n, data in results.items():
        for metrique, valeurs in [("Coût", data["costs"]),
                                   ("Temps (s)", data["times"]),
                                   ("Nb véhicules", data["n_vehicles"])]:
            arr = np.array(valeurs)
            lignes.append({
                "n clients" : n,
                "Métrique"  : metrique,
                "Moyenne"   : round(np.mean(arr), 4),
                "Écart-type": round(np.std(arr), 4),
                "Variance"  : round(np.var(arr), 4),
                "Min"       : round(np.min(arr), 4),
                "Max"       : round(np.max(arr), 4),
                "Médiane"   : round(np.median(arr), 4),
            })

    df = pd.DataFrame(lignes)

    print("\n\n" + "═"*70)
    print("   STATISTIQUES DESCRIPTIVES — 30 RUNS")
    print("═"*70)
    print(df.to_string(index=False))

    # Export CSV
    csv_path = os.path.join(BASE_DIR, "statistics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Tableau exporté → {csv_path}")

    return df# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — MATRICE DE CORRÉLATION
# ═══════════════════════════════════════════════════════════════════

def tracer_matrices_correlation(results):
    """
    Pour chaque taille n, affiche la matrice de corrélation
    entre coût, temps et nombre de véhicules.
    """
    fig, axes = plt.subplots(1, len(N_CLIENTS_LIST),
                             figsize=(6 * len(N_CLIENTS_LIST), 5))
    fig.suptitle("Matrices de corrélation par taille de problème",
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, n in zip(axes, N_CLIENTS_LIST):
        df_n = pd.DataFrame({
            "Coût"        : results[n]["costs"],
            "Temps (s)"   : results[n]["times"],
            "Nb véhicules": results[n]["n_vehicles"]
        })
        corr = df_n.corr()

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax,
                    linewidths=0.5, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"n = {n} clients", fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "correlation_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Matrice de corrélation → {path}")


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — BOÎTES À MOUSTACHES (BOXPLOTS)
# ═══════════════════════════════════════════════════════════════════

def tracer_boxplots(results):
    """
    3 boxplots : un par métrique (coût, temps, véhicules).
    Points individuels superposés pour voir chaque run.
    """
    metriques = [
        ("costs",      "Coût total",         "Coût (unité)"),
        ("times",      "Temps de résolution","Durée (secondes)"),
        ("n_vehicles", "Nombre de véhicules","Nombre de véhicules"),
    ]

    palette = {10: "#4C9BE8", 20: "#F4845F", 50: "#56C596"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Boîtes à moustaches — 30 runs par taille",
                 fontsize=14, fontweight='bold')

    for ax, (cle, titre, ylabel) in zip(axes, metriques):
        # Prépare les données pour seaborn
        data_long = []
        for n in N_CLIENTS_LIST:
            for v in results[n][cle]:
                data_long.append({"n clients": f"n={n}", "valeur": v})
        df_long = pd.DataFrame(data_long)

        # Boxplot
        sns.boxplot(data=df_long, x="n clients", y="valeur",
                    palette=[palette[n] for n in N_CLIENTS_LIST],
                    width=0.5, linewidth=1.5, ax=ax,
                    order=[f"n={n}" for n in N_CLIENTS_LIST])

        # Points individuels (stripplot)
        sns.stripplot(data=df_long, x="n clients", y="valeur",
                      color="black", alpha=0.4, size=4, jitter=True, ax=ax,
                      order=[f"n={n}" for n in N_CLIENTS_LIST])

        ax.set_title(titre, fontsize=12, fontweight='bold')
        ax.set_xlabel("Taille du problème", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Boxplots → {path}")


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 5 — HISTOGRAMMES
# ═══════════════════════════════════════════════════════════════════

def tracer_histogrammes(results):
    """
    Grille 3×3 : lignes = tailles n, colonnes = métriques.
    Chaque histogramme montre la distribution + moyenne + médiane.
    """
    metriques = [
        ("costs",      "Coût total"),
        ("times",      "Temps (s)"),
        ("n_vehicles", "Nb véhicules"),
    ]

    couleurs = {10: "#4C9BE8", 20: "#F4845F", 50: "#56C596"}

    fig, axes = plt.subplots(len(N_CLIENTS_LIST), len(metriques),
                             figsize=(15, 10))
    fig.suptitle("Histogrammes — Distribution sur 30 runs",
                 fontsize=14, fontweight='bold')

    for i, n in enumerate(N_CLIENTS_LIST):
        for j, (cle, titre) in enumerate(metriques):
            ax   = axes[i][j]
            vals = np.array(results[n][cle])
            moy  = np.mean(vals)
            med  = np.median(vals)

            # Histogramme avec courbe de densité
            sns.histplot(vals, kde=True, ax=ax,
                         color=couleurs[n], alpha=0.6,
                         line_kws={"linewidth": 2})

            # Lignes moyenne et médiane
            ax.axvline(moy, color='red',  linestyle='--',
                       linewidth=1.8, label=f"Moyenne : {moy:.2f}")
            ax.axvline(med, color='blue', linestyle=':',
                       linewidth=1.8, label=f"Médiane : {med:.2f}")

            ax.legend(fontsize=7)
            ax.set_title(f"{titre} — n={n}", fontsize=10, fontweight='bold')
            ax.set_xlabel("")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "histogrammes.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogrammes → {path}")


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 6 — EXPORT DES DONNÉES BRUTES
# ═══════════════════════════════════════════════════════════════════

def exporter_donnees_brutes(results):
    """
    Exporte toutes les données brutes dans results.csv
    """
    lignes = []
    for n, data in results.items():
        n_valides = data.get("n_valides", 0)
        for run_idx, (c, t, v) in enumerate(zip(
                data["costs"], data["times"], data["n_vehicles"])):
            lignes.append({
                "n_clients"  : n,
                "run"        : run_idx + 1,
                "cout"       : round(c, 4),
                "temps_s"    : round(t, 4),
                "n_vehicules": v
            })

    df = pd.DataFrame(lignes)
    csv_path = os.path.join(BASE_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Données brutes exportées → {csv_path}")
    
    # Afficher le résumé d'admissibilité
    print("\n" + "═"*70)
    print("   RÉSUMÉ DE L'ADMISSIBILITÉ DES SOLUTIONS")
    print("═"*70)
    for n in sorted([k for k in results.keys() if isinstance(k, int)]):
        n_valides = results[n].get("n_valides", 0)
        total = len(results[n]["costs"])
        print(f"  n = {n:2d} clients : {n_valides:2d}/{total} solutions admissibles ({n_valides/total*100:5.1f}%)")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# ÉTAPE 7 — VÉRIFICATION DE L'ADMISSIBILITÉ DES SOLUTIONS
# ═══════════════════════════════════════════════════════════════════

def valider_solutions(results):
    """
    Vérifie l'admissibilité des meilleures solutions pour chaque taille.
    Affiche un rapport des violations de contraintes.
    """
    print("\n" + "═"*70)
    print("   VALIDATION DES CONTRAINTES VRPTW")
    print("═"*70)
    
    # On va vérifier la première et la meilleure solution de chaque taille
    # (on ne peut pas vérifier tous les runs, trop coûteux)
    
    # Note: results[n]["solutions_best"] doit être rempli par collecter_donnees()
    # Pour l'instant, on fait juste un rapport statistique
    
    for n in N_CLIENTS_LIST:
        print(f"\n▶ Taille n = {n} clients")
        print("  Note : Validation sur l'ensemble des 30 solutions")
        print("  (Affichage du nombre de solutions admissibles)")
        print("")


# ═══════════════════════════════════════════════════════════════════
# COLLECT DES DONNÉES AVEC STOCKAGE DES SOLUTIONS
# ═══════════════════════════════════════════════════════════════════

def collecter_donnees_v2():
    """
    Collecte les données avec stockage des meilleures solutions
    pour vérification ultérieure.
    """
    results = {}

    for n in N_CLIENTS_LIST:
        print(f"\n{'='*55}")
        print(f"  TAILLE n = {n} clients — {N_RUNS} runs en cours...")
        print(f"{'='*55}")

        costs, times, n_vehicles = [], [], []
        solutions_best = []
        instances_list = []
        valides = 0
        succes = 0

        for run in range(N_RUNS):
            try:
                # Génère une instance différente à chaque run (seed = run)
                instance = generate_instance(
                    n         = n,
                    n_vehicles = N_VEHICLES,
                    seed      = run * 100 + n  # seed unique par (run, n)
                )
                instances_list.append(instance)

                t0 = time.time()
                solution, cout, historique = algorithme_genetique(
                    instance,
                    seed=run,
                    **GA_PARAMS
                )
                t1 = time.time()

                duree     = t1 - t0
                nb_routes = len([r for r in solution if len(r) > 0])

                # Vérifier l'admissibilité
                admissible, _ = verifier_solution_admissible(solution, instance)
                if admissible:
                    valides += 1

                costs.append(cout)
                times.append(duree)
                n_vehicles.append(nb_routes)
                solutions_best.append(solution)
                succes += 1

                # Barre de progression maison
                pct  = int((run + 1) / N_RUNS * 30)
                barre = "█" * pct + "░" * (30 - pct)
                status = "✓" if admissible else "⚠"
                print(f"\r  [{barre}] Run {run+1:2d}/{N_RUNS} "
                      f"| coût={cout:8.2f} | t={duree:.1f}s | {status}",
                      end="", flush=True)

            except Exception as e:
                print(f"\n  ⚠️  Run {run+1} échoué : {e}")

        print(f"\n  ✅  {succes}/{N_RUNS} runs réussis | {valides} solutions admissibles ({valides/succes*100:.1f}%)")

        results[n] = {
            "costs"     : costs,
            "times"     : times,
            "n_vehicles": n_vehicles,
            "solutions" : solutions_best,
            "instances" : instances_list,
            "n_valides" : valides
        }

    return results


def main():
    print("\n" + "╔" + "═"*53 + "╗")
    print("║   ÉTUDE STATISTIQUE VRPTW — ALGORITHME GÉNÉTIQUE   ║")
    print("║         30 runs × 3 tailles (n=10, 20, 50)         ║")
    print("╚" + "═"*53 + "╝")

    # 1. Collecte (version améliorée)
    results = collecter_donnees_v2()

    # 2. Stats descriptives
    df_stats = calculer_statistiques(results)

    # 3. Corrélation
    print("\n  Génération des matrices de corrélation...")
    tracer_matrices_correlation(results)

    # 4. Boxplots
    print("\n  Génération des boîtes à moustaches...")
    tracer_boxplots(results)

    # 5. Histogrammes
    print("\n  Génération des histogrammes...")
    tracer_histogrammes(results)

    # 6. Export brut
    exporter_donnees_brutes(results)

    print("\n\n" + "═"*55)
    print("    ANALYSE TERMINÉE")
    print(f"   Figures sauvegardées dans : {FIGURES_DIR}")
    print(f"   results.csv + statistics_summary.csv créés")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()