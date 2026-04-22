"""
grands_tests_genetique.py — Étude statistique complète du VRPTW
================================================================
Lance 30 runs × 5 tailles (n=10, 20, 50, 100, 200) et sauvegarde
les résultats dans resultats/resultats_genetique.csv

Usage :
    python grands_tests_genetique.py
"""

import sys
import os
import time
import copy
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# CHEMINS — structure du projet
# ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../grands_tests/
GA_DIR    = os.path.abspath(os.path.join(BASE_DIR, '..'))                     # .../Algo_genetic/
PHASE4    = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'Phase 4'))  # .../Projet/Phase 4/
RECUIT    = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'Algo_Recuit'))    # .../Phase 3/Algo_Recuit/

for p in [GA_DIR, PHASE4, RECUIT]:
    sys.path.insert(0, p)

from preprocess import generate_instance
from genetic_algorithm import algorithme_genetique, verifier_c1

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
TAILLES     = [10, 20]                  # tailles testées (réduit pour test)
N_RUNS      = 2                         # runs par taille (réduit pour test)
RESULTATS_DIR = os.path.join(BASE_DIR, '..', 'resultats')
CSV_PATH    = os.path.join(RESULTATS_DIR, 'resultats_genetique.csv')

# Hyperparamètres de l'algo génétique
# (adaptés à la taille pour que le temps reste raisonnable)
PARAMS_PAR_TAILLE = {
    10 : {'taille_pop': 20,  'n_generations': 30, 'taux_mutation': 0.3},
    20 : {'taille_pop': 20,  'n_generations': 30, 'taux_mutation': 0.3},
}


# ─────────────────────────────────────────────────────────────────
# UTILITAIRES AFFICHAGE
# ─────────────────────────────────────────────────────────────────

def barre(run, total, largeur=30):
    pct   = int(run / total * largeur)
    barre = '█' * pct + '░' * (largeur - pct)
    return f'[{barre}] {run:2d}/{total}'

def separateur(car='═', n=62):
    return car * n


# ─────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────

def lancer_grands_tests():

    os.makedirs(RESULTATS_DIR, exist_ok=True)

    print()
    print(separateur())
    print('  GRANDS TESTS — ALGORITHME GÉNÉTIQUE VRPTW')
    print(f'  {N_RUNS} runs × tailles {TAILLES}')
    print(separateur())

    lignes  = []
    t_total = time.time()

    for n in TAILLES:
        params     = PARAMS_PAR_TAILLE[n]
        n_vehicles = max(3, n // 8)   # formule partagée avec le reste du projet

        print()
        print(separateur('-'))
        print(f'  n = {n} clients | {n_vehicles} véhicules | '
              f'pop={params["taille_pop"]} | '
              f'gen={params["n_generations"]} | '
              f'mut={params["taux_mutation"]}')
        print(separateur('-'))

        succes    = 0
        t_debut_n = time.time()

        for run in range(N_RUNS):
            try:
                # Seed unique par (run, n) → instances différentes à chaque run
                seed_inst = run * 100 + n
                seed_algo = run

                instance = generate_instance(
                    n          = n,
                    n_vehicles = n_vehicles,
                    seed       = seed_inst
                )

                t0 = time.time()
                solution, cout, historique = algorithme_genetique(
                    instance,
                    seed = seed_algo,
                    **params
                )
                duree = time.time() - t0

                # Métriques
                nb_vehicules = len([r for r in solution if len(r) > 0])
                valide_c1, _, _ = verifier_c1(solution, instance['n'])
                cout_initial = historique['couts_generations'][0]
                cout_final   = historique['couts_generations'][-1]
                amelioration = (cout_initial - cout_final) / cout_initial * 100 \
                               if cout_initial > 0 else 0.0

                lignes.append({
                    'n_clients'        : n,
                    'run'              : run + 1,
                    'cout'             : round(cout, 4),
                    'temps_s'          : round(duree, 4),
                    'n_vehicules'      : nb_vehicules,
                    'cout_initial'     : round(cout_initial, 4),
                    'amelioration_pct' : round(amelioration, 2),
                    'c1_respectee'     : int(valide_c1),
                    'seed_instance'    : seed_inst,
                    'seed_algo'        : seed_algo,
                })
                succes += 1

                # Affichage barre de progression
                print(f'\r  {barre(run+1, N_RUNS)} '
                      f'| coût={cout:9.2f} '
                      f'| t={duree:5.1f}s '
                      f'| véh={nb_vehicules}'
                      f'| C1={"✅" if valide_c1 else "❌"}',
                      end='', flush=True)

            except Exception as e:
                print(f'\n  ⚠️  Run {run+1} échoué : {e}')

        t_fin_n = time.time()
        print(f'\n  ✅  {succes}/{N_RUNS} runs réussis '
              f'— durée totale : {t_fin_n - t_debut_n:.1f}s')

        # Résumé rapide pour cette taille
        sous_df = pd.DataFrame([l for l in lignes if l['n_clients'] == n])
        if not sous_df.empty:
            print(f'     Coût   → moy={sous_df["cout"].mean():.2f} '
                  f'± {sous_df["cout"].std():.2f}')
            print(f'     Temps  → moy={sous_df["temps_s"].mean():.2f}s '
                  f'± {sous_df["temps_s"].std():.2f}s')
            print(f'     C1 OK  → {int(sous_df["c1_respectee"].mean()*100)}% des runs')

    # ── Sauvegarde CSV ─────────────────────────────────────────────
    df = pd.DataFrame(lignes)
    df.to_csv(CSV_PATH, index=False)

    print()
    print(separateur())
    print('  RÉSUMÉ GLOBAL')
    print(separateur())
    print(f'  Durée totale  : {time.time() - t_total:.1f}s')
    print(f'  Lignes totales: {len(df)}')
    print(f'  CSV sauvegardé: {CSV_PATH}')
    print(separateur())

    # ── Tableau récapitulatif dans le terminal ─────────────────────
    print()
    print(f'  {"n":>6} | {"Coût moy":>10} | {"Coût std":>10} | '
          f'{"Temps moy":>10} | {"Véh moy":>8} | {"C1 (%)":>7}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*8}-+-{"-"*7}')

    for n in TAILLES:
        sub = df[df['n_clients'] == n]
        if sub.empty:
            continue
        print(f'  {n:>6} | '
              f'{sub["cout"].mean():>10.2f} | '
              f'{sub["cout"].std():>10.2f} | '
              f'{sub["temps_s"].mean():>10.3f} | '
              f'{sub["n_vehicules"].mean():>8.1f} | '
              f'{sub["c1_respectee"].mean()*100:>6.0f}%')

    print(separateur())
    print()
    print('  ✅  Lance maintenant analyse_statistique_genetique.ipynb')
    print('      pour visualiser les résultats.')
    print()

    return df


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    lancer_grands_tests()
