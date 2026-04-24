# NOTES DE RÉVISION — PROJET VRPTW (usage interne)

## STRUCTURE DU PROJET
- Phase 1 : Modélisation VRPTW + générateur instances (socle commun)
- Phase 2 : NNH + 2-opt (heuristique baseline, O(n²))
- Phase 3 : Recuit Simulé + Algo Génétique (métaheuristiques)
- Phase 4 : Deep Learning Attention Model (RL, inférence O(n))
- Benchmark : 150 instances, 5 tailles (n=10,20,50,100,200), 30 seeds, oracle OR-Tools

## VRPTW CONTRAINTES
- C1 : chaque client visité exactement 1 fois
- C2 : routes commencent/finissent au dépôt
- C3 : capacité ≤ Q
- C4 : arrivée dans fenêtre [a_i, b_i]
- C5 : retour dépôt ≤ H=480 min

## PARAMÈTRES INSTANCES (preprocess.py v3)
- n_vehicles = max(3, ceil(n/10))
- capacity = 1.4 × (total_demand / n_vehicles)
- TW largeur = U(60,90) min, ancrées sur t_min depuis dépôt
- horizon = 480 min, vitesse = 0.6 min/unité

## RÉSULTATS BENCHMARK (gap vs oracle OR-Tools)
NNH : n10=31%, n20=49%, n50=63%, n100=70%, n200=65%
Recuit : n10≈0.3%, n20≈19.5%, n50≈36.8%, n100≈78%, n200≈156%
Génétique : n10≈16%, n20≈36.6%, n50≈90.8%, n100≈123%, n200≈192%
AM : n10≈14.9%, n20≈22.5%, n50≈37.8%, n100≈41.1%, n200≈39.9%

## TEMPS D'EXÉCUTION (médian)
NNH: <0.02s pour tout n
Recuit: 1s (n10) → 10.5s (n200)
Génétique: 0.35s (n10) → 20.7s (n200)
AM: 32ms (n10) → 821ms (n200)

## VALIDITÉ
NNH: 100%, Recuit: ~92%, Génétique: ~85%, AM: 68% (100% avec CTR repair)

## ARCHITECTURE AM (Phase 4)
- Encodeur Transformer statique (3 couches, dim=128) : 1 passe par instance
- Décodeur contextuel par étape : query=contexte+global, keys=noeuds, logits masqués
- Entraînement : REINFORCE (policy gradient, Williams 1992)
- Repair : Cluster-Then-Route (CTR) si infaisable

## NNH ALGO
- Gloutonné constructif : toujours choisir client le plus proche faisable
- Complexité : O(n²) par véhicule
- Limites : pas de retour arrière, pas d'inter-routes

## 2-OPT
- Amélioration locale intra-route
- Δ(i,j) = c(r[i-1],r[i]) + c(r[j],r[j+1]) - c(r[i-1],r[j]) - c(r[i],r[j+1])
- Accepter si Δ>0 ET route inversée valide (C3,C4,C5)
- Gain moyen : 0.06–0.96% (limité par TW)

## RECUIT SIMULÉ
- Accepte solutions dégradées avec proba exp(-Δ/T) (exploration)
- Soft constraints : pénalités λ_C=1000, λ_T=500
- Refroidissement : T ← T × α (α ≈ 0.995-0.999)
- Meilleure méthode pour n≤50

## ALGO GÉNÉTIQUE
- Population → sélection tournoi → PMX crossover → mutation → élitisme
- Diverge sur grandes instances (n≥50)
- Taux mutation 0.2–0.3 selon taille

## COMPLEXITÉ NP-DIFFICULTÉ
- Réduction : Circuit Hamiltonien → TSP → VRPTW
- Karp 1972 : 21 problèmes NP-complets
- n=50 : 50! ≈ 3×10^64 tournées → âge univers ×10^42 à 10^9 op/s
