# Projet RO CESI - Pipeline

## Problème

**VRPTW** (Vehicle Routing Problem with Time Windows) - TSP enrichi : tournée à distance/durée minimale avec retour au dépôt.

**Contraintes retenues :** fenêtres temporelles + multi-véhicules avec capacité  
**Complexité :** NP-difficile - $(n-1)!/2$ tournées possibles

---

## Phase 1 - Modélisation + Générateur d'instances

> **Livrable :** check

- Formaliser $G = (V, E, w)$
- Définir les contraintes VRPTW formellement
- Démontrer la complexité NP (réduction depuis hamiltonien)
- Coder le générateur Python : graphes aléatoires, coordonnées euclidiennes, fenêtres temporelles, capacités
- Sortie : instances standardisées (numpy/JSON) - réutilisées par **toutes** les phases suivantes
- Justifier les paramètres des instances (taille, densité du graphe, bornes des time windows, capacité)

---

## Phase 2 - Méthode 1 : NNH + 2-opt

> **Livrable :** final partie 2

- Nearest Neighbor Heuristic : glouton, $O(n^2)$
- Amélioration locale 2-opt
- Gérer les contraintes VRPTW (pénalités ou rejet)
- Enregistrer : solution, coût, temps de calcul
- Sert de **baseline** de qualité pour comparer tout le reste
- ~20–25 % sous-optimal sans 2-opt, ~5–10 % avec

---

## Phase 3 - Méthode 2 : Recuit Simulé

> **Livrable :** final partie 2

- Simulated Annealing avec perturbations 2-opt
- Hyperparamètres à étudier : $T_0$, $\alpha$, nombre d'itérations
- Gérer les contraintes via pénalités dans la fonction objectif
- Meilleure méthode classique → référence de performance pour le DL
- ~1–5 % sous-optimal
- Temps de calcul croît avec $n$ → point clé de comparaison vs DL

---

## Phase 4 - Méthode 3 : Deep Learning (RL)

> **Livrable :** final - étude expérimentale

- **Approche :** Reinforcement Learning
- **Environnement :** état = villes restantes + heure courante + capacité restante
- **Action :** choisir la prochaine ville à visiter
- **Récompense :** $-\text{distance totale}$ (+ pénalité si contrainte violée)
- **Modèle :** MLP pour commencer, puis réseau d'attention si le temps le permet
- Entraîné sur les instances du générateur phase 1
- **Référence :** Kool et al. 2019 - *Attention, Learn to Solve Routing Problems*
- **Important :** commencer dès la fin de la phase 2, pas en dernière semaine
- **Avantage clé :** temps d'inférence constant (ms) quelle que soit la taille

---

## Phase 5 - Étude expérimentale statistique (3 méthodes)

> **Livrable :** final - plan d'expérience

- **Instances :** $n = 10, 20, 50, 100, 200$ villes
- **Métriques :** qualité de solution (% vs optimal), temps de calcul, écart-type, comportement sous contraintes
- **Argument fort :** inférence DL constante vs recuit exponentiel sur grands $n$
- Courbe d'apprentissage DL à inclure
- Conclusions + perspectives d'amélioration

---

## Livrables CESI

| Livrable | Contenu |
|---|---|
| Livrable check | Phase 1 - modélisation + complexité, pas de code résolution |
| Livrable final | Phases 2-3-4-5 - notebook Jupyter, storytelling, PEP8 |
| Soutenance | Démo sur petites instances, résultats comparatifs, planning |

---

## Outils

| Outil | Usage |
|---|---|
| `networkx` | Graphes |
| `numpy` / `matplotlib` | Calcul + visualisation |
| `scipy` | Optimisation |
| `torch` + `gymnasium` | RL - phase 4 |

---

## Notes

- Le générateur phase 1 doit être propre et réutilisable dès le début - socle de tout
- DL = 3e méthode de résolution, répond à l'exigence "au moins 2" en forçant la note
- Comparaison 3 méthodes = cœur de l'étude expérimentale et argument soutenance
