"""
genetic_algorithm.py — Algorithme Génétique pour le VRPTW
===========================================================
Simple et lisible, basé sur les fonctions du recuit simulé.

Contraintes VRPTW respectées :
  C1 — Couverture         : vérifiée + corrigée après chaque croisement
  C2 — Retour dépôt       : par construction (liste implicite)
  C3 — Capacité           : soft constraint via cout_penalise()
  C4 — Fenêtres de temps  : soft constraint via cout_penalise()
  C5 — Cohérence temps    : via cout_penalise()
  C6 — Sous-cycles        : par construction (structure en liste)
"""

import numpy as np
import time
import copy
import sys
import os

# Ajouter le répertoire frère Algo_Recuit au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Algo_Recuit')))

from simulated_annealing import solution_initiale, cout_penalise, est_valide


# ═══════════════════════════════════════════════════════════════════
# UTILITAIRE C1 — Vérification et réparation de la couverture
# ═══════════════════════════════════════════════════════════════════

def verifier_c1(routes, n):
    """
    Vérifie la contrainte C1 : chaque client de 1 à n apparaît
    exactement une fois dans l'ensemble des routes.

    @param routes : list of lists — la solution à vérifier
    @param n      : int — nombre de clients (hors dépôt)
    @return : (bool, set, set)
              - True si C1 est respectée
              - ensemble des clients en double
              - ensemble des clients manquants
    """
    tous_clients = [client for route in routes for client in route]
    clients_attendus = set(range(1, n + 1))

    doublons   = set(c for c in tous_clients if tous_clients.count(c) > 1)
    manquants  = clients_attendus - set(tous_clients)

    valide = (len(doublons) == 0 and len(manquants) == 0)
    return valide, doublons, manquants


def reparer_c1(routes, n):
    """
    Répare la contrainte C1 si elle est violée après un croisement :
      1. Supprime les doublons (garde la première occurrence)
      2. Réinsère les clients manquants dans la route la moins chargée

    @param routes : list of lists — solution potentiellement invalide
    @param n      : int — nombre de clients (hors dépôt)
    @return : list of lists — solution réparée respectant C1
    """
    K = len(routes)
    routes_rep = copy.deepcopy(routes)

    # ── Étape 1 : supprimer les doublons ──────────────────────────
    # On parcourt toutes les routes dans l'ordre et on garde
    # seulement la PREMIÈRE occurrence de chaque client.
    deja_vus = set()
    for k in range(K):
        route_propre = []
        for client in routes_rep[k]:
            if client not in deja_vus:
                route_propre.append(client)
                deja_vus.add(client)
            # Si client déjà vu → on le saute (doublon supprimé)
        routes_rep[k] = route_propre

    # ── Étape 2 : réinsérer les clients manquants ─────────────────
    clients_attendus = set(range(1, n + 1))
    manquants = clients_attendus - deja_vus

    for client in manquants:
        # On insère dans la route la moins chargée (équilibrage simple)
        k_cible = np.argmin([len(routes_rep[k]) for k in range(K)])
        routes_rep[k_cible].append(client)

    return routes_rep


# ═══════════════════════════════════════════════════════════════════
# PARTIE 1 : Initialisation de la population
# ═══════════════════════════════════════════════════════════════════

def creer_population(instance, taille_pop):
    """
    Crée une population initiale de solutions.

    @param instance    : dict généré par generate_instance()
    @param taille_pop  : int — nombre d'individus dans la population
    @return : list of routes — ['ind1', 'ind2', ...]
    """
    population = []
    for _ in range(taille_pop):
        individu = solution_initiale(instance)
        population.append(individu)
    return population


# ═══════════════════════════════════════════════════════════════════
# PARTIE 2 : Évaluation et sélection des meilleurs
# ═══════════════════════════════════════════════════════════════════

def evaluer_population(population, instance):
    """
    Calcule le coût de chaque individu.

    @param population : list of routes
    @param instance   : dict
    @return : list of float — coûts correspondants
    """
    couts = []
    for individu in population:
        cout = cout_penalise(individu, instance)
        couts.append(cout)
    return couts


def selectionner_meilleurs(population, couts, k):
    """
    Garde les k meilleures solutions de la population.

    @param population : list of routes
    @param couts      : list of float
    @param k          : int — nombre de parents à sélectionner
    @return : (meilleurs_parents, leurs_couts)
    """
    indices_tries = np.argsort(couts)       # tri croissant (moins cher = meilleur)
    indices_les_meilleurs = indices_tries[:k] # ← GARDE les k meilleurs

    meilleurs       = [population[i] for i in indices_les_meilleurs] # les parents retenus
    meilleurs_couts = [couts[i]      for i in indices_les_meilleurs]

    return meilleurs, meilleurs_couts 


# ═══════════════════════════════════════════════════════════════════
# PARTIE 3 : Croisement (Crossover) + vérification C1
# ═══════════════════════════════════════════════════════════════════

def croisement_simple(parent1, parent2, instance):
    """
    Croisement entre deux parents pour produire un enfant.

    Stratégie :
      1. Copier ~50% des routes du parent1 dans l'enfant
      2. Compléter avec les clients manquants dans l'ordre du parent2
      3. Vérifier C1 (couverture) et réparer si nécessaire

    ⚠️  Sans l'étape 3, un doublon dans un parent peut se propager
        silencieusement (set() ne détecte pas les doublons internes).

    @param parent1  : list of lists — routes du parent 1
    @param parent2  : list of lists — routes du parent 2
    @param instance : dict
    @return : list of lists — enfant valide (C1 garantie)
    """
    n = instance['n']           # nombre de clients (sans le dépôt)
    K = instance['n_vehicles']  # nombre de véhicules

    enfant         = [[] for _ in range(K)]
    clients_assigns = set()

    # ── Étape 1 : copier une partie aléatoire du parent1 ──────────
    for k in range(K):
        if np.random.random() < 0.5:
            enfant[k] = parent1[k].copy()
            clients_assigns.update(parent1[k])

    # ── Étape 2 : compléter avec les clients manquants du parent2 ──
    clients_manquants = set(range(1, n + 1)) - clients_assigns

    for client in clients_manquants:
        for route_parent2 in parent2:
            if client in route_parent2:
                # Insérer dans la route la moins chargée
                k_enfant = np.argmin([len(enfant[k]) for k in range(K)])
                enfant[k_enfant].append(client)
                clients_assigns.add(client)
                break

    # ── Étape 3 : vérification et réparation de C1 ────────────────
    # Nécessaire car un parent peut contenir un doublon interne
    # (bug en amont), ou un client peut avoir été raté à l'étape 2.
    valide, doublons, manquants = verifier_c1(enfant, n)

    if not valide:
        # Réparation automatique : suppression des doublons
        # + réinsertion des clients manquants
        enfant = reparer_c1(enfant, n)

        # Double vérification après réparation (mode debug)
        valide_apres, _, _ = verifier_c1(enfant, n)
        if not valide_apres:
            # Cas extrêmement rare : on retourne le parent1 intact
            # (solution de repli sûre)
            return copy.deepcopy(parent1)

    return enfant


# ═══════════════════════════════════════════════════════════════════
# PARTIE 4 : Mutation
# ═══════════════════════════════════════════════════════════════════

def mutation_legere(routes, instance):
    """
    Mutation légère : déplace un client aléatoire vers une autre route.
    Préserve C1 par construction (déplacement = suppression + insertion).

    @param routes   : list of lists — la solution à modifier
    @param instance : dict
    @return : list of lists — solution mutée
    """
    enfant = copy.deepcopy(routes)
    K = len(enfant)

    routes_avec_clients = [k for k in range(K) if len(enfant[k]) > 0]

    if len(routes_avec_clients) < 1:
        return enfant

    # Choisir une route source non vide
    k_src = np.random.choice(routes_avec_clients)

    if len(enfant[k_src]) == 0:
        return enfant

    # Prendre un client aléatoire de la route source
    idx_client = np.random.randint(0, len(enfant[k_src]))
    client     = enfant[k_src].pop(idx_client)  # suppression

    # Choisir une route destination
    k_dst = np.random.randint(0, K)

    # Insertion au début ou à la fin
    if len(enfant[k_dst]) == 0 or np.random.random() < 0.5:
        enfant[k_dst].insert(0, client)
    else:
        enfant[k_dst].append(client)

    # C1 est garantie : on a déplacé le client, pas dupliqué ni supprimé
    return enfant


# ═══════════════════════════════════════════════════════════════════
# PARTIE 5 : Algorithme génétique principal
# ═══════════════════════════════════════════════════════════════════

def algorithme_genetique(instance, taille_pop=20, n_generations=100,
                         taux_mutation=0.3, seed=None):
    """
    Exécute l'algorithme génétique complet.

    Respecte les contraintes VRPTW :
      C1 ── vérifiée + réparée à chaque croisement
      C2 ── par construction (dépôt implicite dans les listes)
      C3 ── soft constraint via cout_penalise()
      C4 ── soft constraint via cout_penalise()
      C5 ── propagation temporelle dans cout_penalise()
      C6 ── par construction (pas de sous-cycle possible)

    @param instance         : dict généré par generate_instance()
    @param taille_pop       : int   — taille de la population
    @param n_generations    : int   — nombre de générations
    @param taux_mutation    : float — probabilité de mutation (0 à 1)
    @param seed             : int   — graine pour la reproductibilité
    @return : (meilleure_solution, meilleur_cout, historique)
    """
    if seed is not None:
        np.random.seed(seed)

    t_debut = time.time()

    # ── Initialisation ────────────────────────────────────────────
    print(" Création de la population initiale...")
    population = creer_population(instance, taille_pop)

    historique        = {'couts_generations': []}
    meilleure_solution = None
    meilleur_cout      = float('inf')

    # ── Boucle des générations ────────────────────────────────────
    for gen in range(n_generations):

        # Évaluation de toute la population
        couts = evaluer_population(population, instance)

        # Mise à jour du meilleur global
        meilleur_cout_gen = min(couts)
        if meilleur_cout_gen < meilleur_cout:
            meilleur_cout      = meilleur_cout_gen
            idx_meilleur       = np.argmin(couts)
            meilleure_solution = copy.deepcopy(population[idx_meilleur])

        historique['couts_generations'].append(meilleur_cout_gen)

        if (gen + 1) % 10 == 0:
            print(f"  Génération {gen + 1}/{n_generations} "
                  f"— Meilleur coût : {meilleur_cout_gen:.2f}")

        # Sélection des parents (environ 33% de la population)
        nb_parents = max(2, taille_pop // 3)
        parents, _ = selectionner_meilleurs(population, couts, nb_parents)

        # ── Création de la nouvelle génération ───────────────────
        nouvelle_population = []

        # Élitisme : on conserve toujours les 2 meilleurs individus
        nouvelle_population.extend([copy.deepcopy(p) for p in parents[:2]])

        # Remplissage par croisement + mutation
        while len(nouvelle_population) < taille_pop:

            # Deux parents choisis au hasard parmi les meilleurs
            idx1, idx2 = np.random.choice(len(parents), size=2, replace=True)
            parent1    = parents[idx1]
            parent2    = parents[idx2]

            # Croisement (C1 vérifiée et réparée à l'intérieur)
            enfant = croisement_simple(parent1, parent2, instance)

            # Mutation avec probabilité taux_mutation
            if np.random.random() < taux_mutation:
                enfant = mutation_legere(enfant, instance)

            nouvelle_population.append(enfant)

        # La nouvelle génération remplace l'ancienne
        population = nouvelle_population[:taille_pop]

    t_fin = time.time()

    print(f"\n Algorithme terminé en {t_fin - t_debut:.2f}s")
    print(f" Meilleure solution trouvée : coût = {meilleur_cout:.2f}")

    return meilleure_solution, meilleur_cout, historique


# ═══════════════════════════════════════════════════════════════════
# PARTIE 6 : Exemple d'utilisation
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    PHASE4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Phase 4')
    sys.path.insert(0, PHASE4)
    from preprocess import generate_instance

    print(" Génération d'une instance VRPTW...")
    instance = generate_instance(n_clients=20, n_vehicles=3, seed=42)

    print("\n Lancement de l'algorithme génétique...\n")
    solution, cout, historique = algorithme_genetique(
        instance,
        taille_pop   = 20,
        n_generations= 50,
        taux_mutation= 0.3,
        seed         = 42
    )

    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"Coût final        : {cout:.2f}")
    print(f"Nombre de routes  : {len([r for r in solution if r])}")
    print(f"Routes            : {solution}")

    # ── Vérification finale de C1 ──────────────────────────────────
    n = instance['n']
    valide, doublons, manquants = verifier_c1(solution, n)
    print("\n" + "="*60)
    print("VÉRIFICATION CONTRAINTE C1")
    print("="*60)
    if valide:
        print("✅ C1 respectée : chaque client apparaît exactement une fois.")
    else:
        print(f"❌ C1 violée !")
        if doublons:
            print(f"   Clients en double  : {doublons}")
        if manquants:
            print(f"   Clients manquants  : {manquants}")
