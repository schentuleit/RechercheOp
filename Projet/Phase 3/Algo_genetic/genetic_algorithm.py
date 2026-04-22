"""
genetic_algorithm.py — Algorithme Génétique pour le VRPTW multi-véhicules
==========================================================================
Contraintes gérées : couverture C1, capacité C3, fenêtres temporelles C4
Opérateurs : sélection élitiste, croisement basé sur routes, mutation 2-opt / or-opt

Interface attendue par grands_tests_genetique.py :
    from genetic_algorithm import algorithme_genetique, verifier_c1

    solution, cout, historique = algorithme_genetique(instance, seed=seed_algo, **params)
    valide_c1, doublons, manquants = verifier_c1(solution, instance['n'])
    historique['couts_generations']  # liste des meilleurs coûts pénalisés par génération
"""

import copy
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. VÉRIFICATION ET RÉPARATION C1 (couverture)
# ─────────────────────────────────────────────────────────────────────────────

def verifier_c1(solution, n):
    """
    Vérifie que chaque client 1..n apparaît exactement une fois dans la solution.

    @param solution : list of lists — routes (ex: [[3,7], [5,2,9], [4,6]])
    @param n        : int — nombre de clients
    @return : (bool, list_doublons, list_manquants)
    """
    tous = [c for route in solution for c in route]

    compteur = {}
    for c in tous:
        compteur[c] = compteur.get(c, 0) + 1

    attendus = set(range(1, n + 1))
    presents = set(tous)

    doublons  = [c for c, nb in compteur.items() if nb > 1]
    manquants = sorted(attendus - presents)

    valide = (len(doublons) == 0 and len(manquants) == 0)
    return valide, doublons, manquants


def reparer_c1(solution, n, rng):
    """
    Répare une solution pour garantir C1 (chaque client exactement une fois).
    1) Supprime les occurrences dupliquées (garde la première)
    2) Insère les clients manquants dans la route la plus courte

    @param solution : list of lists
    @param n        : int
    @param rng      : np.random.Generator
    @return : solution réparée (list of lists, copie profonde)
    """
    sol = copy.deepcopy(solution)

    # Supprimer les doublons (garder la première occurrence)
    vus = set()
    for route in sol:
        i = 0
        while i < len(route):
            c = route[i]
            if c in vus:
                route.pop(i)
            else:
                vus.add(c)
                i += 1

    # Insérer les clients manquants
    presents = set(c for route in sol for c in route)
    manquants = [c for c in range(1, n + 1) if c not in presents]

    # Mélanger pour plus de diversité
    manquants_arr = np.array(manquants)
    rng.shuffle(manquants_arr)

    for client in manquants_arr:
        # Insérer dans la route la plus courte (position aléatoire)
        idx = int(np.argmin([len(r) for r in sol]))
        pos = int(rng.integers(0, len(sol[idx]) + 1))
        sol[idx].insert(pos, int(client))

    return sol


# ─────────────────────────────────────────────────────────────────────────────
# 2. ÉVALUATION DES SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _cout_reel(routes, instance):
    """
    Calcule la distance totale réelle (sans pénalités).
    Utilise la matrice dist pré-calculée de l'instance.

    @param routes   : list of lists
    @param instance : dict
    @return : float
    """
    dist  = instance['dist']
    total = 0.0

    for route in routes:
        if not route:
            continue
        total += dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]

    return total


def _cout_penalise(routes, instance):
    """
    Coût pénalisé = distance + pénalités capacité + fenêtres temporelles + horizon.
    Utilisé pendant l'optimisation pour permettre l'exploration de solutions invalides.

    Pénalités :
        Capacité     : 1000 × dépassement
        Fenêtre TW   : 500  × dépassement (minutes)
        Horizon      : 500  × dépassement (minutes)

    @param routes   : list of lists
    @param instance : dict
    @return : float
    """
    dist     = instance['dist']
    durees   = instance['durees']
    tw       = instance['time_windows']
    service  = instance['service_times']
    demands  = instance['demands']
    capacity = instance['capacity']
    horizon  = instance['horizon']

    total = 0.0

    for route in routes:
        if not route:
            continue

        # Distance réelle
        total += dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]

        # Pénalité capacité
        charge = sum(demands[c] for c in route)
        if charge > capacity + 1e-6:
            total += 1000.0 * (charge - capacity)

        # Pénalité fenêtres temporelles
        t   = 0.0
        pos = 0
        for client in route:
            t = t + durees[pos, client]
            t = max(t, tw[client, 0])           # attendre si arrivée trop tôt
            if t > tw[client, 1]:
                total += 500.0 * (t - tw[client, 1])
            t  += service[client]
            pos = client

        # Pénalité horizon (retour au dépôt)
        if t + durees[pos, 0] > horizon:
            total += 500.0 * (t + durees[pos, 0] - horizon)

    return total


def evaluer_population(population, instance):
    """
    Calcule le coût pénalisé de chaque individu de la population.

    @param population : list of solutions
    @param instance   : dict
    @return : list of float
    """
    return [_cout_penalise(sol, instance) for sol in population]


# ─────────────────────────────────────────────────────────────────────────────
# 3. INITIALISATION DE LA POPULATION
# ─────────────────────────────────────────────────────────────────────────────

def _solution_gloutonne_aleatoire(instance, rng):
    """
    Construit une solution initiale par affectation gloutonne randomisée.
    L'ordre des clients est mélangé aléatoirement → diversité de la population.

    @param instance : dict
    @param rng      : np.random.Generator
    @return : list of lists
    """
    n         = instance['n']
    K         = instance['n_vehicles']
    durees    = instance['durees']
    tw        = instance['time_windows']
    service   = instance['service_times']
    demands   = instance['demands']
    capacity  = instance['capacity']
    horizon   = instance['horizon']

    routes        = [[] for _ in range(K)]
    pos_veh       = np.zeros(K, dtype=int)        # position actuelle de chaque véhicule
    t_dispo       = np.zeros(K, dtype=float)      # heure de disponibilité
    capa_restante = np.full(K, capacity)          # capacité restante

    clients = list(range(1, n + 1))
    rng.shuffle(clients)

    for client in clients:
        assigne = False

        # Ordre de tentative : véhicules triés par disponibilité + bruit
        bruit = rng.uniform(0, 15, size=K)
        ordre = np.argsort(t_dispo + bruit)

        for k in ordre:
            k = int(k)
            # Contrainte capacité
            if demands[client] > capa_restante[k] + 1e-6:
                continue
            # Contrainte fenêtre temporelle
            t_arr = t_dispo[k] + durees[pos_veh[k], client]
            t_arr = max(t_arr, tw[client, 0])
            if t_arr > tw[client, 1] + 2.0:
                continue
            # Contrainte horizon
            if t_arr + service[client] + durees[client, 0] > horizon + 1e-6:
                continue

            routes[k].append(client)
            t_dispo[k]       = t_arr + service[client]
            capa_restante[k] -= demands[client]
            pos_veh[k]        = client
            assigne           = True
            break

        if not assigne:
            # Forcer sur la route la plus courte (solution potentiellement invalide)
            k = int(np.argmin([len(r) for r in routes]))
            routes[k].append(client)

    return routes


def creer_population(instance, taille_pop, rng):
    """
    Génère la population initiale par construction gloutonne randomisée.

    @param instance   : dict
    @param taille_pop : int
    @param rng        : np.random.Generator
    @return : list of solutions
    """
    population = []
    for _ in range(taille_pop):
        sol = _solution_gloutonne_aleatoire(instance, rng)
        sol = reparer_c1(sol, instance['n'], rng)
        population.append(sol)
    return population


# ─────────────────────────────────────────────────────────────────────────────
# 4. SÉLECTION
# ─────────────────────────────────────────────────────────────────────────────

def selectionner_meilleurs(population, couts, k):
    """
    Sélectionne les k individus avec le coût pénalisé le plus faible.

    @param population : list of solutions
    @param couts      : list of float
    @param k          : int
    @return : (solutions, costs) — les k meilleurs
    """
    indices       = np.argsort(couts)[:k]
    meilleurs     = [population[i] for i in indices]
    couts_triees  = [couts[i] for i in indices]
    return meilleurs, couts_triees


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROISEMENT
# ─────────────────────────────────────────────────────────────────────────────

def croisement_simple(parent1, parent2, n, rng):
    """
    Croisement basé sur les routes (Order Crossover adapté VRPTW) :
    1) Copie ~50% des routes du parent1 dans l'enfant
    2) Complète les clients manquants dans l'ordre d'apparition du parent2
    3) Répare C1

    @param parent1, parent2 : solutions (list of lists)
    @param n                : int — nombre de clients
    @param rng              : np.random.Generator
    @return : solution enfant (list of lists)
    """
    K = len(parent1)
    enfant = [[] for _ in range(K)]

    # Choisir aléatoirement ~50% des indices de routes à hériter du parent1
    nb_routes_p1 = max(1, K // 2)
    routes_p1 = set(rng.choice(K, size=nb_routes_p1, replace=False).tolist())

    clients_pris = set()
    for k in routes_p1:
        for c in parent1[k]:
            if c not in clients_pris:
                enfant[k].append(c)
                clients_pris.add(c)

    # Routes restantes (à remplir depuis parent2)
    routes_libres = [i for i in range(K) if i not in routes_p1]
    if not routes_libres:
        routes_libres = list(range(K))

    # Clients manquants dans l'ordre du parent2
    ordre_p2 = [c for route in parent2 for c in route if c not in clients_pris]

    # Distribuer équitablement dans les routes libres
    for idx_c, client in enumerate(ordre_p2):
        k = routes_libres[idx_c % len(routes_libres)]
        enfant[k].append(client)

    return reparer_c1(enfant, n, rng)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _mutation_2opt_intra(sol, rng):
    """Inversion d'un segment dans une route choisie aléatoirement."""
    candidates = [k for k, r in enumerate(sol) if len(r) >= 2]
    if not candidates:
        return sol
    k = int(rng.choice(candidates))
    route = sol[k]
    i, j = sorted(rng.choice(len(route), size=2, replace=False).tolist())
    sol[k][i:j+1] = sol[k][i:j+1][::-1]
    return sol


def _mutation_oropt_inter(sol, rng):
    """Déplace un client vers une autre position (intra ou inter-route)."""
    non_vides = [k for k, r in enumerate(sol) if len(r) >= 1]
    if not non_vides:
        return sol
    k_src  = int(rng.choice(non_vides))
    idx    = int(rng.integers(0, len(sol[k_src])))
    client = sol[k_src].pop(idx)

    K       = len(sol)
    k_dst   = int(rng.integers(0, K))
    pos_ins = int(rng.integers(0, len(sol[k_dst]) + 1))
    sol[k_dst].insert(pos_ins, client)
    return sol


def _mutation_swap(sol, rng):
    """Échange deux clients de routes différentes."""
    non_vides = [k for k, r in enumerate(sol) if len(r) >= 1]
    if len(non_vides) < 2:
        return sol
    k1, k2 = rng.choice(non_vides, size=2, replace=False).tolist()
    i1 = int(rng.integers(0, len(sol[k1])))
    i2 = int(rng.integers(0, len(sol[k2])))
    sol[k1][i1], sol[k2][i2] = sol[k2][i2], sol[k1][i1]
    return sol


def mutation_legere(solution, taux_mutation, rng):
    """
    Applique une mutation aléatoire avec probabilité taux_mutation.
    Trois opérateurs équiprobables : 2-opt intra-route, or-opt inter-route, swap.

    @param solution      : list of lists
    @param taux_mutation : float — probabilité de muter
    @param rng           : np.random.Generator
    @return : solution mutée (copie profonde)
    """
    sol = copy.deepcopy(solution)

    if rng.random() < taux_mutation:
        op = rng.integers(0, 3)
        if op == 0:
            sol = _mutation_2opt_intra(sol, rng)
        elif op == 1:
            sol = _mutation_oropt_inter(sol, rng)
        else:
            sol = _mutation_swap(sol, rng)

    return sol


# ─────────────────────────────────────────────────────────────────────────────
# 7. RECHERCHE LOCALE — 2-OPT PAR ROUTE
# ─────────────────────────────────────────────────────────────────────────────

def _2opt_route(route, dist):
    """
    Applique 2-opt exhaustif sur une seule route (intra-route).
    Répète jusqu'à convergence locale.

    @param route : list of int — indices clients (sans dépôt)
    @param dist  : np.ndarray (n+1, n+1) — matrice de distances
    @return : route améliorée (list of int)
    """
    n = len(route)
    if n < 2:
        return route
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Coût avant : ...-route[i] → route[i+1]-...-route[j] → route[j+1]-...
                a, b = route[i], route[i + 1]
                c, d = route[j], route[(j + 1) % n] if j + 1 < n else 0
                # Gain si on inverse le segment [i+1 .. j]
                avant  = dist[a if i > 0 else 0, b] + dist[c, d if j + 1 < n else 0]
                apres  = dist[a if i > 0 else 0, c] + dist[b, d if j + 1 < n else 0]
                # Note : utilise indices avec dépôt=0 pour les bords
                if apres < avant - 1e-6:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    improved = True
    return route


def _2opt_route_simple(route, dist):
    """
    2-opt simplifié utilisant correctement les arcs dépôt→premier et dernier→dépôt.
    """
    n = len(route)
    if n < 2:
        return route

    r = [0] + route + [0]   # on inclut le dépôt aux deux bouts
    m = len(r)
    improved = True
    while improved:
        improved = False
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                d_avant = dist[r[i-1], r[i]] + dist[r[j], r[j+1]]
                d_apres = dist[r[i-1], r[j]] + dist[r[i], r[j+1]]
                if d_apres < d_avant - 1e-6:
                    r[i:j+1] = r[i:j+1][::-1]
                    improved = True
    return r[1:-1]


def optimiser_routes_2opt(solution, instance):
    """
    Applique 2-opt intra-route sur toutes les routes de la solution.
    Améliore l'ordre des clients sans changer leur affectation aux véhicules.

    @param solution : list of lists
    @param instance : dict
    @return : solution avec routes réordonnées (copie)
    """
    dist = instance['dist']
    sol  = copy.deepcopy(solution)
    for k, route in enumerate(sol):
        if len(route) >= 2:
            sol[k] = _2opt_route_simple(route, dist)
    return sol


# ─────────────────────────────────────────────────────────────────────────────
# 8. ALGORITHME GÉNÉTIQUE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def algorithme_genetique(instance, taille_pop=30, n_generations=100,
                         taux_mutation=0.3, seed=None):
    """
    Algorithme génétique pour le VRPTW.

    Schéma :
        1. Initialisation : population gloutonne randomisée
        2. Pour chaque génération :
            a. Évaluation (coût pénalisé)
            b. Sélection des meilleurs (35% de la pop)
            c. Élitisme : les 2 meilleurs passent sans modification
            d. Croisement + mutation pour compléter la population
        3. Retourne la meilleure solution trouvée (coût réel sans pénalité)

    @param instance      : dict généré par generate_instance()
    @param taille_pop    : int   — taille de la population (défaut 30)
    @param n_generations : int   — nombre de générations (défaut 100)
    @param taux_mutation : float — probabilité de mutation par individu (défaut 0.3)
    @param seed          : int   — graine aléatoire pour reproductibilité
    @return : (meilleure_solution, meilleur_cout_reel, historique)
              historique['couts_generations'] = liste des meilleurs coûts pénalisés
                                               (un par génération, inclus génération 0)
    """
    rng = np.random.default_rng(seed)
    n   = instance['n']

    # ── Initialisation ───────────────────────────────────────────────────────
    population = creer_population(instance, taille_pop, rng)
    # Améliorer chaque individu initial par 2-opt intra-route
    population = [optimiser_routes_2opt(sol, instance) for sol in population]
    couts      = evaluer_population(population, instance)

    idx_best      = int(np.argmin(couts))
    meilleure_sol = copy.deepcopy(population[idx_best])
    meilleur_cout_penalise = couts[idx_best]

    historique = {'couts_generations': [meilleur_cout_penalise]}

    # ── Hyperparamètres de sélection ─────────────────────────────────────────
    n_elites  = 2
    n_parents = max(n_elites + 2, int(taille_pop * 0.35))

    # ── Boucle générationnelle ────────────────────────────────────────────────
    for _ in range(n_generations):

        # 1. Sélection des parents (meilleurs individus)
        parents, parents_couts = selectionner_meilleurs(population, couts, n_parents)

        # 2. Élitisme — les n_elites meilleurs passent intacts
        nouvelle_pop   = [copy.deepcopy(parents[i]) for i in range(n_elites)]
        nouveaux_couts = list(parents_couts[:n_elites])

        # 3. Génération de la descendance par croisement + mutation
        while len(nouvelle_pop) < taille_pop:
            # Choisir deux parents distincts (avec remise si pop petite)
            if n_parents >= 2:
                i1, i2 = rng.choice(n_parents, size=2, replace=False).tolist()
            else:
                i1, i2 = 0, 0

            enfant = croisement_simple(parents[i1], parents[i2], n, rng)
            enfant = mutation_legere(enfant, taux_mutation, rng)
            enfant = reparer_c1(enfant, n, rng)
            enfant = optimiser_routes_2opt(enfant, instance)

            cout_e = _cout_penalise(enfant, instance)
            nouvelle_pop.append(enfant)
            nouveaux_couts.append(cout_e)

        population = nouvelle_pop
        couts      = nouveaux_couts

        # 4. Mise à jour du meilleur global
        idx_best = int(np.argmin(couts))
        if couts[idx_best] < meilleur_cout_penalise:
            meilleur_cout_penalise = couts[idx_best]
            meilleure_sol = copy.deepcopy(population[idx_best])

        historique['couts_generations'].append(meilleur_cout_penalise)

    # ── Post-traitement : 2-opt final sur la meilleure solution ──────────────
    meilleure_sol = optimiser_routes_2opt(meilleure_sol, instance)

    # ── Coût réel final (sans pénalités) ─────────────────────────────────────
    cout_final_reel = _cout_reel(meilleure_sol, instance)

    return meilleure_sol, cout_final_reel, historique
