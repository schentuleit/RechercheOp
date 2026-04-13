"""
simulated_annealing.py — Recuit Simulé pour le VRPTW multi-véhicules
=====================================================================
Contraintes gérées : fenêtres temporelles + capacité des véhicules
Mouvements : 2-opt intra-route, Or-opt (déplacement inter-routes)
"""

import numpy as np
import time
import sys
import os

# Pour pouvoir importer generate_instance et preprocess depuis Phase 4
PHASE4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Phase 4')
sys.path.insert(0, PHASE4)

from preprocess import generate_instance, preprocess # on importe la génération d'instance et le prétraitement de la phase 4 pour réutiliser les mêmes données et fonctions de coût

def solution_initiale(instance):
    """
    Construit une solution initiale par affectation gloutonne.
    Assigne chaque client au véhicule le moins chargé qui peut le servir.

    @param instance : dict généré par generate_instance()
    @return : list of lists — routes[k] = liste des clients du véhicule k
    """
    n          = instance['n']
    K          = instance['n_vehicles']
    coords     = instance['coords']
    durees     = instance['durees']
    tw         = instance['time_windows']
    service    = instance['service_times']
    demands    = instance['demands']
    capacity   = instance['capacity']
    horizon    = instance['horizon']

    routes         = [[] for _ in range(K)]       #une par véhicule
    pos            = np.zeros(K, dtype=int)       # position actuelle (dépôt=0)
    t_dispo        = np.zeros(K, dtype=float)     # heure de disponibilité
    capa_restante  = np.full(K, capacity)         # capacité restante

    #on initialise l'etat de chaque véhicule : tous au dépôt, disponibles à t=0, capacité pleine

    clients_non_assignes = list(range(1, n + 1))

    for client in clients_non_assignes:
        assigne = False
        # Essayer chaque véhicule dans l'ordre du moins occupé
        for k in np.argsort(t_dispo):  #np.argsort(t_dispo) retourne les indices triés par temps de disponibilité croissant.
            k = int(k)
            # Vérifier capacité
            if demands[client] > capa_restante[k] + 1e-6:
                continue
            # Vérifier fenêtre temporelle
            t_arrivee = t_dispo[k] + durees[pos[k], client]
            t_arrivee = max(t_arrivee, tw[client, 0]) # attendre si arrivée trop tôt
            if t_arrivee > tw[client, 1] + 2.0:  # arrivée trop tard → ne pas assigner à ce véhicule, on essaie le suivant
                continue 
            # Vérifier retour dépôt possible
            t_retour = t_arrivee + service[client] + durees[client, 0]
            if t_retour > horizon + 1e-6:
                continue
            # Assigner
            routes[k].append(client)
            t_dispo[k]       = t_arrivee + service[client]
            capa_restante[k] -= demands[client]
            pos[k]            = client
            assigne           = True
            break

        if not assigne:
            # Forcer sur le véhicule le moins chargé (solution potentiellement invalide)
            k = int(np.argmin([len(r) for r in routes]))
            routes[k].append(client) #si aucun véhicule ne peut servir le client, on l'assigne quand même au véhicule le moins chargé (solution invalide mais on laisse le recuit essayer de l'améliorer)

    return routes

def cout_solution(routes, instance):
    """
    Calcule la distance totale d'une solution.

    @param routes   : list of lists — routes[k] = liste des clients du véhicule k
    @param instance : dict généré par generate_instance()
    @return : float — distance totale
    """
    coords = instance['coords']
    total  = 0.0

    for route in routes:
        if not route:
            continue
        # Dépôt → premier client
        total += np.linalg.norm(coords[0] - coords[route[0]])
        # Client → client suivant
        for i in range(len(route) - 1):
            total += np.linalg.norm(coords[route[i]] - coords[route[i + 1]])
        # Dernier client → dépôt
        total += np.linalg.norm(coords[route[-1]] - coords[0])
        #np.linalg.norm calcule la distance euclidienne entre deux points. On somme les distances pour chaque segment de la route, y compris le retour au dépôt.

    return total

def cout_penalise(routes, instance):
    """
    Coût total avec pénalités pour les violations de contraintes.
    Utilisé par le recuit pour permettre l'exploration de solutions invalides.

    Pénalité capacité    : 500 × dépassement
    Pénalité time window : 200 × dépassement en minutes
    Pénalité horizon     : 500 × dépassement

    @param routes   : list of lists
    @param instance : dict
    @return : float — coût pénalisé
    """
    coords   = instance['coords']
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

        # Distance
        total += np.linalg.norm(coords[0] - coords[route[0]])
        for i in range(len(route) - 1):
            total += np.linalg.norm(coords[route[i]] - coords[route[i + 1]])
        total += np.linalg.norm(coords[route[-1]] - coords[0])

        # Pénalité capacité
        charge = sum(demands[c] for c in route)
        if charge > capacity + 1e-6:
            total += 500.0 * (charge - capacity)

        # Pénalité fenêtres temporelles et horizon
        t   = 0.0
        pos = 0
        for client in route:
            t = t + durees[pos, client]
            t = max(t, tw[client, 0])
            if t > tw[client, 1]:
                total += 200.0 * (t - tw[client, 1])
            t  += service[client]
            pos = client

        if t + durees[pos, 0] > horizon:
            total += 500.0 * (t + durees[pos, 0] - horizon)

    return total


def est_valide(route, instance):
    """
    Vérifie qu'une route respecte les fenêtres temporelles et la capacité.

    @param route    : list — liste des clients du véhicule
    @param instance : dict généré par generate_instance()
    @return : bool
    """
    if not route:
        return True

    durees   = instance['durees']
    tw       = instance['time_windows']
    service  = instance['service_times']
    demands  = instance['demands']
    capacity = instance['capacity']
    horizon  = instance['horizon']

    # Vérification capacité
    if sum(demands[c] for c in route) > capacity + 1e-6:
        return False

    # Vérification fenêtres temporelles
    t   = 0.0
    pos = 0  # dépôt
    for client in route:
        t = t + durees[pos, client]
        t = max(t, tw[client, 0])       # attendre si arrivée trop tôt
        if t > tw[client, 1] + 2.0:    # arrivée trop tard → invalide
            return False
        t  += service[client]
        pos = client

    # Vérification retour au dépôt dans l'horizon
    if t + durees[pos, 0] > horizon + 1e-6:
        return False

    return True



def voisin_2opt(routes, instance):
    """
    Génère un voisin par 2-opt intra-route.
    Choisit une route au hasard et inverse un segment de cette route.

    @param routes   : list of lists
    @param instance : dict
    @return : list of lists — nouvelle solution (copie)
    """
    import copy
    nouvelles_routes = copy.deepcopy(routes)

    # Ne considérer que les routes avec au moins 2 clients
    candidates = [k for k, r in enumerate(nouvelles_routes) if len(r) >= 2]
    if not candidates:
        return nouvelles_routes

    # Choisir une route au hasard parmi les candidates
    k     = np.random.choice(candidates)
    route = nouvelles_routes[k]
    n     = len(route)

    # Choisir deux positions i < j au hasard
    i, j = sorted(np.random.choice(n, size=2, replace=False))

    # Inverser le segment entre i et j
    route[i:j+1] = route[i:j+1][::-1]

    return nouvelles_routes

    


def voisin_oropt(routes, instance):
    """
    Génère un voisin par Or-opt : déplace un client vers une autre position.
    Le client peut aller dans sa propre route ou dans une autre.

    @param routes   : list of lists
    @param instance : dict
    @return : list of lists — nouvelle solution (copie)
    """
    import copy
    nouvelles_routes = copy.deepcopy(routes)

    # Routes non vides
    candidates = [k for k, r in enumerate(nouvelles_routes) if len(r) >= 1]
    if not candidates:
        return nouvelles_routes

    # Choisir la route source et le client à déplacer
    k_src  = np.random.choice(candidates)
    idx    = np.random.randint(0, len(nouvelles_routes[k_src]))
    client = nouvelles_routes[k_src].pop(idx)

    # Choisir la route destination (peut être la même)
    K      = len(nouvelles_routes)
    k_dst  = np.random.randint(0, K)

    # Choisir la position d'insertion dans la route destination
    pos_max = len(nouvelles_routes[k_dst])
    pos_ins = np.random.randint(0, pos_max + 1)
    nouvelles_routes[k_dst].insert(pos_ins, client)

    return nouvelles_routes



def generer_voisin(routes, instance):
    """
    Choisit aléatoirement un mouvement de voisinage et l'applique.
    50% de chance pour 2-opt, 50% pour or-opt.

    @param routes   : list of lists
    @param instance : dict
    @return : list of lists — nouvelle solution
    """
    if np.random.random() < 0.5:
        return voisin_2opt(routes, instance)
    else:
        return voisin_oropt(routes, instance)


def recuit_simule(instance, T0=1000.0, alpha=0.995, n_iter=10000, seed=None):
    """
    Algorithme de recuit simulé pour le VRPTW.

    @param instance : dict généré par generate_instance()
    @param T0       : float — température initiale
    @param alpha    : float — taux de refroidissement (ex: 0.995)
    @param n_iter   : int   — nombre d'itérations
    @param seed     : int   — graine aléatoire (reproductibilité)
    @return : (meilleure_solution, meilleur_cout, historique)
              historique = dict avec listes 'couts' et 'temperatures'
    """
    if seed is not None:
        np.random.seed(seed)

    t_debut = time.time()

    # Initialisation 
    solution_courante = solution_initiale(instance)
    cout_courant = cout_penalise(solution_courante, instance)

    meilleure_solution = solution_courante
    meilleur_cout      = cout_courant

    T = T0

    historique = {'couts': [cout_courant], 'temperatures': [T]}

    #  Boucle principale 
    for iteration in range(n_iter):

        # Générer un voisin
        voisin      = generer_voisin(solution_courante, instance)
        cout_voisin = cout_penalise(voisin, instance)

        # Décision d'acceptation
        delta = cout_voisin - cout_courant

        if delta < 0:
            # Le voisin est meilleur alors on accepte toujours
            solution_courante = voisin
            cout_courant      = cout_voisin
        else:
            # Le voisin est moins bon alors on accepte avec probabilité e^(-delta/T)
            proba = np.exp(-delta / T) if T > 1e-10 else 0.0
            if np.random.random() < proba:
                solution_courante = voisin
                cout_courant      = cout_voisin

        # Mettre à jour le meilleur
        if cout_courant < meilleur_cout:
            meilleure_solution = solution_courante
            meilleur_cout      = cout_courant

        # Refroidissement
        T = T * alpha

        # Enregistrer pour l'historique (tous les 100 itérations)
        if iteration % 100 == 0:
            historique['couts'].append(cout_courant)
            historique['temperatures'].append(T)

    historique['temps'] = time.time() - t_debut

    return meilleure_solution, meilleur_cout, historique
