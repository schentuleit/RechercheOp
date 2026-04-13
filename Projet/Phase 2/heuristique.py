"""
heuristique.py — Heuristique NNH + 2-opt pour le VRPTW
========================================================
Phase 2 — Méthode 1 : Nearest Neighbor Heuristic + amélioration locale 2-opt

Contraintes gérées : fenêtres temporelles [a_i, b_i], capacité des véhicules,
                     horizon journalier H, couverture de tous les clients.

Pipeline :
    1. NNH  : construction gloutonne d'une solution valide en O(n²)
    2. 2-opt : amélioration locale intra-route, élimination des croisements
"""

import time
import sys
import os

# Import du générateur d'instances partagé par toute l'équipe (Phase 4)
PHASE4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Phase 4')
sys.path.insert(0, PHASE4)

from preprocess import generate_instance


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def cout_solution(routes, instance):
    """
    Calcule le coût total d'une solution = somme des distances euclidiennes.

    Utilise la matrice dist pré-calculée par generate_instance().
    L'unité est la même que les coordonnées (carte 100×100).
    Pour obtenir le temps en minutes : coût × vitesse (0.6 min/unité).

    @param routes   : list of lists — routes[k] = liste ordonnée des clients du véhicule k
    @param instance : dict généré par generate_instance()
    @return         : float — distance totale (unités spatiales)
    """
    dist  = instance['dist']   # matrice (n+1, n+1) pré-calculée
    total = 0.0

    for route in routes:
        if not route:
            continue

        total += dist[0, route[0]]                                    # dépôt → premier client

        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]                    # client → client suivant

        total += dist[route[-1], 0]                                   # dernier client → dépôt

    return total


def est_valide(route, instance):
    """
    Vérifie qu'une route respecte toutes les contraintes VRPTW.

    Simule le trajet complet du véhicule et contrôle :
    - C3 : somme des demandes ≤ capacité Q
    - C4 : arrivée chez chaque client dans sa fenêtre [a_i, b_i]
    - C5 : retour au dépôt avant l'horizon H

    @param route    : list — liste ordonnée des clients du véhicule
    @param instance : dict
    @return         : bool
    """
    if not route:
        return True

    durees   = instance['durees']
    tw       = instance['time_windows']
    service  = instance['service_times']
    demands  = instance['demands']
    capacity = instance['capacity']
    horizon  = instance['horizon']

    # C3 — capacité (notation du notebook de modélisation)
    if sum(demands[c] for c in route) > capacity + 1e-6:
        return False

    # C4 + C5 — fenêtres temporelles et cohérence temporelle (notation du notebook)
    t   = 0.0
    pos = 0  # dépôt

    for client in route:
        t   = t + durees[pos, client]
        t   = max(t, tw[client, 0])        # attente si arrivée trop tôt (C4 : a_i ≤ t)

        if t > tw[client, 1] + 1e-6:      # arrivée trop tard → C4 violée (t > b_i)
            return False

        t  += service[client]
        pos = client

    if t + durees[pos, 0] > horizon + 1e-6:  # retour dépôt → C5 cohérence temporelle
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Nearest Neighbor Heuristic (NNH)
# ─────────────────────────────────────────────────────────────────────────────

def construire_solution_nnh(instance):
    """
    Construit une solution initiale par l'heuristique du plus proche voisin.

    Pour chaque véhicule, on part du dépôt et on ajoute itérativement
    le client non livré le plus proche qui reste faisable (C3, C4, C5).
    Quand plus aucun client n'est accessible, le véhicule rentre au dépôt
    et le suivant prend la relève.

    n_vehicles est utilisé comme nombre minimum de véhicules ; si l'instance
    en requiert davantage pour couvrir tous les clients, des véhicules
    supplémentaires sont créés automatiquement (garantie de C1).

    Complexité : O(n²) par véhicule, O(n³) au total dans le pire cas (K véhicules en O(n)).

    @param instance : dict généré par generate_instance()
    @return         : list of lists — routes[k] = liste des clients du véhicule k
    """
    n        = instance['n']
    dist     = instance['dist']
    durees   = instance['durees']
    tw       = instance['time_windows']
    service  = instance['service_times']
    demands  = instance['demands']
    capacity = instance['capacity']
    horizon  = instance['horizon']

    routes             = []
    clients_non_livres = set(range(1, n + 1))
    clients_infaisables = []  # clients qu'aucun véhicule ne peut servir

    # Pré-détection des clients fondamentalement infaisables :
    # C3 : demande individuelle dépasse la capacité totale du véhicule
    # C4 : fenêtre temporelle fermée avant même d'arriver depuis le dépôt à t=0
    # C5 : retour au dépôt impossible même si on part immédiatement
    for client in list(clients_non_livres):
        t_arrivee_min = max(durees[0, client], tw[client, 0])
        c3_ok  = demands[client] <= capacity + 1e-6
        tw_ok  = t_arrivee_min <= tw[client, 1] + 1e-6
        c5_ok  = t_arrivee_min + service[client] + durees[client, 0] <= horizon + 1e-6
        if not c3_ok or not tw_ok or not c5_ok:
            clients_infaisables.append(client)
            clients_non_livres.discard(client)

    # On continue à ouvrir des véhicules tant qu'il reste des clients faisables
    while clients_non_livres:
        route             = []
        pos               = 0
        t                 = 0.0
        capacite_restante = capacity

        while True:
            meilleur_client = None
            meilleure_dist  = float('inf')

            for client in clients_non_livres:

                # C3 — capacité
                if demands[client] > capacite_restante + 1e-6:
                    continue

                # Heure d'arrivée (avec attente éventuelle si trop tôt)
                t_arrivee = max(t + durees[pos, client], tw[client, 0])

                # C4 — fenêtre temporelle : arrivée avant fermeture b_i
                if t_arrivee > tw[client, 1] + 1e-6:
                    continue

                # C5 — cohérence temporelle : retour au dépôt avant H
                if t_arrivee + service[client] + durees[client, 0] > horizon + 1e-6:
                    continue

                # Critère de sélection : distance minimale (nearest neighbor)
                d = dist[pos, client]
                if d < meilleure_dist:
                    meilleure_dist  = d
                    meilleur_client = client

            if meilleur_client is None:
                break  # aucun client accessible depuis ici → ce véhicule a terminé

            # Ajout du client à la route
            route.append(meilleur_client)
            clients_non_livres.discard(meilleur_client)

            # Mise à jour de l'état du véhicule
            t_arrivee         = max(t + durees[pos, meilleur_client], tw[meilleur_client, 0])
            t                 = t_arrivee + service[meilleur_client]
            capacite_restante -= demands[meilleur_client]
            pos                = meilleur_client

        if route:
            routes.append(route)
        else:
            # Aucun client n'est accessible depuis le dépôt avec un véhicule neuf.
            # Après la pré-détection, ce cas ne devrait pas se produire.
            # Par sécurité, on considère les clients restants comme infaisables
            # (C1 reste garanti : ils apparaîtront dans une route invalide).
            clients_infaisables.extend(clients_non_livres)
            break

    # Les clients infaisables sont ajoutés en dernier pour garantir C1.
    # Leur route sera marquée invalide par est_valide() → valide=False dans stats.
    if clients_infaisables:
        routes.append(clients_infaisables)

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Amélioration locale 2-opt
# ─────────────────────────────────────────────────────────────────────────────

def ameliorer_2opt(routes, instance):
    """
    Améliore une solution par l'heuristique 2-opt intra-route.

    Pour chaque route, on teste toutes les paires d'arcs (i, j).
    Si inverser le segment [i, j] raccourcit la route ET reste valide
    (contraintes VRPTW), on effectue l'inversion.
    On répète jusqu'à ce qu'aucune amélioration ne soit possible
    (optimum local 2-opt).

    Gain d'une inversion :
        Δ = c(r[i-1], r[i]) + c(r[j], r[j+1])
          - c(r[i-1], r[j]) - c(r[i], r[j+1])
    Si Δ > 0 → l'inversion raccourcit la route.

    Complexité : O(m²) par itération (m = taille de la route), O(m⁴) pire cas total.
    En pratique O(m²) à O(m³) grâce à la convergence rapide sur instances TW-contraintes.

    @param routes   : list of lists — solution produite par NNH
    @param instance : dict
    @return         : list of lists — solution améliorée
    """
    dist = instance['dist']

    amelioration_globale = True

    while amelioration_globale:
        amelioration_globale = False

        for idx_route, route in enumerate(routes):
            n = len(route)
            if n < 3:
                continue

            amelioration_route = True

            while amelioration_route:
                amelioration_route = False

                for i in range(n - 1):
                    for j in range(i + 2, n):

                        node_avant_i = 0 if i == 0     else route[i - 1]
                        node_apres_j = 0 if j == n - 1 else route[j + 1]

                        # Calcul du gain Δ
                        cout_avant = dist[node_avant_i, route[i]] + dist[route[j], node_apres_j]
                        cout_apres = dist[node_avant_i, route[j]] + dist[route[i], node_apres_j]
                        delta      = cout_avant - cout_apres

                        if delta > 1e-6:
                            route_candidate = route[:i] + route[i:j+1][::-1] + route[j+1:]

                            # Valider les contraintes VRPTW avant d'accepter
                            # (une inversion peut raccourcir la distance mais violer les TW)
                            if est_valide(route_candidate, instance):
                                routes[idx_route]    = route_candidate
                                route                = route_candidate
                                amelioration_route   = True
                                amelioration_globale = True
                                break

                    if amelioration_route:
                        break

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def resoudre_heuristique(instance):
    """
    Résout une instance VRPTW par NNH + amélioration 2-opt.

    Pipeline :
        1. NNH  : construction d'une solution valide en O(n²)
        2. 2-opt : amélioration locale de chaque route

    @param instance : dict généré par generate_instance()
    @return         : (routes, cout_final, stats)
                      routes     = list of lists
                      cout_final = float
                      stats      = dict avec métriques de performance
    """
    # Étape 1 — NNH
    t0     = time.time()
    routes = construire_solution_nnh(instance)
    t_nnh  = time.time() - t0

    cout_initial = cout_solution(routes, instance)

    # Étape 2 — 2-opt
    t1     = time.time()
    routes = ameliorer_2opt(routes, instance)
    t_2opt = time.time() - t1

    cout_final = cout_solution(routes, instance)

    # Statistiques
    gain_pct             = (cout_initial - cout_final) / cout_initial * 100 if cout_initial > 0 else 0.0
    n_vehicules          = sum(1 for r in routes if r)
    routes_invalides     = [r for r in routes if r and not est_valide(r, instance)]
    n_infaisables        = sum(len(r) for r in routes_invalides)
    valide               = len(routes_invalides) == 0

    stats = {
        'temps_nnh'              : round(t_nnh, 4),
        'temps_2opt'             : round(t_2opt, 4),
        'temps_total'            : round(t_nnh + t_2opt, 4),
        'cout_initial'           : round(cout_initial, 2),
        'cout_final'             : round(cout_final, 2),
        'gain_2opt_pct'          : round(gain_pct, 2),
        'n_vehicules_utilises'   : n_vehicules,
        'n_clients_infaisables'  : n_infaisables,
        'valide'                 : valide,
    }

    return routes, cout_final, stats


# ─────────────────────────────────────────────────────────────────────────────
# Lancement direct : python heuristique.py [n] [n_vehicles] [seed]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heuristique NNH + 2-opt pour le VRPTW")
    parser.add_argument("--n",        type=int, default=20,   help="Nombre de clients (défaut: 20)")
    parser.add_argument("--vehicles", type=int, default=None, help="Nombre de véhicules (défaut: auto = ceil(n/10), min 3)")
    parser.add_argument("--seed",     type=int, default=42,   help="Graine aléatoire (défaut: 42)")
    args = parser.parse_args()

    # Calcul automatique du nombre de véhicules si non précisé (formule partagée du générateur)
    # On laisse generate_instance() calculer n_vehicles lui-même quand non précisé,
    # pour garantir la cohérence capacité/demande (évite les instances trivalement infaisables).
    if args.vehicles is not None:
        n_vehicles = args.vehicles
        instance   = generate_instance(n=args.n, n_vehicles=n_vehicles, seed=args.seed)
    else:
        instance   = generate_instance(n=args.n, seed=args.seed)
        n_vehicles = instance['n_vehicles']
    routes, cout, stats = resoudre_heuristique(instance)

    SEP  = "=" * 62
    SEP2 = "-" * 62

    # ── En-tête ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  HEURISTIQUE NNH + 2-opt — VRPTW")
    print(SEP)
    print(f"  {'Clients':<20} {args.n}")
    print(f"  {'Véhicules demandés':<20} {n_vehicles}{' (auto)' if args.vehicles is None else ''}")
    print(f"  {'Seed':<20} {args.seed}")
    print(f"  {'Capacité / véhicule':<20} {instance['capacity']:.1f} unités")
    print(f"  {'Horizon journalier':<20} {instance['horizon']:.0f} min (8h)")
    print(SEP2)

    # ── Tableau des routes ────────────────────────────────────────
    print(f"\n  {'Véhicule':<12} {'Clients visités':<35} {'Nb clients'}")
    print(f"  {'-'*10}  {'-'*33}  {'-'*10}")
    for k, route in enumerate(routes):
        if route:
            chemin    = " > ".join(map(str, route))
            nb        = len(route)
            # Tronquer l'affichage si la route est très longue
            if len(chemin) > 33:
                chemin = chemin[:30] + "..."
            print(f"  Véhicule {k+1:<3}  {chemin:<35} {nb}")

    print(f"\n  Véhicules demandés : {n_vehicles}{' (auto)' if args.vehicles is None else ''}")
    print(f"  Véhicules utilisés : {stats['n_vehicules_utilises']}")
    if stats['n_vehicules_utilises'] > n_vehicles:
        print(f"\n  /!\\ ATTENTION : avec {n_vehicles} véhicule(s), il est impossible")
        print(f"      de livrer tous les {args.n} clients dans leurs créneaux horaires.")
        print(f"      L'algorithme a ouvert {stats['n_vehicules_utilises']} véhicules au minimum")
        print(f"      pour garantir que chaque client soit livré (contrainte C1).")
        print(f"      Relancez avec --vehicles {stats['n_vehicules_utilises']} pour correspondre")
        print(f"      à la réalité de cette instance.")

    # ── Tableau des performances ──────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  PERFORMANCES")
    print(SEP2)
    print(f"  {'Coût après NNH':<30} {stats['cout_initial']:>8.2f} unités")
    print(f"  {'Coût après 2-opt':<30} {stats['cout_final']:>8.2f} unités")
    print(f"  {'Gain apporté par le 2-opt':<30} {stats['gain_2opt_pct']:>7.2f} %")
    print(SEP2)
    print(f"  {'Temps NNH':<30} {stats['temps_nnh']:>8.4f} s")
    print(f"  {'Temps 2-opt':<30} {stats['temps_2opt']:>8.4f} s")
    print(f"  {'Temps total':<30} {stats['temps_total']:>8.4f} s")
    print(SEP2)
    valide_str = "OUI" if stats['valide'] else "NON — contraintes violées"
    print(f"  {'Solution valide':<30} {valide_str}")
    print(f"{SEP}\n")
