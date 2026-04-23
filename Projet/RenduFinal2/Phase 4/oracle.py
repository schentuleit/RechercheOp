"""
oracle.py — Générateur de solutions optimales (labels GNN)
===========================================================
Utilise OR-Tools pour résoudre le VRPTW et extraire :
    - Les arêtes utilisées dans la solution optimale  → labels positifs
    - Les arêtes non utilisées                         → labels négatifs

Ces labels servent à entraîner le GNN en classification supervisée.

Usage :
    from oracle import resoudre_instance, generer_dataset
    dataset = generer_dataset(n=10, n_instances=1000)
"""

import numpy as np
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from preprocess import generate_instance, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# 1. RÉSOLUTION OR-TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def resoudre_instance(instance: dict, time_limit_s: float = 5.0):
    """
    Résout une instance VRPTW avec OR-Tools (GLS + PATH_CHEAPEST_ARC).

    Paramètres
    ----------
    instance     : dict de generate_instance()
    time_limit_s : secondes allouées. 3s suffit pour n≤20,
                   10s pour n≤50, 30s pour n≤100.

    Retourne
    --------
    dict :
        tournees       : list[list[int]] — clients par véhicule (sans dépôt)
        cout           : float           — distance euclidienne totale
        aretes_sol     : set[(i,j)]      — arêtes dans la solution (i<j)
        n_clients_servis : int
        faisable       : bool
    ou None si pas de solution trouvée.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n       = instance['n']
    K       = instance['n_vehicles']
    SCALE   = 10   # OR-Tools entiers — précision 0.1 minute

    dist_int = (instance['dist'] * instance['vitesse'] * SCALE).astype(np.int64)
    tw_int   = (instance['time_windows'] * SCALE).astype(np.int64)
    svc_int  = (instance['service_times'] * SCALE).astype(np.int64)
    dem_int  = instance['demands'].astype(np.int64)
    cap_int  = int(instance['capacity'])
    hor_int  = int(instance['horizon'] * SCALE)

    manager = pywrapcp.RoutingIndexManager(n + 1, K, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Coût = distance de trajet
    def dist_cb(i, j):
        return int(dist_int[manager.IndexToNode(i), manager.IndexToNode(j)])
    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # Dimension temps + time windows
    def time_cb(i, j):
        ni, nj = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(dist_int[ni, nj] + svc_int[ni])
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, hor_int, hor_int, False, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')
    for node in range(1, n + 1):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(
            int(tw_int[node, 0]), int(tw_int[node, 1])
        )

    # Dimension capacité
    def dem_cb(i):
        return int(dem_int[manager.IndexToNode(i)])
    dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [cap_int] * K, True, 'Cap'
    )

    # Clients optionnels avec pénalité élevée
    penalty = int(dist_int.max() * 10)
    for node in range(1, n + 1):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Paramètres de recherche
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None

    # Extraire les tournées
    tournees = []
    for v in range(K):
        idx   = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        tournees.append(route)

    # Calculer le coût euclidien réel
    cout = 0.0
    aretes_sol = set()
    for t in tournees:
        chemin = [0] + t + [0]
        for a, b in zip(chemin[:-1], chemin[1:]):
            cout += float(instance['dist'][a, b])
            aretes_sol.add((min(a, b), max(a, b)))

    clients_servis = {c for t in tournees for c in t}

    return {
        'tournees'         : tournees,
        'cout'             : cout,
        'aretes_sol'       : aretes_sol,
        'n_clients_servis' : len(clients_servis),
        'faisable'         : len(clients_servis) == n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACTION DES FEATURES ET LABELS POUR LE GNN
# ─────────────────────────────────────────────────────────────────────────────

def instance_vers_graphe(instance: dict, aretes_sol: set, k_nn: int = 10):
    """
    Convertit une instance et sa solution en données GNN.

    Structure du graphe : k-NN sparse (chaque nœud connecté à ses k voisins
    les plus proches). O(n·k) arêtes au lieu de O(n²).

    Retourne
    --------
    node_features : np.ndarray (N, 7)
        [x, y, demande/Q, tw_early/T, tw_late/T, service/T, est_depot]
    edge_index    : np.ndarray (2, E) — paires (src, dst) pour chaque arête
    edge_features : np.ndarray (E, 4)
        [dist/dist_max, delta_tw, compatibilite_tw, est_dans_knn]
    edge_labels   : np.ndarray (E,)   — 1 si arête dans solution, 0 sinon
    """
    n      = instance['n']
    N      = n + 1
    coords = instance['coords']
    dist   = instance['dist']
    tw     = instance['time_windows']
    dem    = instance['demands']
    svc    = instance['service_times']
    T      = instance['horizon']
    Q      = instance['capacity']

    # ── Features nœuds (N, 7) ────────────────────────────────────────────────
    node_features = np.zeros((N, 7), dtype=np.float32)
    node_features[:, 0] = coords[:, 0] / 100.0          # x normalisé
    node_features[:, 1] = coords[:, 1] / 100.0          # y normalisé
    node_features[:, 2] = dem / Q                        # demande normalisée
    node_features[:, 3] = tw[:, 0] / T                  # ouverture TW
    node_features[:, 4] = tw[:, 1] / T                  # fermeture TW
    node_features[:, 5] = svc / T                        # durée service
    node_features[0, 6] = 1.0                            # flag dépôt

    # ── Construction du graphe k-NN sparse ───────────────────────────────────
    k = min(k_nn, N - 1)
    edges_set = set()

    for i in range(N):
        # k plus proches voisins de i (en distance euclidienne)
        voisins = np.argsort(dist[i])
        cnt = 0
        for j in voisins:
            if j != i:
                edges_set.add((min(i,j), max(i,j)))
                cnt += 1
                if cnt >= k:
                    break

    edges = sorted(edges_set)
    E = len(edges)

    edge_index    = np.array(edges, dtype=np.int64).T  # (2, E)
    edge_features = np.zeros((E, 4), dtype=np.float32)
    edge_labels   = np.zeros(E, dtype=np.float32)

    dist_max = dist.max() + 1e-6

    for idx, (i, j) in enumerate(edges):
        d = dist[i, j]

        # Feature 1 : distance normalisée
        edge_features[idx, 0] = d / dist_max

        # Feature 2 : chevauchement des TW
        # Positif = les fenêtres se chevauchent, négatif = gap entre elles
        overlap = min(tw[i,1], tw[j,1]) - max(tw[i,0], tw[j,0])
        edge_features[idx, 1] = overlap / T

        # Feature 3 : compatibilité temporelle
        # Peut-on aller de i vers j dans les TW en partant de i à son ouverture ?
        t_arrivee = tw[i,0] + instance['durees'][i, j]
        compatible = 1.0 if t_arrivee <= tw[j,1] else 0.0
        edge_features[idx, 2] = compatible

        # Feature 4 : ratio demande combinée / capacité
        edge_features[idx, 3] = (dem[i] + dem[j]) / Q

        # Label : 1 si cette arête est dans la solution optimale
        edge_labels[idx] = 1.0 if (min(i,j), max(i,j)) in aretes_sol else 0.0

    return node_features, edge_index, edge_features, edge_labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. GÉNÉRATION DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generer_dataset(
    n             : int,
    n_instances   : int = 1000,
    seed_offset   : int = 0,
    time_limit_s  : float = 5.0,
    k_nn          : int = 10,
    verbose       : bool = True,
) -> list:
    """
    Génère un dataset de (graphe, labels) pour l'entraînement du GNN.

    Paramètres
    ----------
    n            : nombre de clients par instance
    n_instances  : nombre d'instances à générer
    seed_offset  : offset des seeds (éviter chevauchement train/test)
    time_limit_s : temps OR-Tools par instance
    k_nn         : voisins k-NN pour le graphe sparse
    verbose      : afficher la progression

    Retourne
    --------
    list de dict :
        {node_features, edge_index, edge_features, edge_labels,
         instance, solution}
    """
    import math
    K = max(3, math.ceil(n / 10))

    dataset  = []
    n_ok     = 0
    n_echec  = 0
    t0       = time.time()

    for i in range(n_instances):
        seed = seed_offset + i
        inst = generate_instance(n, n_vehicles=K, seed=seed)
        sol  = resoudre_instance(inst, time_limit_s=time_limit_s)

        if sol is None or not sol['faisable']:
            n_echec += 1
            if verbose and (i+1) % 100 == 0:
                print(f"  [{i+1}/{n_instances}] {n_ok} OK  {n_echec} échecs")
            continue

        nf, ei, ef, el = instance_vers_graphe(inst, sol['aretes_sol'], k_nn=k_nn)

        dataset.append({
            'node_features': nf,
            'edge_index'   : ei,
            'edge_features': ef,
            'edge_labels'  : el,
            'cout_optimal' : sol['cout'],
            'seed'         : seed,
            'n'            : n,
        })
        n_ok += 1

        if verbose and (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = n_ok / elapsed
            print(f"  [{i+1:4d}/{n_instances}] {n_ok} OK  "
                  f"{n_echec} échecs  "
                  f"{rate:.1f} inst/s  "
                  f"ETA: {(n_instances-i-1)/rate:.0f}s")

    if verbose:
        taux_positif = np.mean([d['edge_labels'].mean() for d in dataset]) * 100
        print(f"\n  Dataset n={n} : {n_ok}/{n_instances} instances valides")
        print(f"  Taux arêtes positives : {taux_positif:.1f}%")
        print(f"  Temps total : {time.time()-t0:.0f}s")

    return dataset

def sauvegarder_dataset(dataset: list, chemin: str):
    os.makedirs(os.path.dirname(chemin) or '.', exist_ok=True)
    np.save(chemin, np.array(dataset, dtype=object), allow_pickle=True)
    print(f"Dataset sauvegardé : {chemin}  ({len(dataset)} instances)")

def charger_dataset(chemin: str) -> list:
    """Charge un dataset sauvegardé."""
    data = np.load(chemin, allow_pickle=True)
    return list(data)


# ─────────────────────────────────────────────────────────────────────────────
# 4. POINT D'ENTRÉE — générer les datasets pour toutes les tailles
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import math

    DOSSIER = 'datasets_gnn'
    os.makedirs(DOSSIER, exist_ok=True)

    # Paramètres par taille
    configs = {
        10 : {'n_instances': 2000, 'time_limit_s': 3,  'seed_offset': 0},
        20 : {'n_instances': 2000, 'time_limit_s': 5,  'seed_offset': 10000},
        50 : {'n_instances': 1000, 'time_limit_s': 10, 'seed_offset': 20000},
        100: {'n_instances': 500,  'time_limit_s': 30, 'seed_offset': 30000},
        200: {'n_instances': 200,  'time_limit_s': 60, 'seed_offset': 40000},
    }

    for n, cfg in configs.items():
        print(f"\n{'='*55}")
        print(f"  Génération dataset n={n}  ({cfg['n_instances']} instances)")
        print(f"{'='*55}")

        dataset = generer_dataset(
            n           = n,
            n_instances = cfg['n_instances'],
            seed_offset = cfg['seed_offset'],
            time_limit_s= cfg['time_limit_s'],
            k_nn        = 10,
            verbose     = True,
        )

        chemin = os.path.join(DOSSIER, f'dataset_n{n}.npy')
        sauvegarder_dataset(dataset, chemin)
