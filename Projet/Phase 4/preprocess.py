import numpy as np

def generate_instance(n, n_vehicles=3, seed=42):
    """
    Génère une instance aléatoire du VRPTW.
    @param n : nombre de clients (hors dépôt)
    @param n_vehicles : nombre de véhicules disponibles
    @param seed : int, graine pour la reproductibilité
    @return : dict avec les données de l'instance
    """
    rng     = np.random.default_rng(seed)
    horizon = 480.0  # 8 heures en minutes

    # Facteur de conversion distance → minutes
    vitesse = 0.6  # minutes par unité de distance

    # Coordonnées
    depot   = np.array([[50.0, 50.0]])
    clients = rng.uniform(0, 100, size=(n, 2))
    coords  = np.concatenate([depot, clients], axis=0)

    # Matrice des distances euclidiennes
    diff  = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist  = np.sqrt((diff ** 2).sum(axis=2))

    # Matrice des durées de trajet en minutes
    durees = dist * vitesse

    # Demandes
    demands     = np.zeros(n + 1)
    demands[1:] = rng.integers(1, 31, size=n).astype(float)

    # Capacité
    total_demand = demands.sum()
    capacity     = total_demand / n_vehicles * 1.2

    # Durées de service
    service_times      = np.zeros(n + 1)
    service_times[1:]  = rng.uniform(5, 15, size=n)

    # Fenêtres temporelles : indépendantes des distances
    # Trois profils de clients simulant des comportements réels :
    #   - Strict  (40%) : créneau de 30 à 60 min, heure imposée
    #   - Modéré  (40%) : créneau de 90 à 150 min, demi-journée
    #   - Large   (20%) : créneau de 200 à 300 min, quasi disponible
    time_windows    = np.zeros((n + 1, 2))
    time_windows[0] = [0, horizon]  # dépôt : toute la journée

    profils = rng.choice(['strict', 'modere', 'large'],
                         size=n,
                         p=[0.4, 0.4, 0.2])

    for i in range(1, n + 1):
        profil = profils[i - 1]

        if profil == 'strict':
            # Créneau court : 30 à 60 min
            largeur = rng.uniform(30, 60)
            # Heure d'ouverture répartie sur toute la journée
            # en laissant de la place pour la durée du créneau
            a_i = rng.uniform(0, horizon - largeur)

        elif profil == 'modere':
            # Créneau moyen : 90 à 150 min
            largeur = rng.uniform(90, 150)
            a_i     = rng.uniform(0, horizon - largeur)

        else:  # large
            # Créneau large : 200 à 300 min
            largeur = rng.uniform(200, 300)
            a_i     = rng.uniform(0, horizon - largeur)

        b_i = min(horizon, a_i + largeur)
        time_windows[i] = [round(a_i, 1), round(b_i, 1)]

    return {
        'n'             : n,
        'n_vehicles'    : n_vehicles,
        'coords'        : coords,
        'dist'          : dist,
        'durees'        : durees,
        'demands'       : demands,
        'capacity'      : round(capacity, 1),
        'time_windows'  : time_windows,
        'service_times' : service_times,
        'horizon'       : horizon,
        'vitesse'       : vitesse,
        'seed'          : seed,
    }

def preprocess(instance):
    """
    Convertit une instance brute en tenseurs normalisés.
    Retourne node_features (n+1, 6) et les scalaires utiles.
    """
    T = instance['horizon']       # 480.0
    Q = instance['capacity']

    coords   = instance['coords']        / 100.0      # (n+1, 2)
    demands  = instance['demands'][:, None] / Q       # (n+1, 1)
    tw       = instance['time_windows']  / T          # (n+1, 2)
    service  = instance['service_times'][:, None] / T # (n+1, 1)

    # Shape finale : (n+1, 6)
    # index 0 = dépôt, index 1..n = clients
    node_features = np.concatenate([coords, demands, tw, service], axis=1)

    return node_features