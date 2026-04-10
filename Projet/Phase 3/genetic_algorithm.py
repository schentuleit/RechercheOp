"""
genetic_algorithm.py — Algorithme Génétique pour le VRPTW
===========================================================
Simple et lisible, basé sur les fonctions du recuit simulé.
"""

import numpy as np
import time
import copy
from simulated_annealing import solution_initiale, cout_penalise, est_valide


# PARTIE 1 : Initialisation de la population

def creer_population(instance, taille_pop):
    """
    Crée une population initiale de solutions.
    
    @param instance    : dict généré par generate_instance()
    @param taille_pop  : int — nombre d'individus dans la population
    @return : list of routes — ['ind1', 'ind2', ...]
    """
    population = []
    for _ in range(taille_pop):
        # On crée une nouvelle solution aléatoire (enfant différent à chaque fois)
        individu = solution_initiale(instance)
        population.append(individu)
    return population


# PARTIE 2 : Évaluation et sélection des meilleurs

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
    # On choisit les meilleurs
    indices_tries = np.argsort(couts)  # Tri par coût croissant (moins cher = meilleur)
    indices_les_meilleurs = indices_tries[:k]
    
    meilleurs = [population[i] for i in indices_les_meilleurs]
    meilleurs_couts = [couts[i] for i in indices_les_meilleurs]
    
    return meilleurs, meilleurs_couts


# PARTIE 3 : Croisement (Crossover)

def croisement_simple(parent1, parent2, instance):
    """
    Ici on fait le mélange : le bébé prend des routes du parent1,
    puis on complète avec les clients manquants dans l'ordre du parent2.
    
    @param parent1  : list of lists — routes du parent 1
    @param parent2  : list of lists — routes du parent 2
    @param instance : dict
    @return : list of lists — l'enfant (nouvelle solution)
    """
    n = instance['n']  # nombre de clients (sans le dépôt)
    K = instance['n_vehicles']  # nombre de véhicules
    
    # On va construire l'enfant progressivement
    enfant = [[] for _ in range(K)]
    clients_assigns = set()
    
    # Étape 1 : copier une partie aléatoire du parent1
    # On prend environ 50% des routes du parent1
    for k in range(K):
        if np.random.random() < 0.5:
            # On copie cette route complète
            enfant[k] = parent1[k].copy()
            clients_assigns.update(parent1[k])
    
    # Étape 2 : chercher les clients manquants et les placer grâce au parent2
    clients_manquants = set(range(1, n + 1)) - clients_assigns
    
    for client in clients_manquants:
        # On cherche où est ce client dans le parent2
        for k_parent2, route_parent2 in enumerate(parent2):
            if client in route_parent2:
                # Trouver une place dans l'enfant
                # On cherche le véhicule avec le moins de charge (glouton simple)
                k_enfant = np.argmin([len(enfant[k]) for k in range(K)])
                enfant[k_enfant].append(client)
                clients_assigns.add(client)
                break
    
    return enfant


# PARTIE 4 : Mutation

def mutation_legere(routes, instance):
    """
    On fait une petite modification : déplacer un client vers une autre route.
    C'est une mutation légère pour explorer le voisinage.
    
    @param routes   : list of lists — la solution à modifier
    @param instance : dict
    @return : list of lists — la solution mutée
    """
    enfant = copy.deepcopy(routes)
    K = len(enfant)
    
    # Routes non vides
    routes_avec_clients = [k for k in range(K) if len(enfant[k]) > 0]
    
    if len(routes_avec_clients) < 1:
        return enfant
    
    # Choisir une route d'où on va prendre un client
    k_src = np.random.choice(routes_avec_clients)
    
    if len(enfant[k_src]) == 0:
        return enfant
    
    # Prendre un client aléatoire
    idx_client = np.random.randint(0, len(enfant[k_src]))
    client = enfant[k_src].pop(idx_client)
    
    # Choisir une route destination (peut être différente ou la même)
    k_dst = np.random.randint(0, K)
    
    # Insérer le client avec un peu d'ordre (au début ou à la fin pour plus de chances)
    if len(enfant[k_dst]) == 0 or np.random.random() < 0.5:
        enfant[k_dst].insert(0, client)
    else:
        enfant[k_dst].append(client)
    
    return enfant


# PARTIE 5 : L'algorithme génétique principal

def algorithme_genetique(instance, taille_pop=20, n_generations=100, 
                        taux_mutation=0.3, seed=None):
    """
    Exécute l'algorithme génétique.
    
    @param instance         : dict généré par generate_instance()
    @param taille_pop       : int — taille de la population
    @param n_generations    : int — nombre de générations
    @param taux_mutation    : float — probabilité de mutation (0 à 1)
    @param seed             : int — graine pour la reproductibilité
    @return : (meilleure_solution, meilleur_cout, historique)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_debut = time.time()
    
    # Initialisation
    print(" Création de la population initiale...")
    population = creer_population(instance, taille_pop)
    
    # Historique pour voir l'évolution
    historique = {'couts_generations': []}
    meilleure_solution = None
    meilleur_cout = float('inf')
    
    # Générations
    for gen in range(n_generations):
        # On évalue tout le monde
        couts = evaluer_population(population, instance)
        
        # On garde le meilleur si c'est mieux qu'avant
        meilleur_cout_gen = min(couts)
        if meilleur_cout_gen < meilleur_cout:
            meilleur_cout = meilleur_cout_gen
            idx_meilleur = np.argmin(couts)
            meilleure_solution = copy.deepcopy(population[idx_meilleur])
        
        historique['couts_generations'].append(meilleur_cout_gen)
        
        if (gen + 1) % 10 == 0:
            print(f"  Génération {gen + 1}/{n_generations} — Meilleur coût : {meilleur_cout_gen:.2f}")
        
        # Sélection des meilleurs (les parents)
        nb_parents = max(2, taille_pop // 3)  # Environ 33% sont parents
        parents, _ = selectionner_meilleurs(population, couts, nb_parents)
        
        # Création de la nouvelle génération
        nouvelle_population = []
        
        # On garde toujours les meilleurs (élitisme)
        nouvelle_population.extend([copy.deepcopy(p) for p in parents[:2]])
        
        # On crée de nouveaux enfants par croisement
        while len(nouvelle_population) < taille_pop:
            # Choisir deux parents au hasard
            idx1, idx2 = np.random.choice(len(parents), size=2, replace=True)
            parent1 = parents[idx1]
            parent2 = parents[idx2]
            
            # Croisement pour créer un enfant
            enfant = croisement_simple(parent1, parent2, instance)
            
            # Mutation avec probabilité taux_mutation
            if np.random.random() < taux_mutation:
                enfant = mutation_legere(enfant, instance)
            
            nouvelle_population.append(enfant)
        
        # La nouvelle population devient la population actuelle
        population = nouvelle_population[:taille_pop]
    
    t_fin = time.time()
    
    print(f"\n Algorithme terminé en {t_fin - t_debut:.2f}s")
    print(f" Meilleure solution trouvée : coût = {meilleur_cout:.2f}")
    
    return meilleure_solution, meilleur_cout, historique


# PARTIE 6 : Exemple d'utilisation

if __name__ == "__main__":
    import sys
    import os
    
    # Import de generate_instance depuis Phase 4
    PHASE4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Phase 4')
    sys.path.insert(0, PHASE4)
    from preprocess import generate_instance
    
    # Créer une instance de test
    print(" Génération d'une instance VRPTW...")
    instance = generate_instance(n_clients=20, n_vehicles=3, seed=42)
    
    # Lancer l'algorithme génétique
    print("\n Lancement de l'algorithme génétique...\n")
    solution, cout, historique = algorithme_genetique(
        instance,
        taille_pop=20,
        n_generations=50,
        taux_mutation=0.3,
        seed=42
    )
    
    # Afficher quelques infos
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"Coût final        : {cout:.2f}")
    print(f"Nombre de routes  : {len([r for r in solution if r])}")
    print(f"Routes            : {solution}")
