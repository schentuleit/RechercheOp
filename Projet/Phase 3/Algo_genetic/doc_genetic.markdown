#Rapport de Recherche Opérationnelle : Algorithme Génétique

## 1. Introduction
Bienvenue dans ce Notebook dédié à la résolution du problème de **tournées de véhicules avec fenêtres temporelles (VRPTW)**. Pour répondre aux enjeux de l'ADEME, nous avons implémenté une métaheuristique puissante : l'**Algorithme Génétique**.

## 2. Pourquoi la Métaheuristique ?
Le problème de livraison est un casse-tête mathématique dit **NP-Difficile**. Cela signifie que :
* Si on a 5 clients, c'est facile.
* Si on a 50 clients, il y a plus de trajets possibles que de grains de sable sur Terre.

Une métaheuristique ne cherche pas la solution "parfaite" (qui prendrait des années à calculer), mais trouve une **excellente solution** en quelques secondes.



## 3. La Méthode : L'Évolution Naturelle
L'algorithme génétique imite la théorie de l'évolution de Darwin. Voici notre processus :

### A. Création de la Population
On génère au départ une cinquantaine de solutions au hasard. Certaines sont médiocres, d'autres sont un peu plus logiques.

### B. La Sélection des Champions
On évalue chaque trajet. Un trajet reçoit une **bonne note** s'il est court. Il reçoit une **pénalité (mauvaise note)** s'il arrive en retard chez un client ou s'il dépasse la capacité du camion.

### C. Le Croisement (Crossover)
On prend deux trajets performants (les parents) pour créer un nouveau trajet (l'enfant).
* **L'idée :** L'enfant récupère les meilleurs quartiers du Papa et les meilleures séquences de la Maman.
* **Exemple :** Si Papa est efficace sur Paris-Nord et Maman sur Paris-Sud, l'enfant sera peut-être le champion de tout Paris.



### D. La Mutation
De temps en temps, on inverse deux villes dans un trajet au hasard. 
* **Pourquoi ?** Pour éviter que tous nos camions fassent la même chose. Cela permet d'explorer des chemins totalement nouveaux.

## 4. Résultats et Analyse
Grâce à cette approche, notre flotte de véhicules s'adapte dynamiquement aux contraintes :
1. **Économie d'énergie :** Réduction des kilomètres parcourus.
2. **Respect des horaires :** Les pénalités forcent l'algorithme à respecter les fenêtres de temps des clients.
3. **Capacité optimisée :** Les camions sont remplis de manière équilibrée.

## 5. Conclusion pour l'ADEME
L'algorithme génétique est particulièrement robuste pour les territoires complexes. Il permet de trouver des solutions durables et scalables, répondant directement aux objectifs de réduction des émissions de gaz à effet de serre.
