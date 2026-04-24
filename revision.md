# Explication Complète du Projet — VRPTW & Méthodes d'Optimisation

---

## PARTIE 1 — Le Contexte et la Problématique

### 1.1 D'où vient ce projet ?

Ce projet répond à un appel de l'**ADEME** — l'Agence de la transition écologique (anciennement Agence De l'Environnement et de la Maîtrise de l'Énergie). L'ADEME finance des projets qui réduisent les émissions de CO₂. Dans ce contexte, l'enjeu est la **mobilité intelligente** : comment optimiser les trajets de livraison pour qu'ils consomment moins de carburant, parcourent moins de kilomètres, et utilisent moins de véhicules.

Applications concrètes : livraison de colis, collecte de déchets, maintenance d'équipements urbains, transport de personnes en zones peu denses.

### 1.2 Le problème en langage courant

Imaginez que vous êtes directeur logistique d'un entrepôt Amazon. Chaque matin, vous avez :
- **N clients** à livrer dans la journée
- **K camions** disponibles, chacun avec une capacité maximale (ils ne peuvent pas transporter plus que X kg)
- Chaque client a commandé une certaine quantité (sa **demande**)
- Chaque client a spécifié un **créneau horaire** ("livrez-moi entre 10h et 11h maximum")
- La journée se termine à 16h00 (les camions doivent être rentrés)

**Question** : Qui livre qui, dans quel ordre, pour parcourir le **minimum de kilomètres** au total ?

C'est exactement le **VRPTW** — *Vehicle Routing Problem with Time Windows*.

---

## PARTIE 2 — Les Fondements Mathématiques

### 2.1 Qu'est-ce qu'un graphe ?

Un **graphe** est une structure mathématique qui représente des objets et leurs relations.

```
Objets = "sommets" (ou nœuds)
Relations = "arêtes" (ou arcs)
```

**Exemple** : Un réseau de villes où chaque ville est un sommet, et chaque route entre deux villes est une arête avec un poids (la distance).

On distingue :
- **Graphe non orienté** : les arêtes sont symétriques (Paris → Lyon = Lyon → Paris)
- **Graphe orienté** (ou *digraphe*) : les arcs ont un sens (sens unique, coûts asymétriques)
- **Graphe valué** (ou *pondéré*) : chaque arête/arc porte un poids (distance, temps, coût)

### 2.2 Graphe incomplet, graphe complet — le choix de modélisation

Un **graphe incomplet** est un graphe où certaines paires de sommets ne sont pas reliées directement. Un réseau routier réel est incomplet : on ne peut pas aller directement de l'entrepôt à chaque client en ligne droite — il faut suivre les routes.

Un **graphe complet** ($K_n$) est un graphe où **chaque sommet est directement relié à tous les autres**. Pour $n$ sommets, un graphe complet a $\frac{n(n-1)}{2}$ arêtes non orientées, ou $n(n-1)$ arcs orientés. Pour $n=50$, ça fait déjà **2 450 arcs** possibles.

**Pourquoi passer du graphe incomplet au graphe complet ?**

En pratique, les routes réelles forment un graphe incomplet. Mais notre modélisation *calcule la distance Euclidienne* entre chaque paire de clients (coordonnées x, y) comme approximation du chemin le plus court. Cela crée une **matrice de distances complète** $C = (c_{ij})_{i,j \in V}$ où :

$$c_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

Le passage graphe incomplet → complet est donc un **choix de modélisation** qui :
1. Simplifie la formulation mathématique (tout arc existe)
2. Reste valide car la distance Euclidienne satisfait l'inégalité triangulaire : $c_{ik} \leq c_{ij} + c_{jk}$
3. Signifie qu'on suppose qu'il est toujours possible d'aller directement d'un point à un autre

**Limitation** : Dans un contexte réel (réseau routier avec sens interdits, bouchons), la matrice serait issue d'un calcul d'itinéraire (API Google Maps, OSRM) plutôt que Euclidienne.

### 2.3 Qu'est-ce que la complexité algorithmique ?

La **complexité** d'un algorithme mesure comment le temps de calcul évolue quand la taille du problème augmente.

- **O(n)** : Si n double, le temps double → linéaire, très bon
- **O(n²)** : Si n double, le temps quadruple → quadratique, acceptable
- **O(n!)** : Si n augmente de 1, le temps est multiplié par n → factorielle, catastrophique

### 2.4 Qu'est-ce que NP-difficile ? (La notion centrale)

**D'abord : P vs NP**

En informatique théorique, on classe les problèmes selon leur difficulté :
- **Classe P** : Problèmes que l'on peut **résoudre** en temps polynomial (O(n^k)). Ex : trier une liste, trouver le plus court chemin dans un graphe.
- **Classe NP** : Problèmes dont on peut **vérifier une solution** en temps polynomial, mais dont on ne sait pas les *résoudre* efficacement. Ex : si quelqu'un vous donne une tournée, vous pouvez vérifier en O(n) qu'elle est valide — mais trouver la meilleure tournée prend exponentiellement plus de temps.

**NP-complet** : Un problème NP qui est "le plus dur possible" dans NP. Si on savait en résoudre un en temps polynomial, on pourrait résoudre *tous* les problèmes NP en temps polynomial (ce qui prouverait P=NP, l'une des grandes conjectures non résolues). Karp (1972) a listé les 21 premiers problèmes NP-complets, incluant le TSP et le *Circuit Hamiltonien*.

**NP-difficile** : Au moins aussi dur qu'un problème NP-complet. Le VRPTW est NP-difficile car le **TSP** (problème du voyageur de commerce) est un cas particulier du VRPTW — et le TSP est NP-complet.

**La preuve par réduction** (pour le jury) :

1. Le **TSP** demande : "Trouver la tournée la plus courte visitant une fois chaque ville." C'est un cas particulier du VRPTW avec : 1 seul véhicule, capacité infinie, aucune fenêtre temporelle.
2. Si on pouvait résoudre le VRPTW en temps polynomial, on pourrait résoudre le TSP en temps polynomial → contradiction avec le fait que TSP est NP-complet.
3. Donc VRPTW est **au moins aussi difficile** que TSP → NP-difficile.

**La chaîne de réductions complète** :
```
Circuit Hamiltonien  →  TSP  →  CVRP  →  VRPTW
(NP-complet, Karp 1972)       (+ capacité) (+ fenêtres temporelles)
```
Chaque flèche est une réduction polynomiale : ajouter des contraintes ne rend le problème que plus difficile.

**Conséquence pratique** :

| n clients | Tournées possibles | Temps à 10⁹ op/s |
|---|---|---|
| 10 | 181 440 | < 1 milliseconde |
| 15 | 43 milliards | 43 secondes |
| 20 | 60 quadrillions | 2 000 ans |
| 50 | ≈ 3 × 10⁶⁴ | âge de l'univers × 10⁴² |

**Conclusion** : Pour n ≥ 20, il est **mathématiquement impossible** de tester toutes les solutions. On doit utiliser des méthodes approchées (heuristiques, métaheuristiques, deep learning).

### 2.5 Taxonomie des approches algorithmiques

Il existe quatre grandes familles de méthodes pour résoudre les problèmes d'optimisation combinatoire comme le VRPTW :

#### Méthodes exactes
Garantissent l'optimalité mais ont une complexité exponentielle en pratique.

| Méthode | Principe | Limite |
|---|---|---|
| **Branch & Bound** | Exploration arborescente avec bornes | Exponentiel, viable n ≤ 15 |
| **Branch & Cut** | B&B + coupes valides (Gomory, etc.) | Viable n ≤ 50 avec effort |
| **Branch & Price** | Génération de colonnes + B&B | Meilleur pour VRPTW exact |
| **Programmation Dynamique** | Bellman-Held-Karp pour TSP en O(2ⁿ·n) | Infaisable au-delà de n=25 |
| **CP-SAT (OR-Tools)** | Satisfaction de contraintes + SAT | **Notre oracle**, industriel |

#### Heuristiques constructives
Construisent une solution de zéro, rapides mais sans garantie de qualité.

| Méthode | Principe |
|---|---|
| **NNH** (notre Phase 2) | Plus proche voisin à chaque étape |
| **Algorithme de Clarke-Wright** | Économies de fusion de routes |
| **Greedy insertion** | Insérer le client le moins coûteux |
| **Sweep** | Balayage angulaire, regroupement géographique |

#### Métaheuristiques
Frameworks généraux pour l'optimisation, guidant la recherche locale.

| Méthode | Mécanisme clé | Notre usage |
|---|---|---|
| **Recuit Simulé (SA)** | Acceptation probabiliste de solutions pires | Phase 3 |
| **Algo Génétique (GA)** | Évolution de population, croisement, mutation | Phase 3 |
| **Recherche Tabou** | Mémoire des mouvements récents (interdits) | Non implémenté |
| **ALNS** | Destruction partielle + réinsertion adaptative | Non implémenté |
| **VNS** | Plusieurs voisinages imbriqués | Non implémenté |

#### Apprentissage automatique (Deep Learning)
Apprennent à résoudre à partir d'exemples, inférence quasi-instantanée.

| Méthode | Article de référence |
|---|---|
| **Pointer Network** | Vinyals et al. (2015) — premier réseau séquence-à-séquence pour TSP |
| **Attention Model (AM)** | Kool et al. (2019) — notre Phase 4 |
| **POMO** | Kwon et al. (2020) — départs multiples, état de l'art |
| **L2D** | Li et al. (2021) — apprentissage de décomposition |

---

## PARTIE 3 — La Modélisation Formelle du VRPTW

### 3.1 Les variables de décision

En optimisation, une **variable de décision** est ce qu'on cherche à déterminer.

$$x_{ij}^k \in \{0, 1\}$$

- $x_{ij}^k = 1$ si le véhicule $k$ va du client $i$ au client $j$
- $x_{ij}^k = 0$ sinon

$$\tau_i^k \in \mathbb{R}^+$$

- $\tau_i^k$ = l'heure à laquelle le véhicule $k$ arrive au client $i$

### 3.2 Les contraintes C1 à C5

**C1 — Couverture** (chaque client livré exactement une fois) :
$$\sum_{k \in K} \sum_{j \in V} x_{ij}^k = 1 \quad \forall i \neq 0$$

*En français* : La somme de tous les arcs arrivant au client $i$ (pour tous les véhicules) vaut 1. Personne n'est oublié, personne n'est livré deux fois.

**C2 — Flux** (les routes sont fermées, dépôt→clients→dépôt) :
$$\sum_j x_{0j}^k = \sum_j x_{j0}^k \leq 1 \quad \forall k$$

*En français* : Chaque véhicule quitte le dépôt une fois et y revient une fois.

**C3 — Capacité** (le camion ne déborde pas) :
$$\sum_{i \in \text{route}_k} d_i \leq Q \quad \forall k$$

*En français* : La somme des demandes des clients servis par le véhicule $k$ ne dépasse pas la capacité $Q$.

**C4 — Fenêtres temporelles** (arrivée dans le créneau) :
$$a_i \leq \tau_i^k \leq b_i \quad \forall i, k$$

*En français* : On ne peut pas arriver après la fermeture ($b_i$). Arriver trop tôt → on attend. Arriver trop tard → refusé.

**C5 — Horizon journalier** (retour avant fin de journée) :
$$\tau_{\text{retour}}^k \leq H = 480 \text{ min}$$

*En français* : Le véhicule doit être rentré au dépôt avant la fin de la journée (8h de travail).

### 3.3 La fonction objectif

$$\min \sum_{k \in K} \sum_{i \in V} \sum_{j \in V} c_{ij} \cdot x_{ij}^k$$

*En français* : Minimiser la somme des distances parcourues par tous les véhicules sur tous leurs arcs.

### 3.4 Le générateur d'instances (`preprocess.py`)

Pour tester nos algorithmes, on a besoin d'instances (= cas concrets). Le générateur crée des problèmes aléatoires mais reproductibles grâce aux **seeds** (graines aléatoires).

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| Nombre de véhicules | $K = \max(3, \lceil n/10 \rceil)$ | Empiriquement : garantit faisabilité |
| Capacité | $Q = 1.4 \times \frac{\sum d_i}{K}$ | +40% de marge sur la demande moyenne |
| Fenêtres TW | largeur $\sim U(60, 90)$ min | Réaliste, ancrées sur temps min depuis dépôt |
| Horizon | $H = 480$ min | 8 heures de travail |
| Coordonnées | $\sim U([0, 100]^2)$ | Carte 100×100 unités |

**Pourquoi des instances synthétiques ?** Les instances réelles (fichiers Solomon 1987, CVRPLIB) sont des benchmarks figés. Nos instances synthétiques permettent de contrôler la difficulté et d'avoir exactement les tailles souhaitées.

**Taille acceptable vs taille réelle** : En logistique opérationnelle, n peut aller de 20 (petit transporteur) à 10 000+ (Amazon dans une grande métropole). Nos tailles de test (n=10 à 200) couvrent les cas où les résultats sont différentiables entre méthodes.

### 3.5 Plan d'expérience

**Objectif** : Comparer équitablement 4 paradigmes algorithmiques sur les mêmes instances avec les mêmes métriques.

**Facteurs** :
- Taille des instances : $n \in \{10, 20, 50, 100, 200\}$ — progression géométrique couvrant petite, moyenne, grande échelle
- Seeds : 30 seeds par taille (seeds 0 à 29), reproductibles

**Justification des 30 seeds** : Le *Théorème Central Limite* garantit qu'avec $n \geq 30$ observations, la distribution empirique de la moyenne est approximativement normale. Cela permet des **intervalles de confiance à 95%** valides. Avec seulement 5 seeds, la variance serait trop large.

**Métriques mesurées** :
| Métrique | Formule | Interprétation |
|---|---|---|
| Gap vs oracle | $(c_{\text{algo}} - c^*) / c^* \times 100\%$ | Éloignement de l'optimal |
| Temps d'exécution | Médian sur 30 seeds | Robustesse aux outliers |
| Taux de validité | % solutions respectant C1–C5 | Fiabilité opérationnelle |
| Coût moyen | $\bar{c}_{\text{algo}}$ | Qualité absolue |
| IC 95% | $\bar{x} \pm t_{0.975, 29} \times s/\sqrt{30}$ | Précision statistique |

**Total** : 5 tailles × 30 seeds = **150 instances** résolues par 4 méthodes + oracle = **750 résolutions**.

---

## PARTIE 4 — Phase 2 : NNH + 2-opt (La Méthode Heuristique)

### 4.1 L'heuristique NNH (*Nearest Neighbor Heuristic*)

**NNH** = "Plus proche voisin" en français. C'est une **heuristique constructive gloutonne**.

**Glouton** (greedy en anglais) signifie : à chaque étape, prendre la meilleure décision *immédiate*, sans se soucier des conséquences futures. Comme quelqu'un qui mange toujours le meilleur morceau dans son assiette sans penser qu'il lui restera peut-être les moins bons pour la fin.

**Algorithme pas à pas** :
```
1. Le camion part du dépôt, heure = 0, charge = 0
2. Regarder tous les clients non encore livrés
3. Parmi ceux qui sont accessibles (capacité OK, créneau OK, retour possible),
   choisir LE PLUS PROCHE (distance minimale)
4. Aller vers ce client, mettre à jour heure et charge
5. Répéter étape 2 jusqu'à ce qu'aucun client ne soit accessible
6. Rentrer au dépôt → Route du camion 1 terminée
7. Ouvrir un nouveau camion, recommencer avec les clients restants
8. Répéter jusqu'à ce que tous les clients soient livrés
```

**Complexité** : O(n²) par route, O(n²) en pratique au total.

**Limite fondamentale** : NNH ne revient jamais en arrière. Si à une étape on a fait un mauvais choix, on ne peut pas corriger. C'est pourquoi NNH peut sur-utiliser les véhicules.

### 4.2 L'amélioration 2-opt

Une fois que NNH a construit une solution, on la **raffine** avec 2-opt.

**Intuition** : Imaginez une tournée qui fait des croisements (comme deux fils enchevêtrés). En "décroisant" les arcs, on raccourcit le chemin total.

```
Avant :  Dépôt → A → B → C → D → Dépôt  (avec croisement entre B→C et A→D)
Après :  Dépôt → A → C → B → D → Dépôt  (segment [B,C] inversé → plus court)
```

**Formule du gain** pour inverser le segment entre les positions $i$ et $j$ :
$$\Delta(i,j) = \underbrace{c(r_{i-1}, r_i) + c(r_j, r_{j+1})}_{\text{distance avant}} - \underbrace{c(r_{i-1}, r_j) + c(r_i, r_{j+1})}_{\text{distance après}}$$

Si $\Delta > 0$ → raccourcit la route → on accepte, **seulement si** C3, C4, C5 restent respectées.

**Pourquoi le 2-opt est limité ici** : Les fenêtres temporelles sont strictes. Inverser un segment change l'ordre de visite → certains clients seraient visités hors créneau → inversion rejetée. Gain moyen : seulement **0.06 à 0.96%**.

### 4.3 Résultats NNH + 2-opt

| n | Gap vs optimal (oracle) | Temps |
|---|---|---|
| 10 | ~31 % | < 1 ms |
| 20 | ~49 % | < 1 ms |
| 50 | ~63 % | < 3 ms |
| 100 | ~70 % | < 1 ms |
| 200 | ~65 % | ~2 ms |

Force : **instantané** et **100% de solutions valides**. Limite : 30–70% au-dessus de l'optimal.

---

## PARTIE 5 — Phase 3 : Les Métaheuristiques

### 5.1 Pourquoi "méta" heuristique ?

Une **métaheuristique** est un cadre général qui guide la recherche de solutions en utilisant des mécanismes intelligents pour explorer l'espace des solutions et s'échapper des optima locaux. Le préfixe "méta" vient du grec = "au-delà" — au-delà des simples heuristiques.

La différence clé entre heuristique et métaheuristique :
- **Heuristique** : stratégie simple, construire ou améliorer une solution, souvent sans mémoire
- **Métaheuristique** : mécanisme de pilotage de la recherche (probabilité, population, mémoire), capable de s'échapper des optima locaux

### 5.2 Le Recuit Simulé (*Simulated Annealing*, SA)

**L'analogie physique** : En métallurgie, le **recuit** consiste à chauffer un métal jusqu'à une température élevée puis à le refroidir lentement. À haute température, les atomes bougent beaucoup et peuvent quitter des configurations sous-optimales. En refroidissant doucement, ils s'installent dans une configuration de basse énergie optimale. (Kirkpatrick, Gelatt & Vecchi, 1983)

**L'algorithme** :
```
solution ← NNH (initialisation)
T ← T₀  (température initiale élevée)

Répéter jusqu'à convergence :
    voisin ← perturber(solution)   (inverser un segment, déplacer un client)
    Δ ← coût(voisin) - coût(solution)
    
    Si Δ < 0  →  accepter voisin  (solution meilleure, toujours acceptée)
    Si Δ ≥ 0  →  accepter avec probabilité exp(-Δ/T)  (solution pire, parfois acceptée)
    
    T ← T × α  (refroidissement : α ≈ 0.995)
```

**L'équation clé** : $P(\text{accepter une solution pire}) = e^{-\Delta/T}$

- Si $T$ grand → probabilité ≈ 1 → on explore largement
- Si $T$ petit → probabilité ≈ 0 → on n'accepte que les améliorations

**Paramètres du Recuit Simulé et impact sur la convergence** :

| Paramètre | Valeur utilisée | Trop faible | Trop élevé |
|---|---|---|---|
| $T_0$ (temp. initiale) | $\sim 1000$ | Bloqué dès le départ | Trop de mouvements aléatoires |
| $\alpha$ (cooling rate) | $0.995$ | Refroidissement brutal, convergence prématurée | Trop d'itérations nécessaires |
| Nb itérations par T | $100 \times n$ | Mauvaise exploration | Temps de calcul trop long |
| $\lambda_C$ (pénalité capa) | $1000$ | Violations capacité non corrigées | Espace trop contraint |
| $\lambda_T$ (pénalité temps) | $500$ | Violations TW non corrigées | Espace trop contraint |

**Convergence** : Le SA converge théoriquement vers l'optimal global si $T$ décroît suffisamment lentement (refroidissement logarithmique). En pratique, on utilise un refroidissement géométrique ($T \leftarrow T\alpha$) qui est plus rapide mais ne garantit pas l'optimal.

**Soft constraints** : Contrairement à NNH, le SA accepte temporairement des violations de contraintes en les **pénalisant** dans le coût :
$$\text{coût total} = \text{distance} + \lambda_C \times \text{violation\_capacité} + \lambda_T \times \text{violation\_temps}$$

Cela permet d'explorer des zones que NNH ne peut jamais atteindre.

**Résultats SA** :

| n | Gap vs optimal | Temps |
|---|---|---|
| 10 | ~0.3 % | ~1 s |
| 20 | ~19.5 % | ~1.5 s |
| 50 | ~36.8 % | ~2.9 s |
| 100 | ~78 % | ~5.4 s |
| 200 | ~156 % | ~10.5 s |

Excellent pour $n \leq 50$, se dégrade ensuite (espace trop grand, temps insuffisant).

### 5.3 L'Algorithme Génétique (*Genetic Algorithm*, GA)

**L'analogie biologique** : Darwin a montré que les espèces évoluent par **sélection naturelle**. L'algorithme génétique reproduit ce mécanisme sur une *population* de solutions. (Holland, 1975)

**Pipeline** :
```
1. INITIALISATION : P solutions aléatoires (chaque "individu" = une tournée)
2. ÉVALUATION    : Calculer le fitness (coût pénalisé) de chaque individu
3. SÉLECTION     : Choisir parents par tournoi (prendre le meilleur sur 2 aléatoires)
4. CROSSOVER     : Combiner deux parents (PMX : transférer un segment + compléter)
5. MUTATION      : Modifier aléatoirement (taux 20-30%) → diversité
6. NOUVELLE GÉN. : Garder les P meilleurs (élitisme : toujours conserver le meilleur)
7. Répéter 2→6 pendant N générations
```

**Le crossover PMX** (*Partially Mapped Crossover*) : Choisir un segment du parent 1, le transférer au fils, puis remplir les positions restantes par les éléments du parent 2 qui ne créent pas de doublon. Garantit qu'aucun client n'est visité deux fois.

**Paramètres du GA et impact** :

| Paramètre | Valeur utilisée | Impact |
|---|---|---|
| Taille population $P$ | $40$ | Trop petit → diversité faible ; trop grand → coûteux |
| Taux croisement $p_c$ | $0.8$ | Trop bas → peu d'échange génétique |
| Taux mutation $p_m$ | $0.2$–$0.3$ | Trop bas → convergence prématurée ; trop haut → marche aléatoire |
| Nb générations | $200$ | Trop peu → pas convergé ; trop → temps de calcul |
| Taille tournoi | $2$ | Plus grand → pression de sélection plus forte |

**Convergence prématurée** : Quand tous les individus de la population se ressemblent trop (diversité génétique → 0), le GA ne peut plus s'améliorer. Indicateur : l'écart-type du fitness population → 0. Solution : augmenter $p_m$ ou réintroduire des individus aléatoires.

**Résultats GA** :

| n | Gap vs optimal | Temps |
|---|---|---|
| 10 | ~16 % | ~0.35 s |
| 20 | ~36.6 % | ~0.71 s |
| 50 | ~90.8 % | ~2.9 s |
| 100 | ~123 % | ~9 s |
| 200 | ~192 % | ~21 s |

Compétitif pour $n \leq 20$, diverge fortement ensuite (convergence prématurée de la population).

---

## PARTIE 6 — Phase 4 : Le Deep Learning (*Attention Model*)

### 6.1 Pourquoi le Deep Learning ?

Les métaheuristiques **itèrent** à chaque résolution (des milliers d'opérations par instance). L'idée du Deep Learning : **apprendre à construire de bonnes solutions** à partir d'exemples, pour que l'inférence soit quasi-instantanée — comme la différence entre quelqu'un qui cherche laborieusement aux échecs toutes les combinaisons, et un Grand Maître qui "voit" directement la bonne réponse grâce à son expérience accumulée.

### 6.2 Le Reinforcement Learning (RL)

L'**apprentissage par renforcement** : un agent apprend en interagissant avec un environnement.
- **Action** : choisir le prochain client à visiter
- **Récompense** : $-c_{ij}$ (distance parcourue, négative car on minimise)
- **Politique** $\pi_\theta$ : la stratégie de décision (le réseau de neurones)
- **Objectif** : maximiser la récompense cumulée = minimiser la distance totale

**Pourquoi le RL plutôt que l'apprentissage supervisé ?** En supervisé, il faudrait des milliers de solutions optimales labellisées — qui coûtent chacune des minutes à calculer avec l'oracle. Le RL permet d'apprendre sans labels : le réseau s'améliore en comparant ses propres solutions.

### 6.3 L'Architecture Transformer

Inspiré de *"Attention, Learn to Solve Routing Problems!"* (Kool et al., 2019, ICLR).

**Le mécanisme d'attention** : Calculer une pertinence pondérée entre une requête $q$ (le contexte courant) et chaque candidat $k_i$ :
$$\text{score}_i = \frac{q \cdot k_i}{\sqrt{d}} \xrightarrow{\text{softmax}} \text{probabilités sur les clients}$$

**Architecture** :
```
ENCODEUR (1 passe par instance, mise en cache) :
  Entrée : 6 features par client [x, y, demande, a_i, b_i, service_time]
  → 3 couches de Multi-Head Attention (dim=128, 8 têtes)
  → Sortie : représentation riche h_i ∈ R^128 pour chaque client

DÉCODEUR (N étapes, une par client à visiter) :
  Requête    = concat(h_actuel, h_moyen, contexte_dynamique)
  Clés       = concat(h_i, features_dynamiques_i)
  Score_i    = tanh(q·k_i / √128) × 10  [borné dans [-10, 10]]
  Masque     = score = -∞ si client déjà visité ou infaisable (C3/C4)
  Action     = argmax(softmax(scores))
```

**Paramètres du modèle AM** :

| Paramètre | Valeur | Impact |
|---|---|---|
| Dimension encodeur $d$ | 128 | Plus grand → plus expressif mais plus lourd |
| Nb couches Transformer | 3 | Profondeur de la représentation |
| Nb têtes d'attention | 8 | Capture plusieurs types de relations |
| Borne logits $C$ | 10 | Stabilité numérique du softmax |
| Taille batch entraînement | 128 | Variance du gradient |
| Nb épisodes entraînement | 100 000 | Qualité de la politique apprise |

**Entraînement REINFORCE** (Williams, 1992) :
```
Pour chaque instance :
  1. Dérouler la politique → obtenir une tournée, calculer le coût G
  2. Avantage : A = G - baseline (moyenne mobile des coûts passés)
  3. Gradient : ∇L = -A × Σ log π(action_t | état_t)
  4. Mettre à jour les poids → les bonnes décisions sont renforcées
```

### 6.4 Résultats et Limitation

| n | Gap AM vs optimal | Temps inférence | Validité |
|---|---|---|---|
| 10 | ~14.9 % | 32 ms | 100% |
| 20 | ~22.5 % | 80 ms | 70% |
| 50 | ~37.8 % | 207 ms | 45% |
| 100 | ~41.1 % | 381 ms | 35% |
| 200 | ~39.9 % | 821 ms | 25% |

**Distribution Shift** : modèle entraîné sur $n=10$, testé sur $n=200$ → 75% d'infaisabilité.

**Solution CTR (Cluster-Then-Route)** :
1. K-means : découper les 200 clients en 20 clusters de ~10 clients
2. AM résout chaque cluster (reste dans son domaine d'entraînement $n \approx 10$)
3. Fusionner → validité 100%, gap acceptable

---

## PARTIE 7 — La Comparaison Finale (Benchmark)

### 7.1 Le protocole

**150 instances standard** : 30 seeds × 5 tailles ($n \in \{10, 20, 50, 100, 200\}$), résolues par les 4 méthodes + l'oracle OR-Tools.

L'**oracle** (OR-Tools CP-SAT de Google) est un solver exact — il trouve l'optimal garanti. Il sert de référence absolue pour calculer le gap.

### 7.2 Tableau comparatif

**Gap moyen vs oracle** :

| n | NNH+2opt | Recuit SA | Génétique | Attention Model |
|---|---|---|---|---|
| 10 | 31% | **0.3%** | 16% | 14.9% |
| 20 | 49% | **19.5%** | 36.6% | 22.5% |
| 50 | 63% | **36.8%** | 90.8% | 37.8% |
| 100 | 70% | 78% | 123% | **41.1%** |
| 200 | 65% | 156% | 192% | **39.9%** |

**Temps d'exécution** :

| n | NNH+2opt | Recuit SA | Génétique | Attention Model |
|---|---|---|---|---|
| 10 | **0.01 ms** | 1 s | 0.35 s | 32 ms |
| 50 | **0.2 ms** | 2.9 s | 2.9 s | 207 ms |
| 200 | **2 ms** | 10.5 s | 20.7 s | 821 ms |

### 7.3 Présentation des résultats statistiques et des graphiques

Les résultats sont présentés avec les graphiques suivants :

**G1 — Temps d'exécution médian vs n** (échelle log-log)
Montre la croissance du temps d'exécution pour chaque méthode en fonction de la taille des instances. Échelle logarithmique sur les deux axes pour visualiser les ordres de grandeur (NNH : ms, SA : secondes).

**G2 — Coût moyen ± IC 95%**
Barres d'erreur représentant l'intervalle de confiance à 95% (loi t, 29 degrés de liberté). Permet de voir si les différences entre méthodes sont statistiquement significatives ou dues à la variance des instances.

**G3 — Coût et temps sur double axe Y**
Graphique combiné permettant de visualiser simultanément qualité et rapidité pour chaque méthode. Utile pour identifier les méthodes sur le front de Pareto (meilleur compromis qualité-temps).

**G4 — Gap vs oracle par méthode et taille**
La métrique centrale : $(c_\text{algo} - c^*) / c^* \times 100\%$. Montre clairement que SA domine pour n ≤ 50, que AM domine pour n ≥ 100, et que GA diverge.

**G5 — Coefficient de variation (σ/μ)**
Mesure la variabilité des résultats. Un CV élevé signifie que la méthode est instable (bons résultats sur certaines instances, mauvais sur d'autres). GA a un CV très élevé pour n ≥ 50, révélant son instabilité.

### 7.4 Recommandations

| Contexte | Méthode | Raison |
|---|---|---|
| Temps réel, volume massif | NNH + 2-opt | < 2 ms, 100% valide |
| Qualité prioritaire, n ≤ 50 | Recuit Simulé | Quasi-optimal |
| Grandes instances (n ≥ 100) | AM + CTR | Croissance linéaire, bon gap |
| Prototypage | Algo Génétique | Simple à comprendre |

**Le trade-off fondamental** :
```
Rapidité ↑ ←————————————————————→ Qualité ↑
NNH         AM          Recuit SA    Oracle
(ms)       (< 1s)        (10s)     (minutes)
```

---

## PARTIE 8 — Références Scientifiques

Les travaux fondateurs et articles clés du projet :

### Problème de base

| Référence | Contribution |
|---|---|
| **Dantzig & Ramser (1959)** — "The Truck Dispatching Problem", *Management Science* | Premier article sur le VRP ; formulation originale du problème de tournées |
| **Solomon (1987)** — "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints", *Operations Research* | Définit les instances benchmarks VRPTW (C1, C2, R1, R2, RC1, RC2) encore utilisées aujourd'hui |
| **Savelsbergh & Sol (1995)** — "The General Pickup and Delivery Problem", *Transportation Science* | Preuve de NP-complétude du VRPTW |

### Complexité

| Référence | Contribution |
|---|---|
| **Cook (1971)** — "The complexity of theorem-proving procedures", *STOC* | Preuve que SAT est NP-complet (premier problème NP-complet démontré) |
| **Karp (1972)** — "Reducibility among combinatorial problems", *Complexity of Computer Computations* | 21 problèmes NP-complets dont TSP et Circuit Hamiltonien |

### Métaheuristiques

| Référence | Contribution |
|---|---|
| **Kirkpatrick, Gelatt & Vecchi (1983)** — "Optimization by Simulated Annealing", *Science* | Fonde le Recuit Simulé ; démontre la convergence théorique |
| **Holland (1975)** — *Adaptation in Natural and Artificial Systems*, MIT Press | Fonde les algorithmes génétiques ; schémas, fitness, sélection |

### Deep Learning pour l'optimisation combinatoire

| Référence | Contribution |
|---|---|
| **Vinyals, Fortunato & Jaitly (2015)** — "Pointer Networks", *NeurIPS* | Premier réseau de neurones pour résoudre le TSP ; mécanisme d'attention comme "pointeur" |
| **Williams (1992)** — "Simple Statistical Gradient-Following Algorithms for Connectionist RL", *Machine Learning* | Algorithme REINFORCE ; fondement théorique de l'entraînement de nos politiques |
| **Kool, Van Hoof & Welling (2019)** — "Attention, Learn to Solve Routing Problems!", *ICLR* | **Modèle utilisé en Phase 4** ; Transformer encodeur + décodeur contextuel pour VRPTW |
| **Kwon et al. (2020)** — "POMO: Policy Optimization with Multiple Optima", *NeurIPS* | État de l'art actuel ; départs multiples → < 1% gap sur n=100 |
| **Li et al. (2021)** — "Learning to Delegate for Large-Scale Vehicle Routing", *NeurIPS* | L2D : décomposition apprise pour grandes instances |

### Solvers exacts

| Référence | Contribution |
|---|---|
| **Régin (1994)** — "A filtering algorithm for constraints of difference in CSPs", *AAAI* | Base théorique des contraintes de différence dans CP-SAT |
| **OR-Tools (Google, 2010–présent)** — open-source | **Notre oracle** ; CP-SAT solver de référence industrielle |

---

## PARTIE 9 — 25 Questions de Jury avec Réponses Développées

---

### BLOC A — Compréhension du problème

**Q1 : Qu'est-ce que le VRPTW et en quoi est-il différent du TSP ?**

Le **TSP** cherche la tournée la plus courte visitant une fois chaque ville avec un seul voyageur, sans contrainte de capacité ni de temps. Le **VRPTW** étend le TSP en ajoutant : (1) plusieurs véhicules avec capacité limitée, (2) des fenêtres temporelles $[a_i, b_i]$ où chaque client doit être visité, (3) un horizon journalier $H$. Le VRPTW est donc strictement plus difficile : tout TSP est un VRPTW dégénéré (1 véhicule, $Q=\infty$, fenêtres $[0, +\infty]$).

---

**Q2 : Pourquoi avez-vous choisi de minimiser la distance totale plutôt que le nombre de véhicules ou le temps total ?**

La distance totale est le proxy standard du coût opérationnel (carburant ∝ distance). Dans notre formulation, le nombre de véhicules est fixé a priori par $K = \max(3, \lceil n/10 \rceil)$, donc ce n'est pas une variable à optimiser. Une extension naturelle serait un objectif bicritère (distance + nombre de véhicules actifs), mais cela complexifie la formulation.

---

**Q3 : Expliquez la contrainte C4. Que se passe-t-il si on arrive avant l'ouverture ? Après la fermeture ?**

Si le véhicule arrive **avant** l'ouverture ($\tau_i < a_i$) : il **attend** sur place, l'heure effective de service devient $\max(\tau_i, a_i) = a_i$. Cette attente consomme du temps horizon.

Si le véhicule arrive **après** la fermeture ($\tau_i > b_i$) : la **contrainte est violée**, la solution est **infaisable** (rejet strict dans NNH+2opt, pénalité dans Recuit/Génétique).

---

**Q4 : Pourquoi utilise-t-on des seeds (graines aléatoires) dans le générateur d'instances ?**

Pour la **reproductibilité**. Avec la même seed, le générateur produit exactement la même instance. Cela permet de : (1) comparer équitablement les 4 méthodes sur les *mêmes* problèmes, (2) reproduire les expériences, (3) déboguer. Sans seed fixe, chaque exécution produirait des instances différentes, rendant la comparaison impossible.

---

**Q5 : Pourquoi le nombre de véhicules est-il calculé comme $K = \max(3, \lceil n/10 \rceil)$ ?**

C'est une formule empiriquement calibrée pour garantir la faisabilité. Le $\lceil n/10 \rceil$ donne ~1 véhicule pour 10 clients (charge moyenne raisonnable). Le $\max(3, ...)$ assure un minimum de 3 véhicules même pour $n=10$, car avec moins les fenêtres temporelles rendent souvent les instances infaisables. Le facteur $1.4$ sur la capacité ajoute une marge de 40% sur la demande moyenne.

---

**Q6 : Comment passez-vous d'un graphe incomplet (réseau routier réel) à un graphe complet utilisé dans votre modèle ?**

En calculant la **matrice de distances Euclidienne** $c_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}$ entre chaque paire de points. Cela suppose que l'on peut se déplacer en ligne droite entre n'importe quels deux points — approximation valable pour des instances synthétiques uniformément distribuées. En contexte réel, on remplacerait la distance Euclidienne par le résultat d'un calcul d'itinéraire (OSRM, Google Maps), mais la structure du modèle resterait identique.

---

### BLOC B — NNH + 2-opt

**Q7 : Quelle est la complexité de votre algorithme NNH + 2-opt ? Justifiez.**

- **NNH** : Pour chaque client ajouté, on compare avec tous les restants → $O(n^2)$ en pratique.
- **2-opt** : Pour une route de longueur $m$, on teste $O(m^2)$ paires → $O(n^3)$ pire cas global, $O(n^2)$ en pratique car convergence rapide avec TW.
- **Total** : dominé par $O(n^2)$ en pratique.

---

**Q8 : Pourquoi le gain apporté par le 2-opt est-il aussi faible (< 1%) dans votre contexte VRPTW ?**

Les fenêtres temporelles **contraignent fortement** les inversions 2-opt. Une inversion qui raccourcirait la distance change l'ordre de visite → certains clients seraient visités hors créneau → la contrainte C4 est violée → l'inversion est **rejetée**. Plus les fenêtres sont étroites (60–90 min dans notre cas), plus le 2-opt est limité.

---

**Q9 : Pourquoi NNH utilise-t-il parfois plus de véhicules que nécessaire ?**

NNH est **glouton et sans retour arrière**. Si à une étape il assigne un client à un mauvais véhicule (parce que c'était le plus proche à cet instant), ce choix ne peut plus être corrigé. Le véhicule se retrouve bloqué plus tôt que nécessaire, obligeant l'ouverture d'un nouveau véhicule. Une solution à moins de véhicules existe mais NNH ne peut pas la trouver car elle nécessiterait de l'optimisation inter-routes.

---

### BLOC C — Métaheuristiques

**Q10 : Quelle est la différence fondamentale entre le Recuit Simulé et NNH ?**

**NNH** est une **construction unidirectionnelle** : il construit la solution de zéro, fait des choix définitifs, n'explore qu'une seule solution.

**Recuit Simulé** est une **recherche locale itérative** : il part d'une solution existante, puis la perturbe légèrement. La différence clé : il peut **accepter des solutions pires** temporairement pour s'échapper des optima locaux — ce que NNH ne fait jamais.

---

**Q11 : Expliquez le mécanisme de refroidissement. Pourquoi ne pas fixer une température constante ?**

Le refroidissement $T \leftarrow T \times \alpha$ crée une **phase d'exploration** (haute T) puis une **phase d'exploitation** (basse T). Avec une température constante élevée → l'algorithme n'exploite jamais, ne converge pas. Avec une température constante basse → il reste bloqué dans le premier optimum local. Le refroidissement progressif imite le recuit physique : explorer librement d'abord, puis figer progressivement.

---

**Q12 : Quel est l'impact du paramètre α (cooling rate) sur la qualité de la solution du Recuit Simulé ?**

- **α trop faible** (ex. 0.9) : Refroidissement brutal. L'algorithme passe trop peu de temps à haute température → exploration insuffisante → converge vers un optimum local de mauvaise qualité.
- **α trop élevé** (ex. 0.9999) : Refroidissement très lent → l'algorithme explore très longtemps → meilleure qualité mais temps de calcul prohibitif.
- **α = 0.995** (notre choix) : Compromis empirique — refroidit en $\approx 1000$ paliers, permet exploration puis convergence dans le temps imparti.

---

**Q13 : Pourquoi l'Algo Génétique diverge-t-il pour n ≥ 50 ?**

1. **Convergence prématurée** : La population converge rapidement vers un optimum local (tous les individus se ressemblent) avant d'avoir suffisamment exploré.
2. **Explosion de l'espace** : Pour $n=50$, il y a $50!$ solutions possibles. Une population de 40 individus n'en couvre qu'une infime fraction. Pour $n=10$, l'espace est suffisamment petit pour que la population soit représentative.

---

**Q14 : Qu'est-ce que les "soft constraints" et pourquoi les métaheuristiques en ont-elles besoin ?**

**Hard constraints** : Toute solution les violant est rejetée (infaisable). C'est ce que fait NNH.

**Soft constraints** : Les violations sont autorisées mais pénalisées dans le coût :
$$\text{coût} = \text{distance} + \lambda_C \times \text{violations\_capacité} + \lambda_T \times \text{violations\_temps}$$

Les métaheuristiques en ont besoin car elles explorent l'espace en perturbant des solutions. Si toute solution légèrement infaisable est rejetée, l'algorithme ne peut pas traverser les zones infaisables pour atteindre de meilleures zones de l'autre côté.

---

### BLOC D — Deep Learning

**Q15 : Qu'est-ce que le mécanisme d'attention et pourquoi est-il adapté au VRPTW ?**

Le mécanisme d'attention calcule une pertinence pondérée entre une requête $q$ (contexte courant) et chaque client $k_i$ :
$$\text{score}_i = \frac{q \cdot k_i}{\sqrt{d}} \xrightarrow{\text{softmax}} \text{probabilités}$$

Adapté au VRPTW car : (1) la décision du prochain client dépend du **contexte global** (tous les clients restants) → l'attention capture ces dépendances ; (2) le masquage des clients infaisables est naturel (score = $-\infty$) ; (3) l'encodeur Transformer capture des relations entre clients.

---

**Q16 : Qu'est-ce que le "distribution shift" et comment l'avez-vous résolu ?**

**Distribution shift** = le modèle est évalué sur une distribution différente de celle d'entraînement. Ici : entraîné sur $n=10$, testé sur $n=200$. Le réseau n'a jamais vu 200 points → solutions infaisables (validité 25% pour $n=200$).

**Solution CTR** : Découper en clusters de ~10 clients, résoudre chaque cluster avec l'AM (qui reste dans son domaine), assembler les routes → validité 100%.

---

**Q17 : Comparez le temps d'inférence de l'AM avec les autres méthodes. Pourquoi dit-on que l'AM "passe à l'échelle" mieux ?**

L'AM a une complexité d'inférence **O(n²)** (n étapes × n scores), mais surtout il **n'itère pas** : une seule passe par instance, contrairement au Recuit qui fait des milliers d'itérations. Pour $n=1000$, le Recuit prendrait des heures, l'AM resterait en quelques secondes.

---

**Q18 : Qu'est-ce que l'algorithme REINFORCE et pourquoi est-il utilisé ici ?**

REINFORCE est un algorithme de **gradient de politique**. Il maximise l'espérance du retour en renforçant les actions ayant mené à de bons résultats :
$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[(G - b) \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

On l'utilise car le problème est **non-différentiable** (on ne peut pas dériver le coût total par rapport aux décisions discrètes de routing) et on n'a pas besoin de solutions optimales comme labels d'entraînement.

---

### BLOC E — Comparaison et Analyse

**Q19 : Si vous deviez déployer une solution en production, quelle méthode choisiriez-vous ?**

- **Petit transporteur** (n ≤ 30, 1 optimisation/jour) : **Recuit Simulé** — qualité quasi-optimale, temps acceptable.
- **Opérateur urbain** (n ≈ 50–200, centaines d'optimisations/heure) : **AM + CTR** — inférence < 1s, scalable.
- **Planning critique** (qualité maximale, n ≤ 50, temps ≤ 30 min) : **Oracle OR-Tools** — si le temps est acceptable, prendre l'optimal.

---

**Q20 : Le Recuit Simulé obtient un gap de 156% pour n=200. Comment expliquez-vous cette dégradation par rapport à NNH (65%) ?**

Le Recuit utilise des **soft constraints**. Pour $n=200$ avec un nombre d'itérations fixe, l'algorithme manque de temps pour réduire les violations tout en améliorant le coût. Il converge vers une solution avec violations qui, mesurée en coût réel (sans pénalités), est très dégradée. NNH construit une solution **toujours valide** par construction — sa garantie de validité prime sur la qualité théoriquement meilleure mais infaisable du SA.

---

**Q21 : Votre benchmark utilise 30 seeds par taille. Pourquoi pas 5 ? Pourquoi pas 1000 ?**

**Avec 5 seeds** : trop peu, variance élevée, intervalles de confiance très larges.

**Avec 30 seeds** : compromis standard. Permet des IC à 95% raisonnables. $n=30$ est le seuil à partir duquel on peut approcher la loi normale (théorème central limite).

**Avec 1000 seeds** : statistiquement parfait mais 4 méthodes × 5 tailles × 1000 seeds = 20 000 résolutions → pour le Recuit à 10s/instance : **55 heures de calcul**. Non viable.

---

**Q22 : Quelles améliorations apporteriez-vous à l'Attention Model si vous refaisiez le projet ?**

1. **Curriculum learning** : Entraîner progressivement n=10 → n=20 → n=50.
2. **Data augmentation** : Mélanger toutes les tailles pendant l'entraînement.
3. **POMO** (Policy Optimization with Multiple Optima) : Plusieurs départs simultanés, prendre la meilleure solution.
4. **Instances réelles** (CVRPLIB) : Améliorer la généralisation.
5. **Beam Search** : Explorer les k meilleures actions à chaque étape pour améliorer la qualité.

---

**Q23 : Qu'est-ce qu'un problème NP-complet vs NP-difficile ?**

**NP-complet** : Problème qui est (a) dans NP (vérifiable en temps polynomial) ET (b) NP-difficile (tout problème NP s'y réduit). C'est le "sommet de NP".

**NP-difficile** : Au moins aussi difficile qu'un problème NP-complet, mais pas nécessairement dans NP lui-même. Le VRPTW est NP-difficile : au moins aussi dur que le TSP (NP-complet), et sa version décisionnelle est dans NP, donc techniquement NP-complet.

---

**Q24 : Pourquoi utilise-t-on OR-Tools comme oracle plutôt qu'un autre solver ?**

OR-Tools (Google) est un **solver CP-SAT** open-source de référence industrielle qui : (1) garantit l'optimalité, (2) gère nativement les contraintes temporelles du VRPTW, (3) est bien documenté et reproductible, (4) permet une comparaison équitable avec d'autres équipes utilisant le même oracle.

---

**Q25 : Quelles métriques statistiques utilisez-vous ? Pourquoi la médiane plutôt que la moyenne ?**

Métriques utilisées : moyenne, médiane, écart-type, CV (σ/μ), IC à 95% (loi t de Student), boxplots.

**Médiane vs Moyenne** : La distribution des coûts peut être asymétrique (quelques instances très difficiles tirent la moyenne vers le haut). La médiane est **robuste aux outliers** : une instance exceptionnellement mauvaise ne pollue pas la tendance centrale.

---

**Q26 : Vos résultats seraient-ils différents avec des données réelles ?**

Très probablement. Nos instances sont **synthétiques et uniformément distribuées**. Les instances réelles présentent souvent des **clusters spatiaux** (zones résidentielles, industrielles) et des **corrélations temporelles** (livraisons groupées). Les algorithmes RL/AM tendent à mieux généraliser sur des instances structurées. De plus, les contraintes réelles (sens interdits, bouchons, fermetures de rues) rendraient la matrice de distances non-euclidienne.

---

**Q27 : Quel est l'apport de ce projet par rapport à l'état de l'art ?**

Ce projet est un **travail pédagogique de comparaison** plutôt qu'une contribution scientifique originale. Son apport :
1. **Comparaison équitable** de 4 paradigmes algorithmiques sur le même générateur d'instances.
2. **Validation empirique** que l'AM rivalise avec le Recuit pour n ≥ 100 malgré l'absence d'entraînement sur ces tailles.
3. **Quantification du distribution shift** : 75% d'infaisabilité pour n=200 avec modèle entraîné sur n=10.

L'état de l'art actuel (2024) utilise POMO (Kwon et al., 2020) ou L2D (Li et al., 2021) qui obtiennent < 5% de gap sur des instances jusqu'à n=100 avec entraînement multi-taille.

---

**Q28 : Comment les paramètres du GA (population, mutation) affectent-ils la convergence ?**

La **taille de population** contrôle la diversité initiale : une population trop petite (P=10) converge prématurément car les individus se ressemblent dès les premières générations ; trop grande (P=200), le calcul du fitness devient le goulot d'étranglement.

Le **taux de mutation** est le contrôle de la diversité continue : trop faible (pm=0.01) → convergence prématurée garantie sur grandes instances ; trop élevé (pm=0.8) → la recherche devient aléatoire, on perd les bonnes structures trouvées par croisement.

Pour n≥50, même avec des paramètres bien calibrés, le GA diverge car l'espace de solutions est trop vaste pour être couvert par une population de 40–100 individus.

---

## RÉSUMÉ EXÉCUTIF

**Problème** : Minimiser la distance totale de tournées pour livrer n clients depuis un dépôt, en respectant la capacité des véhicules et les fenêtres temporelles d'accessibilité. Problème NP-difficile.

**4 méthodes développées** :

| Méthode | Paradigme | Temps | Qualité | Scalabilité |
|---------|-----------|-------|---------|------------|
| **NNH + 2-opt** | Heuristique constructive gloutonne | O(n²) | ~30–70% au-dessus optimal | ✅ Excellent |
| **Recuit Simulé** | Métaheuristique probabiliste | O(n × iter) | Quasi-optimal n ≤ 50 | ⚠️ Moyen |
| **Algo Génétique** | Évolution de population | O(pop × gen × n) | Diverge pour n ≥ 50 | ❌ Faible |
| **Attention Model** | Deep Learning (RL) | O(n) inférence | Bon compromis n ≥ 100 | ✅ Excellent |

**Message principal** : Il n'existe pas de méthode universellement meilleure. Le choix dépend du contexte opérationnel (budget temps, taille des instances, contraintes qualité).
