# Complexité empirique — Comment lire le graphique log-log

---

## 0. Vulgarisation — l'idée en langage courant

### Le problème concret

Tu as un algorithme. Tu le fais tourner sur des problèmes de taille 10, 20, 50, 100, 200 clients. Tu mesures à chaque fois combien de temps il met. Tu obtiens un tableau comme ça :

| $n$ (clients) | Temps mesuré |
|---|---|
| 10 | 0.0001 s |
| 20 | 0.0004 s |
| 50 | 0.0025 s |
| 100 | 0.010 s |
| 200 | 0.040 s |

**Question** : Si demain tu dois résoudre un problème avec 1000 clients, combien de temps ça va prendre ?

Pour répondre, il faut trouver la **loi** qui relie le temps à la taille. C'est ça, la complexité empirique.

---

### L'idée intuitive : chercher la loi cachée

Regarde les données ci-dessus. Quand $n$ double (de 100 à 200), le temps est multiplié par 4 (de 0.010 à 0.040). Quand $n$ est multiplié par 5 (de 10 à 50), le temps est multiplié par 25. Dans les deux cas : **le temps = $n$ au carré**.

C'est une **loi puissance** : $T = a \times n^p$. Ici $p = 2$.

Le graphique log-log permet de trouver $p$ **visuellement et mathématiquement**, même quand ce n'est pas aussi évident.

---

### Pourquoi "log-log" ?

Imagine que tu veux peser des objets dont la masse va de 1 gramme (une fourmi) à 1 000 000 grammes (une voiture). Sur une balance classique graduée de 1 en 1, la fourmi est invisible. Une **échelle logarithmique** espace les ordres de grandeur : 1g, 10g, 100g, 1000g, … sont équidistants. Chaque graduation = ×10.

C'est pareil ici : nos temps vont de 0.0001 s à 0.04 s. Sur un axe normal, les petites valeurs s'écrasent. Sur un axe log, tout est lisible.

Mais il y a un bonus mathématique : **en log-log, une loi puissance devient une droite**. Et la pente de cette droite, c'est directement l'exposant $p$ — c'est-à-dire la complexité.

---

### Le résultat en clair

Sur le graphique, le code calcule la pente de la droite et affiche :

$$T \propto n^{2.03} \quad (R^2 = 0.99)$$

- **$n^{2.03}$** : le temps croît comme $n^2$ → c'est un algorithme $O(n^2)$, confirmé par les données réelles
- **$R^2 = 0.99$** : la droite colle à 99% aux points mesurés → le modèle est fiable

En pratique : si $n$ double, le temps est multiplié par $2^{2.03} \approx 4$. Si $n$ est multiplié par 10, le temps est multiplié par $10^{2.03} \approx 107$.

---

## 1. Le problème de départ

On a mesuré le **temps d'exécution moyen** de l'algorithme pour différentes tailles d'instances $n \in \{10, 20, 50, 100, 200\}$.

On veut répondre à : **"Comment le temps croît-il quand $n$ augmente ?"**

La théorie dit que NNH est $O(n^2)$. Le graphique log-log permet de **vérifier cette affirmation sur les données réelles**, sans supposer quoi que ce soit à l'avance.

---

## 2. Le modèle : loi puissance

On suppose que le temps suit une **loi puissance** :

$$T(n) = a \cdot n^p$$

où :
- $a > 0$ est une constante (dépend de la machine, du langage, etc.)
- $p$ est l'**exposant de complexité** — ce qu'on cherche à estimer

Exemples :
| Si $p =$ | Complexité | Signification |
|---|---|---|
| $1$ | $O(n)$ | Linéaire |
| $2$ | $O(n^2)$ | Quadratique |
| $3$ | $O(n^3)$ | Cubique |
| $0.5$ | $O(\sqrt{n})$ | Sous-linéaire |

---

## 3. L'astuce log-log : linéariser la loi puissance

On applique $\log_{10}$ des deux côtés de $T(n) = a \cdot n^p$ :

$$\log_{10} T = \log_{10}(a \cdot n^p)$$

$$\log_{10} T = \log_{10} a + p \cdot \log_{10} n$$

En posant $Y = \log_{10} T$ et $X = \log_{10} n$, on obtient :

$$\boxed{Y = \underbrace{p}_{\text{pente}} \cdot X + \underbrace{\log_{10} a}_{\text{ordonnée à l'origine}}}$$

C'est l'équation d'une **droite** dans l'espace $(\log n,\ \log T)$.

**Conclusion clé :** dans un repère log-log, toute loi puissance $T = a \cdot n^p$ devient une droite de pente $p$.

---

## 4. Ce que fait le code

```python
log_n = np.log10(TAILLES)          # X = log10([10, 20, 50, 100, 200])
log_t = np.log10(means)            # Y = log10([temps moyen pour chaque n])

slope, intercept, r, _, _ = sp_stats.linregress(log_n, log_t)
```

`linregress` ajuste la droite $Y = \text{slope} \cdot X + \text{intercept}$ par **moindres carrés** (minimise la somme des carrés des écarts entre les points et la droite).

- `slope` $= \hat{p}$ → **exposant empirique** estimé
- `intercept` $= \log_{10} \hat{a}$ → constante du modèle
- `r²` → **coefficient de détermination** : fraction de la variance expliquée par le modèle ($r^2 = 1$ = ajustement parfait)

La courbe retracée dans l'espace original :

```python
t_fit = 10**(intercept + slope * np.log10(n_fit))
```

est équivalente à :

$$\hat{T}(n) = 10^{\text{intercept}} \cdot n^{\text{slope}} = \hat{a} \cdot n^{\hat{p}}$$

---

## 5. Lecture du résultat sur le graphique

Le label affiché est :

$$T \propto n^{2.03} \quad (R^2 = 0.99)$$

Ce qui signifie :

1. **$\hat{p} = 2.03 \approx 2$** → le temps croît comme $n^2$ → confirme empiriquement $O(n^2)$
2. **$R^2 = 0.99$** → le modèle loi puissance explique 99 % de la variance des temps mesurés → excellent ajustement, pas de bruit dominant

---

## 6. Pourquoi log-log et pas un axe linéaire ?

Sur un axe **linéaire**, les courbes $n^{1.8}$, $n^2$ et $n^{2.2}$ se ressemblent beaucoup visuellement — difficile de les distinguer.

Sur un axe **log-log**, elles deviennent trois droites de pentes différentes ($1.8$, $2.0$, $2.2$) — parfaitement séparables à l'œil, et la pente se lit directement sur le graphique.

```
log T
  │                          ╱  pente 2.2  (O(n^2.2))
  │                        ╱
  │                     ╱──   pente 2.0  (O(n^2))
  │                  ╱─
  │              ╱──         pente 1.8  (O(n^1.8))
  │          ╱──
  └─────────────────────── log n
```

---

## 7. Résumé en une phrase

> Le graphique log-log transforme une loi puissance en droite : la **pente de cette droite est directement l'exposant de complexité** $p$, estimé sur les temps d'exécution réels.
