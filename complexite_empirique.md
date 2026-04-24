# Complexité empirique — Comment lire le graphique log-log

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
