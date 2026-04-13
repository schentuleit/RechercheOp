"""
verification_dl.py  —  Phase A : vérification de cohérence
===========================================================
Adapté à l'architecture RÉELLE du projet :
    - Encodeur Transformer (3 couches, 8 têtes, d_h=128)
    - Décodeur à attention (Kool 2019)
    - Environnement VRPTW multi-véhicules (VRPTWEnv)
    - Préprocessing via preprocess.py (node_features shape (N, 6))

NE PAS MODIFIER les fonctions d'inférence sans synchroniser avec train.py.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch

# ── Imports locaux (même dossier que model.py / env.py) ──────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from preprocess import generate_instance, preprocess
from model      import VRPTWModel
from env        import VRPTWEnv


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def charger_modele(chemin: str, device: torch.device) -> tuple:
    """
    Charge un checkpoint .pt sauvegardé par train.py.

    Retourne
    --------
    model   : VRPTWModel en mode eval()
    history : dict de l'historique d'entraînement
    config  : dict des hyperparamètres
    """
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Checkpoint introuvable : {chemin}")

    ckpt = torch.load(chemin, map_location=device, weights_only=False)

    model = VRPTWModel(
        input_dim=6,
        d_h=128,
        n_heads=8,
        n_layers=3,
        d_ff=512,
        C=10.0,
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    history = ckpt.get('history', {})
    config  = ckpt.get('config', {})
    epoch   = ckpt.get('epoch', '?')

    print(f"  [OK] Chargé : {os.path.basename(chemin)}  (epoch {epoch})")
    if history.get('dist_greedy'):
        print(f"       Meilleure greedy entraînement : {min(history['dist_greedy']):.1f}")

    return model, history, config


def charger_tous_les_modeles(chemins: dict, device: torch.device) -> dict:
    """
    chemins : {n: chemin_vers_checkpoint}
    Retourne : {n: (model, history, config)}
    """
    modeles = {}
    for n, chemin in chemins.items():
        try:
            modeles[n] = charger_modele(chemin, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    return modeles


# ─────────────────────────────────────────────────────────────────────────────
# 2. INFÉRENCE  —  recopie exacte de run_episode(greedy=True) depuis train.py
# ─────────────────────────────────────────────────────────────────────────────

def _tous_bloques(env: VRPTWEnv) -> bool:
    """Vérifie si aucun véhicule ne peut encore se déplacer."""
    t_save = env.t_dispo.copy()
    for k in range(env.K):
        env.t_dispo    = np.full(env.K, np.inf)
        env.t_dispo[k] = t_save[k]
        masque         = env.compute_mask()
        env.t_dispo    = t_save.copy()
        if masque.any():
            return False
    return True


def inferer_tournee(
    model: VRPTWModel,
    instance: dict,
    device: torch.device,
    greedy: bool = True,
) -> dict:
    """
    Fait tourner le modèle sur une instance — mode greedy par défaut.

    Reproduit run_episode(greedy=True) de train.py exactement,
    avec en plus la collecte du chemin par véhicule.

    Retourne
    --------
    dict :
        tournees        : list[list[int]]  — chemin de chaque véhicule
        cout_total      : float
        temps_inference : float  (secondes)
        violations      : list[str]
        env             : VRPTWEnv  (état final, pour debug)
    """
    node_features = preprocess(instance)
    env = VRPTWEnv(instance, node_features)
    obs = env.reset()

    nf_tensor = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        H, h_bar = model.encode(nf_tensor)

    # Suivi du chemin par véhicule
    chemins      = {k: [0] for k in range(instance['n_vehicles'])}
    n_deblocages = 0  # skips masque vide — indicateur du vrai problème
    max_steps    = instance['n'] * instance['n_vehicles'] * 4 + 50

    debut = time.perf_counter()

    for _ in range(max_steps):
        if env.is_done():
            break

        ctx     = env.get_context_features()
        mask_np = obs['mask']

        # Blocage d'un véhicule — on le saute et on passe au suivant
        if not mask_np.any():
            k_actif = env.get_active_vehicle()
            env.t_dispo[k_actif] = env.horizon + 1.0
            n_deblocages += 1
            if _tous_bloques(env):
                env.force_done()
                break
            obs = env._get_obs()
            continue

        k_actif = ctx['k']

        last_node_t = torch.tensor([ctx['last_node']], dtype=torch.long).to(device)
        capa_t      = torch.tensor([ctx['capa_norm']], dtype=torch.float32).to(device)
        t_t         = torch.tensor([ctx['t_norm']],    dtype=torch.float32).to(device)
        mask_t      = torch.tensor(~mask_np, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            lp_t      = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
            action, _ = model.select_action(lp_t, greedy=greedy)

        ville = action.item()
        chemins[k_actif].append(ville)

        obs, _ = env.step(ville, log_prob=0.0)

    if not env.is_done():
        env.force_done()

    temps_inference = time.perf_counter() - debut

    # Fermer chaque chemin vers le dépôt
    for k in range(instance['n_vehicles']):
        if len(chemins[k]) > 1 and chemins[k][-1] != 0:
            chemins[k].append(0)

    tournees   = [chemins[k] for k in range(instance['n_vehicles'])]
    violations = verifier_validite(tournees, instance, env)

    return {
        'tournees'       : tournees,
        'cout_total'     : env.total_dist,
        'temps_inference': temps_inference,
        'violations'     : violations,
        'n_deblocages'   : n_deblocages,
        'env'            : env,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. VÉRIFICATEUR DE VALIDITÉ
# ─────────────────────────────────────────────────────────────────────────────

def verifier_validite(tournees: list, instance: dict, env: VRPTWEnv) -> list:
    """
    Contrôle les 4 propriétés fondamentales d'une solution VRPTW valide.

    1. Tous les clients sont visités (exactement une fois)
    2. Chaque chemin commence et finit au dépôt
    3. Les capacités ne sont pas dépassées
    4. Les clients non servis sont signalés (force_done peut en laisser)
    """
    violations = []
    n_clients  = instance['n']
    demands    = instance['demands']
    capacity   = instance['capacity']

    clients_attendus = set(range(1, n_clients + 1))
    clients_visites  = []

    for k, chemin in enumerate(tournees):
        clients_chemin = [v for v in chemin if v != 0]
        clients_visites.extend(clients_chemin)

        if chemin and (chemin[0] != 0 or chemin[-1] != 0):
            violations.append(f"Véhicule {k} : ne commence/finit pas au dépôt")

        charge = sum(demands[v] for v in clients_chemin)
        if charge > capacity + 1e-6:
            violations.append(
                f"Véhicule {k} : charge={charge:.1f} > capacité={capacity:.1f}"
            )

    visites_set = set(clients_visites)
    manquants   = clients_attendus - visites_set
    if manquants:
        violations.append(f"Clients non visités : {sorted(manquants)}")

    if len(clients_visites) != len(visites_set):
        from collections import Counter
        doublons = [c for c, cnt in Counter(clients_visites).items() if cnt > 1]
        violations.append(f"Clients visités plusieurs fois : {doublons}")

    non_servis = int((~env.visited[1:]).sum())
    if non_servis > 0 and not manquants:
        violations.append(f"force_done déclenché : {non_servis} client(s) non servi(s)")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION D'UNE INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

COULEURS_VEHICULES = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']


def afficher_tournee(instance: dict, resultat: dict, titre: str = "", ax=None):
    """
    Affiche la solution multi-véhicules sur le graphe de l'instance.

    Code couleur :
        ■  Dépôt   — rouge carré
        ●  Clients — bleu
        →  Chemin  — une couleur par véhicule, flèches numérotées
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    coords     = instance['coords']
    tw         = instance['time_windows']
    tournees   = resultat['tournees']
    violations = resultat['violations']
    n_vehicles = instance['n_vehicles']

    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for k, chemin in enumerate(tournees):
        if len(chemin) <= 1:
            continue
        couleur = COULEURS_VEHICULES[k % len(COULEURS_VEHICULES)]
        for step, (a, b) in enumerate(zip(chemin[:-1], chemin[1:])):
            xa, ya = coords[a]
            xb, yb = coords[b]
            ax.annotate(
                "",
                xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(
                    arrowstyle="-|>", color=couleur,
                    lw=1.5, mutation_scale=11,
                    connectionstyle="arc3,rad=0.05",
                ),
            )
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            ax.text(mx, my, str(step + 1), fontsize=6.5, color=couleur,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))

    for i, (x, y) in enumerate(coords):
        if i == 0:
            ax.scatter(x, y, s=220, c='#e63946', zorder=6, marker='s')
            ax.text(x, y - 4.5, 'Dépôt', ha='center', fontsize=8,
                    color='#e63946', fontweight='bold')
        else:
            ax.scatter(x, y, s=70, c='#457b9d', zorder=5)
            label = f"{i}\n[{tw[i,0]:.0f},{tw[i,1]:.0f}]"
            ax.text(x + 1.2, y + 1.5, label, fontsize=6, color='#343a40', va='bottom')

    statut         = 'Valide' if not violations else f'{len(violations)} violation(s)'
    couleur_statut = '#2d6a4f' if not violations else '#c1121f'
    ax.set_title(f"{titre}\nCoût : {resultat['cout_total']:.1f}  |  {statut}",
                 fontsize=9.5, color=couleur_statut, pad=8)

    handles = []
    for k in range(n_vehicles):
        if len(tournees[k]) > 1:
            handles.append(mlines.Line2D(
                [], [], color=COULEURS_VEHICULES[k % len(COULEURS_VEHICULES)],
                lw=2, label=f'Véhicule {k}'
            ))
    handles += [
        mpatches.Patch(color='#e63946', label='Dépôt'),
        mpatches.Patch(color='#457b9d', label='Client'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7.5, framealpha=0.85)

    if violations:
        msg = '\n'.join(violations[:4])
        ax.text(0.02, 0.02, msg, transform=ax.transAxes,
                fontsize=6.5, color='#c1121f', va='bottom',
                bbox=dict(boxstyle='round', fc='#fff0f0', ec='#c1121f', alpha=0.9))

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOUCLE DE VÉRIFICATION COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

def run_verification_complete(
    modeles:          dict,
    tailles:          list = None,
    n_instances_test: int  = 5,
    n_vehicles:       int  = 3,
    seed_offset:      int  = 2000,
    device:           torch.device = None,
) -> dict:
    """
    Phase A — vérification sur toutes les tailles entraînées.

    Paramètres
    ----------
    modeles          : {n: (model, history, config)}
    tailles          : list[int] ou None
    n_instances_test : int — instances par taille
    n_vehicles       : int — véhicules par instance
    seed_offset      : int — seeds test = seed_offset + i
    device           : torch.device
    """
    if device is None:
        device = torch.device('cpu')
    if tailles is None:
        tailles = sorted(modeles.keys())

    rapport = {}

    for n in tailles:
        if n not in modeles:
            print(f"\n[SKIP] Pas de modèle pour n={n}")
            continue

        model, history, config = modeles[n]
        print(f"\n{'='*55}")
        print(f" Vérification — n={n} clients  |  {n_vehicles} véhicules")
        print(f"{'='*55}")

        instances, resultats          = [], []
        n_valides, couts, temps_list  = 0, [], []

        for i in range(n_instances_test):
            seed = seed_offset + i
            inst = generate_instance(n, n_vehicles=n_vehicles, seed=seed)
            res  = inferer_tournee(model, inst, device=device, greedy=True)

            instances.append(inst)
            resultats.append(res)
            couts.append(res['cout_total'])
            temps_list.append(res['temps_inference'] * 1000)

            est_valide = len(res['violations']) == 0
            if est_valide:
                n_valides += 1

            statut = 'OK' if est_valide else f"ERREUR ({len(res['violations'])} violation(s))"
            debl   = res.get('n_deblocages', 0)
            print(f"  Inst {i+1} (seed={seed}) | coût={res['cout_total']:7.1f}"
                  f" | t={res['temps_inference']*1000:.1f}ms"
                  f" | déblocages={debl} | {statut}")
            for v in res['violations']:
                print(f"    → {v}")

        bilan = {
            'n_valides'     : n_valides,
            'n_total'       : n_instances_test,
            'taux_validite' : n_valides / n_instances_test,
            'cout_moyen'    : float(np.mean(couts)),
            'cout_std'      : float(np.std(couts)),
            'temps_moyen_ms': float(np.mean(temps_list)),
        }

        print(f"\n  Bilan n={n} : {n_valides}/{n_instances_test} valides"
              f" ({bilan['taux_validite']*100:.0f}%)"
              f" | coût moy={bilan['cout_moyen']:.1f} ± {bilan['cout_std']:.1f}"
              f" | t moy={bilan['temps_moyen_ms']:.1f}ms")

        rapport[n] = {'instances': instances, 'resultats': resultats, 'bilan': bilan}

    return rapport


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS DU RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def afficher_rapport_visuel(rapport: dict, n_colonnes: int = 3, max_par_taille: int = 3):
    """Grille de tournées pour toutes les tailles."""
    tailles     = sorted(rapport.keys())
    n_affichees = sum(min(len(rapport[n]['instances']), max_par_taille) for n in tailles)
    n_lignes    = max(1, int(np.ceil(n_affichees / n_colonnes)))

    fig, axes = plt.subplots(n_lignes, n_colonnes,
                             figsize=(7 * n_colonnes, 7 * n_lignes))
    axes = np.array(axes).flatten()

    idx = 0
    for n in tailles:
        donnees = rapport[n]
        for k in range(min(max_par_taille, len(donnees['instances']))):
            afficher_tournee(
                donnees['instances'][k],
                donnees['resultats'][k],
                titre=f"n={n} | Instance {k+1} (seed={2000+k})",
                ax=axes[idx],
            )
            idx += 1

    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phase A — Vérification des tournées (Attention Model + REINFORCE)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('verification_tournees.png', dpi=150, bbox_inches='tight')
    print("\nFigure sauvegardée : verification_tournees.png")
    plt.show()


def afficher_bilan_global(rapport: dict):
    """Tableau de synthèse + taux de validité + temps d'inférence."""
    tailles = sorted(rapport.keys())
    taux    = [rapport[n]['bilan']['taux_validite'] * 100 for n in tailles]
    temps   = [rapport[n]['bilan']['temps_moyen_ms']      for n in tailles]

    print("\n" + "="*62)
    print(f"{'BILAN GLOBAL — PHASE A  (Attention Model)':^62}")
    print("="*62)
    print(f"{'n':>5} | {'Validité':>9} | {'Coût moy':>10} | {'Coût std':>9} | {'t inférence':>11}")
    print("-"*62)
    for n in tailles:
        b = rapport[n]['bilan']
        print(f"{n:>5} | {b['taux_validite']*100:>8.0f}% | "
              f"{b['cout_moyen']:>10.1f} | {b['cout_std']:>9.1f} | "
              f"{b['temps_moyen_ms']:>9.1f} ms")
    print("="*62)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    couleurs = ['#2d6a4f' if t == 100 else '#e09f3e' if t >= 60 else '#c1121f'
                for t in taux]
    bars = ax1.bar([str(n) for n in tailles], taux, color=couleurs, edgecolor='none')
    ax1.set_ylim(0, 115)
    ax1.axhline(100, color='#2d6a4f', lw=1, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Nombre de clients n', fontsize=11)
    ax1.set_ylabel('Tournées valides (%)', fontsize=11)
    ax1.set_title('Taux de validité par taille', fontsize=12)
    ax1.set_facecolor('#f8f9fa')
    for bar, t in zip(bars, taux):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{t:.0f}%", ha='center', va='bottom', fontsize=10)

    ax2.plot([str(n) for n in tailles], temps, 'o-', color='#534AB7', lw=2, ms=8)
    for i, (n, t) in enumerate(zip(tailles, temps)):
        ax2.text(i, t + max(temps) * 0.01, f"{t:.1f}ms",
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Nombre de clients n', fontsize=11)
    ax2.set_ylabel("Temps d'inférence moyen (ms)", fontsize=11)
    ax2.set_title("Temps d'inférence — argument clé vs recuit simulé", fontsize=12)
    ax2.set_facecolor('#f8f9fa')
    ax2.annotate("Doit rester quasi-constant\n→ argument fort soutenance",
                 xy=(0.97, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=8, color='#534AB7',
                 bbox=dict(boxstyle='round', fc='#EEEDFE', ec='#534AB7', alpha=0.8))

    plt.tight_layout()
    plt.savefig('bilan_phase_a.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée : bilan_phase_a.png")
    plt.show()


def afficher_courbe_apprentissage(history: dict, n: int, ax=None):
    """Courbe sample/greedy depuis l'historique d'entraînement."""
    if not history.get('dist_greedy'):
        print(f"  Pas d'historique pour n={n}")
        return
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history['epoch'], history['dist_sample'],
            color='#adb5bd', lw=1.2, label='Sample (stochastique)', alpha=0.8)
    ax.plot(history['epoch'], history['dist_greedy'],
            color='#534AB7', lw=2, label='Greedy (déterministe)')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Distance totale', fontsize=11)
    ax.set_title(f"Courbe d'apprentissage — n={n}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_facecolor('#f8f9fa')
    return ax


def afficher_toutes_courbes_apprentissage(modeles: dict):
    """Grille des courbes d'apprentissage pour tous les modèles."""
    tailles  = sorted(modeles.keys())
    n_cols   = min(3, len(tailles))
    n_lignes = int(np.ceil(len(tailles) / n_cols))
    fig, axes = plt.subplots(n_lignes, n_cols,
                             figsize=(8 * n_cols, 4 * n_lignes))
    axes = np.array(axes).flatten()

    for i, n in enumerate(tailles):
        _, history, _ = modeles[n]
        afficher_courbe_apprentissage(history, n, ax=axes[i])
    for j in range(len(tailles), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Courbes d'apprentissage — convergence des modèles",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('courbes_apprentissage.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée : courbes_apprentissage.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {DEVICE}\n")

    # ── ADAPTE CES CHEMINS à ton arborescence ────────────────────────────────
    # train.py sauvegarde dans :
    #   checkpoints_n{n}/model_best.pt       ← recommandé
    #   checkpoints_n{n}/model_n{n}_final.pt
    CHEMINS = {
        10 : 'checkpoints_n10/model_best.pt',
        20 : 'checkpoints_n20/model_best.pt',
        50 : 'checkpoints_n50/model_best.pt',
        100: 'checkpoints_n100/model_best.pt',
        200: 'checkpoints_n200/model_best.pt',
    }

    modeles = charger_tous_les_modeles(CHEMINS, DEVICE)

    if not modeles:
        print("Aucun modèle chargé. Vérifier les chemins dans CHEMINS.")
    else:
        # Étape 0 — vérifier que les modèles ont bien convergé
        afficher_toutes_courbes_apprentissage(modeles)

        # Étape 1 — inférence + vérification de validité
        rapport = run_verification_complete(
            modeles          = modeles,
            tailles          = sorted(modeles.keys()),
            n_instances_test = 5,
            n_vehicles       = 3,
            seed_offset      = 2000,
            device           = DEVICE,
        )

        # Étape 2 — visualisation des chemins
        afficher_rapport_visuel(rapport, n_colonnes=3, max_par_taille=3)

        # Étape 3 — bilan global + décision feu vert / rouge
        afficher_bilan_global(rapport)
