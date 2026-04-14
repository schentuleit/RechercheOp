"""
train.py — Entraînement POMO + REINFORCE
=========================================
Kool et al. 2019 + POMO (Kwon et al. 2020) adapté VRPTW multi-véhicules — v3

Différences vs v2 (Kool pur) :
    - run_episode_from(first_node) : épisode depuis un premier client forcé
    - run_pomo(n_starts)           : n_starts rollouts sur la même instance,
                                     baseline = moyenne des coûts,
                                     gradient sur le meilleur rollout
    - train() : paramètre n_starts, log enrichi (best/baseline/gap_pomo)
    - __main__ : appels séparés par taille avec n_vehicles=None (dynamique)

model.py et preprocess.py sont inchangés.
"""

import torch
import numpy as np
import time
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from preprocess import generate_instance, preprocess, n_vehicles_dynamique
from model      import VRPTWModel
from env        import VRPTWEnv


# ─────────────────────────────────────────────────────────────────────────────
# 1. UTILITAIRE — blocage total
# ─────────────────────────────────────────────────────────────────────────────

def _tous_bloques(env: VRPTWEnv) -> bool:
    t_save = env.t_dispo.copy()
    for k in range(env.K):
        env.t_dispo    = np.full(env.K, np.inf)
        env.t_dispo[k] = t_save[k]
        masque         = env.compute_mask()
        env.t_dispo    = t_save.copy()
        if masque.any():
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ÉPISODE DEPUIS UN PREMIER CLIENT OPTIONNELLEMENT FORCÉ
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_from(model, instance, node_features, H, h_bar, device,
                     first_node=None, greedy=False):
    """
    Roule un épisode complet.

    H, h_bar sont pré-calculés et partagés entre tous les rollouts POMO
    d'une même instance — l'encodeur ne tourne qu'une seule fois.

    first_node : int ou None
        Si fourni, ce client est forcé comme premier pas (sans gradient).
        Si None, le modèle choisit librement dès le premier pas.

    Retourne (cout, log_probs, entropies, valide)
        valide=False si first_node était inaccessible → rollout ignoré.
    """
    env = VRPTWEnv(instance, node_features)

    if first_node is not None:
        obs, valide = env.reset_with_first_node(first_node)
        if not valide:
            return float('inf'), [], [], False
    else:
        obs    = env.reset()
        valide = True

    log_probs_episode = []
    entropy_episode   = []
    max_steps = instance['n'] * instance['n_vehicles'] * 4 + 50

    for _ in range(max_steps):
        if env.is_done():
            break

        ctx     = env.get_context_features()
        mask_np = obs['mask']

        if not mask_np.any():
            k = env.get_active_vehicle()
            prochaine = env.prochaine_tw_accessible()

            if prochaine < float('inf'):
                # Faire avancer l'horloge du véhicule jusqu'à la prochaine TW
                env.t_dispo[k] = prochaine
                obs = env._get_obs()
                continue
            else:
                # Vraiment bloqué — tuer ce véhicule
                env.t_dispo[k] = env.horizon + 1.0
                if _tous_bloques(env):
                    env.force_done()
                    break
                obs = env._get_obs()
                continue

        last_node_t = torch.tensor([ctx['last_node']], dtype=torch.long).to(device)
        capa_t      = torch.tensor([ctx['capa_norm']], dtype=torch.float32).to(device)
        t_t         = torch.tensor([ctx['t_norm']],    dtype=torch.float32).to(device)
        mask_t      = torch.tensor(~mask_np, dtype=torch.bool).unsqueeze(0).to(device)

        if greedy:
            with torch.no_grad():
                lp_t      = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
                action, _ = model.select_action(lp_t, greedy=True)
        else:
            lp_t      = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
            lp_cpu    = lp_t.detach().cpu()
            probs_cpu = lp_cpu.exp()
            probs_cpu = torch.nan_to_num(probs_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            probs_cpu = probs_cpu.clamp(min=0.0)
            probs_cpu = probs_cpu / (probs_cpu.sum(dim=-1, keepdim=True) + 1e-8)
            action_cpu = torch.multinomial(probs_cpu, num_samples=1).squeeze(-1)
            action     = action_cpu.to(lp_t.device).clamp(0, lp_t.shape[-1] - 1)
            log_prob   = lp_t.gather(1, action.unsqueeze(-1)).squeeze(-1)
            log_probs_episode.append(log_prob)
            entropy = -(probs_cpu * lp_cpu).sum(dim=-1).to(lp_t.device)
            entropy_episode.append(entropy)

        obs, _ = env.step(action.item(), log_prob=0.0)

    if not env.is_done():
        env.force_done()

    return env.total_dist, log_probs_episode, entropy_episode, valide


# ─────────────────────────────────────────────────────────────────────────────
# 3. POMO — n_starts rollouts sur la même instance
# ─────────────────────────────────────────────────────────────────────────────

def run_pomo(model, instance, node_features, device, n_starts=None):
    """
    Lance n_starts rollouts stochastiques depuis n_starts premiers clients
    différents, sur la même instance.

    Stratégie :
        - Encodeur tourne UNE SEULE FOIS → H, h_bar partagés
        - n_starts décodages séquentiels, chacun depuis un first_node différent
        - baseline = moyenne des coûts valides
        - gradient = sur le rollout avec le meilleur coût
        - avantage = L_best - baseline (≤ 0 → gradient correct)

    n_starts=None → tous les n clients comme points de départ.
    Réduire à 10 si GPU < 8 Go.
    """
    n = instance['n']
    if n_starts is None:
        n_starts = n

    # Encodage unique partagé entre tous les rollouts
    nf_tensor = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(device)
    H, h_bar  = model.encode(nf_tensor)

    # Choisir les first_nodes
    if n_starts >= n:
        first_nodes = list(range(1, n + 1))
    else:
        pas = n / n_starts
        first_nodes = [max(1, min(n, round(1 + i * pas))) for i in range(n_starts)]
        first_nodes = list(dict.fromkeys(first_nodes))

    tous_couts, tous_lps, tous_ent = [], [], []

    for fn in first_nodes:
        cout, lps, ent, valide = run_episode_from(
            model, instance, node_features, H, h_bar,
            device, first_node=fn, greedy=False,
        )
        if not valide:
            continue
        tous_couts.append(cout)
        tous_lps.append(lps)
        tous_ent.append(ent)

    # Garde-fou : aucun rollout valide → rollout libre
    if not tous_couts:
        cout, lps, ent, _ = run_episode_from(
            model, instance, node_features, H, h_bar,
            device, first_node=None, greedy=False,
        )
        tous_couts = [cout]
        tous_lps   = [lps]
        tous_ent   = [ent]

    baseline = float(np.mean(tous_couts))
    idx_best = int(np.argmin(tous_couts))

    return (tous_couts[idx_best], tous_lps[idx_best],
            tous_ent[idx_best], baseline, len(tous_couts))


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOSS REINFORCE avec baseline POMO
# ─────────────────────────────────────────────────────────────────────────────

def compute_reinforce_loss(log_probs, entropies, L_best, baseline, entropy_coeff=0.01):
    """
    avantage = L_best - baseline
    Toujours ≤ 0 par construction : pousse la politique vers le meilleur rollout.
    """
    if not log_probs:
        return torch.tensor(0.0, requires_grad=True)

    sum_log_probs  = torch.stack(log_probs).sum()
    avantage       = float(L_best - baseline)
    reinforce_loss = torch.nan_to_num(avantage * sum_log_probs, nan=0.0)

    if entropies:
        mean_entropy  = torch.nan_to_num(torch.stack(entropies).mean(), nan=0.0)
        entropy_bonus = -entropy_coeff * mean_entropy
    else:
        entropy_bonus = torch.tensor(0.0)

    return reinforce_loss + entropy_bonus


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_clients          = 20,
    n_vehicles         = None,
    n_starts           = None,
    n_epochs           = 200,
    steps_per_epoch    = 50,
    accumulation       = 8,
    lr                 = 1e-4,
    lr_decay_every     = 50,
    lr_decay_gamma     = 0.5,
    entropy_coeff      = 0.01,
    save_every         = 10,
    collapse_threshold = 30.0,
    collapse_patience  = 4,
    save_dir           = None,
    device_str         = 'auto',
    checkpoint_path    = None,
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if n_vehicles is None:
        n_vehicles = n_vehicles_dynamique(n_clients)

    if n_starts is None:
        n_starts = n_clients

    if save_dir is None:
        save_dir = os.path.join(HERE, f'checkpoints_n{n_clients}')
    os.makedirs(save_dir, exist_ok=True)

    model     = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma,
    )

    history = {
        'epoch'         : [],
        'dist_best'     : [],
        'dist_baseline' : [],
        'loss'          : [],
        'gap_pomo'      : [],
        'lr'            : [],
    }
    epoch_start = 1
    config = {
        'n_clients'   : n_clients,
        'n_vehicles'  : n_vehicles,
        'n_starts'    : n_starts,
        'lr'          : lr,
        'accumulation': accumulation,
        'mode'        : 'POMO',
    }

    if checkpoint_path is not None:
        ckpt        = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        history     = ckpt.get('history', history)
        epoch_start = ckpt['epoch'] + 1
        config      = ckpt.get('config', config)
        if 'lr' not in history:
            history['lr'] = [lr] * len(history['epoch'])
        for _ in range(epoch_start - 1):
            scheduler.step()
        print(f"  Reprise depuis epoch {epoch_start}")
        if history.get('dist_best'):
            print(f"  Meilleur POMO connu : {min(history['dist_best']):.1f}")

    print(f"\n{'='*65}")
    print(f"  Entraînement POMO-VRPTW — n={n_clients} clients")
    print(f"{'='*65}")
    print(f"  Device        : {device}")
    print(f"  Paramètres    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_clients     : {n_clients}  |  n_vehicles : {n_vehicles}  |  n_starts : {n_starts}")
    print(f"  Epochs        : {epoch_start} → {n_epochs} × {steps_per_epoch} steps")
    print(f"  Batch simulé  : {accumulation} instances × {n_starts} rollouts POMO")
    print(f"  Rollouts/ep.  : {steps_per_epoch * accumulation * n_starts:,}")
    print(f"  lr            : {optimizer.param_groups[0]['lr']:.2e}  "
          f"decay ×{lr_decay_gamma}/{lr_decay_every} epochs")
    print(f"{'='*65}\n")

    best_dist                = min(history['dist_best']) if history.get('dist_best') else float('inf')
    epochs_sans_amelioration = 0
    best_path                = os.path.join(save_dir, 'model_best.pt')

    for epoch in range(epoch_start, n_epochs + 1):
        t0 = time.time()
        losses, dist_bests, dist_baselines = [], [], []
        model.train()

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            step_losses, step_bests, step_baselines = [], [], []

            for _ in range(accumulation):
                instance      = generate_instance(n_clients, seed=None)
                node_features = preprocess(instance)

                L_best, lps, ent, baseline, _ = run_pomo(
                    model, instance, node_features, device, n_starts=n_starts,
                )

                loss = compute_reinforce_loss(lps, ent, L_best, baseline, entropy_coeff)
                loss = loss / accumulation
                if loss.requires_grad:
                    loss.backward()

                step_losses.append(loss.item() * accumulation)
                step_bests.append(L_best)
                step_baselines.append(baseline)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            losses.append(np.mean(step_losses))
            dist_bests.append(np.mean(step_bests))
            dist_baselines.append(np.mean(step_baselines))

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        t1         = time.time()

        m_best     = float(np.mean(dist_bests))
        m_baseline = float(np.mean(dist_baselines))
        m_loss     = float(np.mean(losses))
        gap_pomo   = (m_baseline - m_best) / m_baseline * 100 if m_baseline > 0 else 0

        history['epoch'].append(epoch)
        history['dist_best'].append(m_best)
        history['dist_baseline'].append(m_baseline)
        history['loss'].append(m_loss)
        history['gap_pomo'].append(gap_pomo)
        history['lr'].append(current_lr)

        print(f"  Epoch {epoch:3d}/{n_epochs}"
              f"  best={m_best:7.1f}"
              f"  baseline={m_baseline:7.1f}"
              f"  gap_pomo={gap_pomo:+5.1f}%"
              f"  loss={m_loss:9.1f}"
              f"  lr={current_lr:.1e}"
              f"  {t1-t0:.1f}s")

        if m_best < best_dist:
            best_dist                = m_best
            epochs_sans_amelioration = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'history'    : history,
                'config'     : config,
            }, best_path)
            print(f"  ★ Nouveau meilleur : {m_best:.1f} → {best_path}")
        else:
            epochs_sans_amelioration += 1
            degradation = (m_best - best_dist) / best_dist * 100
            if degradation > 15:
                print(f"  ⚠  Dégradation {degradation:+.1f}% vs meilleur ({best_dist:.1f})")
            if degradation > collapse_threshold and epochs_sans_amelioration >= collapse_patience:
                print(f"\n  EFFONDREMENT — arrêt epoch {epoch}  meilleur={best_dist:.1f}")
                break

        if epoch % save_every == 0:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'history'    : history,
                'config'     : config,
            }, path)
            print(f"  → checkpoint : {path}")

    final = os.path.join(save_dir, f'model_n{n_clients}_final.pt')
    torch.save({
        'epoch'      : n_epochs,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'history'    : history,
        'config'     : config,
    }, final)
    print(f"\n  Terminé — {final}")
    print(f"  Meilleur POMO — {best_path}  (best={best_dist:.1f})\n")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path, device_str='auto'):
    """Charge un checkpoint .pt — retourne (model, history, config)."""
    device = torch.device(
        'cuda' if (device_str == 'auto' and torch.cuda.is_available()) else 'cpu'
    )
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    cfg = ckpt.get('config', {})
    print(f"  Chargé : {os.path.basename(path)}  "
          f"(epoch {ckpt['epoch']}  mode={cfg.get('mode','?')}  "
          f"n_starts={cfg.get('n_starts','?')})")
    return model, ckpt.get('history', {}), cfg


# ─────────────────────────────────────────────────────────────────────────────
# 7. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    DOSSIER = '.'  # adapte ce chemin

    model_n10, history_n10 = train(
        n_clients       = 10,
        n_vehicles      = None,   # K=3
        n_starts        = None,   # 10 rollouts
        n_epochs        = 200,
        steps_per_epoch = 50,
        accumulation    = 8,
        lr              = 1e-4,
        lr_decay_every  = 50,
        lr_decay_gamma  = 0.5,
        entropy_coeff   = 0.01,
        save_every      = 10,
        save_dir        = os.path.join(DOSSIER, 'checkpoints_n10'),
    )

    model_n20, history_n20 = train(
        n_clients       = 20,
        n_vehicles      = None,   # K=3
        n_starts        = None,   # 20 rollouts
        n_epochs        = 200,
        steps_per_epoch = 50,
        accumulation    = 8,
        lr              = 1e-4,
        lr_decay_every  = 50,
        lr_decay_gamma  = 0.5,
        entropy_coeff   = 0.01,
        save_every      = 10,
        save_dir        = os.path.join(DOSSIER, 'checkpoints_n20'),
    )

    model_n50, history_n50 = train(
        n_clients       = 50,
        n_vehicles      = None,   # K=5
        n_starts        = None,   # 50 rollouts
        n_epochs        = 200,
        steps_per_epoch = 50,
        accumulation    = 8,
        lr              = 1e-4,
        lr_decay_every  = 50,
        lr_decay_gamma  = 0.5,
        entropy_coeff   = 0.01,
        save_every      = 10,
        save_dir        = os.path.join(DOSSIER, 'checkpoints_n50'),
    )

    model_n100, history_n100 = train(
        n_clients       = 100,
        n_vehicles      = None,   # K=10
        n_starts        = 10,     # limiter si GPU < 8 Go, sinon None
        n_epochs        = 200,
        steps_per_epoch = 50,
        accumulation    = 8,
        lr              = 1e-4,
        lr_decay_every  = 50,
        lr_decay_gamma  = 0.5,
        entropy_coeff   = 0.01,
        save_every      = 10,
        save_dir        = os.path.join(DOSSIER, 'checkpoints_n100'),
    )

    model_n200, history_n200 = train(
        n_clients       = 200,
        n_vehicles      = None,   # K=20
        n_starts        = 10,     # limiter si GPU < 8 Go, sinon None
        n_epochs        = 200,
        steps_per_epoch = 50,
        accumulation    = 8,
        lr              = 1e-4,
        lr_decay_every  = 50,
        lr_decay_gamma  = 0.5,
        entropy_coeff   = 0.01,
        save_every      = 10,
        save_dir        = os.path.join(DOSSIER, 'checkpoints_n200'),
    )