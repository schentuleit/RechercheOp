"""
train.py — Entraînement REINFORCE + baseline greedy
=====================================================
Kool et al. 2019 adapté multi-véhicules — v2

Fonctionnalités :
    - Gradient accumulation (batch simulé)
    - Learning rate scheduler (StepLR)
    - Sauvegarde automatique du meilleur modèle
    - Détection d'effondrement + early stopping
    - Reprise depuis checkpoint
"""

import torch
import numpy as np
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from preprocess import generate_instance, preprocess
from model      import VRPTWModel
from env        import VRPTWEnv


def run_episode(model, instance, node_features, device, greedy=False):
    env = VRPTWEnv(instance, node_features)
    obs = env.reset()
    nf_tensor = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(device)

    if greedy:
        with torch.no_grad():
            H, h_bar = model.encode(nf_tensor)
    else:
        H, h_bar = model.encode(nf_tensor)

    log_probs_episode = []
    entropy_episode   = []
    max_steps = instance['n'] * instance['n_vehicles'] * 2 + 10

    for _ in range(max_steps):
        if env.is_done():
            break
        ctx     = env.get_context_features()
        mask_np = obs['mask']

        if not mask_np.any():
            k = env.get_active_vehicle()
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
            lp_t       = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
            lp_cpu     = lp_t.detach().cpu()
            probs_cpu  = lp_cpu.exp()
            probs_cpu  = torch.nan_to_num(probs_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            probs_cpu  = probs_cpu.clamp(min=0.0)
            probs_cpu  = probs_cpu / (probs_cpu.sum(dim=-1, keepdim=True) + 1e-8)
            action_cpu = torch.multinomial(probs_cpu, num_samples=1).squeeze(-1)
            action     = action_cpu.to(lp_t.device).clamp(0, lp_t.shape[-1] - 1)
            log_prob   = lp_t.gather(1, action.unsqueeze(-1)).squeeze(-1)
            log_probs_episode.append(log_prob)
            entropy    = -(probs_cpu * lp_cpu).sum(dim=-1).to(lp_t.device)
            entropy_episode.append(entropy)

        obs, _ = env.step(action.item(), log_prob=0.0)

    if not env.is_done():
        env.force_done()

    return env.total_dist, log_probs_episode, entropy_episode


def _tous_bloques(env):
    t_save = env.t_dispo.copy()
    for k in range(env.K):
        env.t_dispo    = np.full(env.K, np.inf)
        env.t_dispo[k] = t_save[k]
        masque         = env.compute_mask()
        env.t_dispo    = t_save.copy()
        if masque.any():
            return False
    return True


def compute_reinforce_loss(log_probs_sample, entropy_episode, L_sample, L_greedy,
                           entropy_coeff=0.01):
    if not log_probs_sample:
        return torch.tensor(0.0, requires_grad=True)
    sum_log_probs  = torch.stack(log_probs_sample).sum()
    avantage       = float(L_sample - L_greedy)
    reinforce_loss = torch.nan_to_num(avantage * sum_log_probs, nan=0.0)
    if entropy_episode:
        mean_entropy  = torch.nan_to_num(torch.stack(entropy_episode).mean(), nan=0.0)
        entropy_bonus = -entropy_coeff * mean_entropy
    else:
        entropy_bonus = torch.tensor(0.0)
    return reinforce_loss + entropy_bonus


def train(
    n_clients           = 20,
    n_vehicles          = 3,
    n_epochs            = 200,
    steps_per_epoch     = 200,
    accumulation        = 8,
    lr                  = 1e-4,
    lr_decay_every      = 50,
    lr_decay_gamma      = 0.5,
    entropy_coeff       = 0.01,
    save_every          = 10,
    collapse_threshold  = 30.0,
    collapse_patience   = 4,
    save_dir            = None,
    device_str          = 'auto',
    checkpoint_path     = None,
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if save_dir is None:
        save_dir = os.path.join(HERE, f'checkpoints_n{n_clients}')
    os.makedirs(save_dir, exist_ok=True)

    model     = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma
    )

    history = {
        'epoch': [], 'dist_sample': [], 'dist_greedy': [],
        'loss': [], 'gap': [], 'lr': [],
    }
    epoch_start = 1
    config = {
        'n_clients': n_clients, 'n_vehicles': n_vehicles,
        'lr': lr, 'accumulation': accumulation,
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
        if history['dist_greedy']:
            print(f"  Meilleure greedy : {min(history['dist_greedy']):.1f}")

    print(f"\n{'='*60}")
    print(f"  Entraînement VRPTW — n={n_clients} clients")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Paramètres    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_clients     : {n_clients}  |  n_vehicles : {n_vehicles}")
    print(f"  Epochs        : {epoch_start} → {n_epochs} × {steps_per_epoch} steps")
    print(f"  Batch simulé  : {accumulation} instances")
    print(f"  Total vu      : {(n_epochs - epoch_start + 1) * steps_per_epoch * accumulation:,} instances")
    print(f"  lr            : {optimizer.param_groups[0]['lr']:.2e}  decay ×{lr_decay_gamma}/{lr_decay_every} epochs")
    print(f"{'='*60}\n")

    best_greedy              = min(history['dist_greedy']) if history['dist_greedy'] else float('inf')
    epochs_sans_amelioration = 0
    best_path                = os.path.join(save_dir, 'model_best.pt')

    for epoch in range(epoch_start, n_epochs + 1):
        t0 = time.time()
        losses, dist_s, dist_g = [], [], []
        model.train()

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            step_losses, step_dist_s, step_dist_g = [], [], []

            for acc in range(accumulation):
                instance      = generate_instance(n_clients, n_vehicles, seed=None)
                node_features = preprocess(instance)

                L_s, lps, entropies = run_episode(
                    model, instance, node_features, device, greedy=False
                )
                model.eval()
                L_g, _, _ = run_episode(
                    model, instance, node_features, device, greedy=True
                )
                model.train()

                loss = compute_reinforce_loss(lps, entropies, L_s, L_g, entropy_coeff)
                loss = loss / accumulation
                if loss.requires_grad:
                    loss.backward()

                step_losses.append(loss.item() * accumulation)
                step_dist_s.append(L_s)
                step_dist_g.append(L_g)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            losses.append(np.mean(step_losses))
            dist_s.append(np.mean(step_dist_s))
            dist_g.append(np.mean(step_dist_g))

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        t1         = time.time()
        m_s        = np.mean(dist_s)
        m_g        = np.mean(dist_g)
        m_loss     = np.mean(losses)
        gap        = (m_s - m_g) / m_g * 100 if m_g > 0 else 0

        history['epoch'].append(epoch)
        history['dist_sample'].append(m_s)
        history['dist_greedy'].append(m_g)
        history['loss'].append(m_loss)
        history['gap'].append(gap)
        history['lr'].append(current_lr)

        print(f"  Epoch {epoch:3d}/{n_epochs}"
              f"  sample={m_s:7.1f}"
              f"  greedy={m_g:7.1f}"
              f"  gap={gap:+5.1f}%"
              f"  loss={m_loss:9.1f}"
              f"  lr={current_lr:.1e}"
              f"  {t1-t0:.1f}s")

        # ── Meilleur modèle ───────────────────────────────────────────
        if m_g < best_greedy:
            best_greedy              = m_g
            epochs_sans_amelioration = 0
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'history': history, 'config': config,
            }, best_path)
            print(f"  ★ Nouveau meilleur : {m_g:.1f} → {best_path}")
        else:
            epochs_sans_amelioration += 1
            degradation = (m_g - best_greedy) / best_greedy * 100
            if degradation > 15:
                print(f"  ⚠  Dégradation {degradation:+.1f}% vs meilleur ({best_greedy:.1f})")
            if degradation > collapse_threshold and epochs_sans_amelioration >= collapse_patience:
                print(f"\n  EFFONDREMENT — arrêt epoch {epoch}  meilleur={best_greedy:.1f}")
                print(f"  Modèle sauvegardé : {best_path}")
                break

        if epoch % save_every == 0:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'history': history, 'config': config,
            }, path)
            print(f"  → checkpoint : {path}")

    final = os.path.join(save_dir, f'model_n{n_clients}_final.pt')
    torch.save({
        'epoch': n_epochs, 'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'history': history, 'config': config,
    }, final)
    print(f"\n  Terminé — {final}")
    print(f"  Meilleur    — {best_path}  (greedy={best_greedy:.1f})\n")
    return model, history


def load_model(path, device_str='auto'):
    """Charge un checkpoint .pt — retourne (model, history, config)"""
    device = torch.device('cuda' if (device_str == 'auto' and torch.cuda.is_available()) else 'cpu')
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    model  = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Chargé : {path}  (epoch {ckpt['epoch']})")
    return model, ckpt.get('history', {}), ckpt.get('config', {})


if __name__ == '__main__':
    # Par défaut : entraîne n=20
    train(n_clients=20)