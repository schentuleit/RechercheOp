"""
train.py — Entraînement REINFORCE + baseline greedy
====================================================
Kool et al. 2019 adapté multi-véhicules.
Fix : masque inversé (~mask_np) — env retourne True=valide, decode_step attend True=invalide.
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
        mask_np = obs['mask']   # True = ville VALIDE (convention env.py)

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

        # ~mask_np : inversion car decode_step attend True = ville INVALIDE
        # env.compute_mask() retourne True = valide → on inverse pour masked_fill
        mask_t = torch.tensor(~mask_np, dtype=torch.bool).unsqueeze(0).to(device)

        if greedy:
            with torch.no_grad():
                lp_t      = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
                action, _ = model.select_action(lp_t, greedy=True)
        # APRÈS — sampling sur CPU pour éviter les assert CUDA
        else:
            lp_t    = model.decode_step(H, h_bar, last_node_t, capa_t, t_t, mask_t)
            lp_cpu  = lp_t.detach().cpu()

            # Sampling sur CPU
            probs_cpu = lp_cpu.exp()
            probs_cpu = torch.nan_to_num(probs_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            probs_cpu = probs_cpu.clamp(min=0.0)
            probs_cpu = probs_cpu / (probs_cpu.sum(dim=-1, keepdim=True) + 1e-8)
            action_cpu = torch.multinomial(probs_cpu, num_samples=1).squeeze(-1)

            # Récupérer log_prob depuis le tensor GPU original (garde le gradient)
            action = action_cpu.to(lp_t.device)
            N = lp_t.shape[-1]
            action = action.clamp(0, N - 1)
            log_prob = lp_t.gather(1, action.unsqueeze(-1)).squeeze(-1)

            log_probs_episode.append(log_prob)

            # Entropie sur CPU
            entropy = -(probs_cpu * lp_cpu).sum(dim=-1).to(lp_t.device)
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
    reinforce_loss = avantage * sum_log_probs
    reinforce_loss = torch.nan_to_num(reinforce_loss, nan=0.0)

    if entropy_episode:
        mean_entropy  = torch.stack(entropy_episode).mean()
        mean_entropy  = torch.nan_to_num(mean_entropy, nan=0.0)
        entropy_bonus = -entropy_coeff * mean_entropy
    else:
        entropy_bonus = torch.tensor(0.0, requires_grad=True)

    return reinforce_loss + entropy_bonus


def train(
    n_clients       = 20,
    n_vehicles      = 3,
    n_epochs        = 100,
    steps_per_epoch = 200,
    lr              = 1e-4,
    entropy_coeff   = 0.01,
    save_every      = 10,
    save_dir        = None,
    device_str      = 'auto',
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if save_dir is None:
        save_dir = os.path.join(HERE, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    model     = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"  Entraînement VRPTW — Kool adapté multi-véhicules")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Paramètres    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_clients     : {n_clients}  |  n_vehicles : {n_vehicles}")
    print(f"  Epochs        : {n_epochs} × {steps_per_epoch} steps  |  lr : {lr}")
    print(f"  entropy_coeff : {entropy_coeff}")
    print(f"{'='*60}\n")

    history = {'epoch': [], 'dist_sample': [], 'dist_greedy': [], 'loss': [], 'gap': []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        losses, dist_s, dist_g = [], [], []
        model.train()

        for _ in range(steps_per_epoch):
            instance      = generate_instance(n_clients, n_vehicles, seed=None)
            node_features = preprocess(instance)

            optimizer.zero_grad()
            L_s, lps, entropies = run_episode(model, instance, node_features, device, greedy=False)

            model.eval()
            L_g, _, _ = run_episode(model, instance, node_features, device, greedy=True)
            model.train()

            loss = compute_reinforce_loss(lps, entropies, L_s, L_g, entropy_coeff)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(loss.item() if loss.requires_grad else 0.0)
            dist_s.append(L_s)
            dist_g.append(L_g)

        t1     = time.time()
        m_s    = np.mean(dist_s)
        m_g    = np.mean(dist_g)
        m_loss = np.mean(losses)
        gap    = (m_s - m_g) / m_g * 100 if m_g > 0 else 0

        history['epoch'].append(epoch)
        history['dist_sample'].append(m_s)
        history['dist_greedy'].append(m_g)
        history['loss'].append(m_loss)
        history['gap'].append(gap)

        print(f"  Epoch {epoch:3d}/{n_epochs}"
              f"  sample={m_s:7.1f}"
              f"  greedy={m_g:7.1f}"
              f"  gap={gap:+5.1f}%"
              f"  loss={m_loss:9.1f}"
              f"  {t1-t0:.1f}s")

        if epoch % save_every == 0:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(), 'history': history,
                'config': {'n_clients': n_clients, 'n_vehicles': n_vehicles,
                           'lr': lr, 'entropy_coeff': entropy_coeff},
            }, path)
            print(f"  → checkpoint : {path}")

    final = os.path.join(save_dir, 'model_final.pt')
    torch.save({
        'epoch': n_epochs, 'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(), 'history': history,
        'config': {'n_clients': n_clients, 'n_vehicles': n_vehicles,
                   'lr': lr, 'entropy_coeff': entropy_coeff},
    }, final)
    print(f"\n  Terminé — modèle sauvegardé : {final}\n")
    return model, history


def load_model(path, device_str='auto'):
    device = torch.device('cuda' if (device_str == 'auto' and torch.cuda.is_available()) else 'cpu')
    ckpt   = torch.load(path, map_location=device)
    model  = VRPTWModel(input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Chargé : {path}  (epoch {ckpt['epoch']})")
    return model, ckpt.get('history', {}), ckpt.get('config', {})


if __name__ == '__main__':
    train(
        n_clients       = 20,
        n_vehicles      = 3,
        n_epochs        = 100,
        steps_per_epoch = 200,
        lr              = 1e-4,
        entropy_coeff   = 0.01,
        save_every      = 10,
    )