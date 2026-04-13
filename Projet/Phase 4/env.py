"""
env.py — Environnement VRPTW multi-véhicules
=============================================
v3 — ajout POMO

Conventions d'index :
    - index 0     = dépôt
    - index 1..n  = clients
    - véhicules 0..K-1

Modification vs v2 :
    - reset_with_first_node(first_node) : reset standard puis force
      le premier client visité par le véhicule 0. Utilisé par POMO
      pour diversifier les rollouts sans toucher à l'architecture.
"""

import numpy as np


class VRPTWEnv:

    def __init__(self, instance, node_features):
        self.coords        = instance['coords']
        self.demands_raw   = instance['demands']
        self.tw_raw        = instance['time_windows']
        self.service_raw   = instance['service_times']
        self.durees        = instance['durees']
        self.capacity      = instance['capacity']
        self.horizon       = instance['horizon']
        self.n_clients     = instance['n']
        self.K             = instance['n_vehicles']
        self.node_features = node_features
        self.N             = self.n_clients + 1

        self.visited       = None
        self.pos           = None
        self.t_dispo       = None
        self.capa_restante = None
        self.log_probs     = []
        self.total_dist    = 0.0
        self.done          = False

    # ── Reset standard ────────────────────────────────────────────────────────

    def reset(self):
        self.visited       = np.zeros(self.N, dtype=bool)
        self.visited[0]    = True
        self.pos           = np.zeros(self.K, dtype=int)
        self.t_dispo       = np.zeros(self.K, dtype=float)
        self.capa_restante = np.full(self.K, self.capacity, dtype=float)
        self.log_probs     = []
        self.total_dist    = 0.0
        self.done          = False
        return self._get_obs()

    # ── Reset POMO ────────────────────────────────────────────────────────────

    def reset_with_first_node(self, first_node: int):
        """
        Reset standard puis force le premier client visité.

        Utilisé par POMO : on génère n rollouts depuis le même dépôt,
        chacun avec un premier client différent. Le dépôt reste index 0,
        la structure dépôt → tournée → dépôt est inchangée.

        Paramètres
        ----------
        first_node : int
            Index du premier client à visiter (1 ≤ first_node ≤ n).
            Le pas est effectué sans log_prob (pas de gradient sur ce pas forcé).

        Retourne
        --------
        obs : dict — état après le premier pas forcé
        valide : bool — False si first_node est invalide dès le départ
                        (hors TW ou hors capacité), auquel que ce rollout
                        est ignoré dans run_pomo().
        """
        self.reset()

        # Vérifier que first_node est dans le masque initial
        masque_initial = self.compute_mask()
        if not masque_initial[first_node]:
            # Client inaccessible dès le départ — rollout invalide
            return self._get_obs(), False

        # Effectuer le pas forcé sans log_prob (pas de gradient)
        obs, _ = self.step(first_node, log_prob=0.0)
        return obs, True

    # ── Transition ────────────────────────────────────────────────────────────

    def step(self, ville, log_prob):
        if self.done:
            return self._get_obs(), self.done

        k         = int(np.argmin(self.t_dispo))
        trajet    = self.durees[self.pos[k], ville]
        t_arrivee = self.t_dispo[k] + trajet

        t_arrivee = max(t_arrivee, self.tw_raw[ville, 0])
        t_arrivee = min(t_arrivee, self.tw_raw[ville, 1])

        dist_parcourue  = np.linalg.norm(self.coords[self.pos[k]] - self.coords[ville])
        self.total_dist += dist_parcourue

        self.t_dispo[k]        = t_arrivee + self.service_raw[ville]
        self.capa_restante[k] -= self.demands_raw[ville]
        self.pos[k]            = ville

        if ville != 0:
            self.visited[ville] = True

        self.log_probs.append(log_prob)

        if self.visited[1:].all():
            self._retour_depot_tous()
            self.done = True

        return self._get_obs(), self.done

    # ── Masque ────────────────────────────────────────────────────────────────

    def compute_mask(self):
        """
        Retourne un tableau bool (N,) — True = ville VALIDE pour le véhicule actif.
        Index 0 (dépôt) toujours False.
        """
        k      = int(np.argmin(self.t_dispo))
        masque = np.zeros(self.N, dtype=bool)

        for i in range(1, self.N):
            if self.visited[i]:
                continue
            if self.demands_raw[i] > self.capa_restante[k] + 1e-6:
                continue
            trajet    = self.durees[self.pos[k], i]
            t_arrivee = max(self.t_dispo[k] + trajet, self.tw_raw[i, 0])
            if t_arrivee > self.tw_raw[i, 1] + 2.0:
                continue
            t_retour = t_arrivee + self.service_raw[i] + self.durees[i, 0]
            if t_retour > self.horizon + 1e-6:
                continue
            masque[i] = True

        return masque

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def get_active_vehicle(self):
        return int(np.argmin(self.t_dispo))

    def get_context_features(self):
        k         = self.get_active_vehicle()
        capa_norm = self.capa_restante[k] / self.capacity
        t_norm    = self.t_dispo[k] / self.horizon
        return {
            'k'         : k,
            'last_node' : self.pos[k],
            'capa_norm' : float(np.clip(capa_norm, 0.0, 1.0)),
            't_norm'    : float(np.clip(t_norm,    0.0, 1.0)),
        }

    def get_reward(self):
        return -self.total_dist

    def _get_obs(self):
        return {
            'node_features' : self.node_features,
            'visited'       : self.visited.copy(),
            'context'       : self.get_context_features(),
            'mask'          : self.compute_mask(),
            'done'          : self.done,
        }

    def _retour_depot_tous(self):
        for k in range(self.K):
            if self.pos[k] != 0:
                self.total_dist += np.linalg.norm(self.coords[self.pos[k]] - self.coords[0])
                self.pos[k]      = 0

    def is_done(self):
        return self.done

    def n_clients_restants(self):
        return int((~self.visited[1:]).sum())

    def force_done(self):
        n_non_servis = int((~self.visited[1:]).sum())
        if n_non_servis > 0:
            self.total_dist += n_non_servis * self.durees.max() * 2.0
        self._retour_depot_tous()
        self.done = True