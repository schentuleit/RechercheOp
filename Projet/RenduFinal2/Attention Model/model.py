from __future__ import annotations

import torch
import torch.nn as nn


class VRPTWAttentionModel(nn.Module):
    """
    Attention Model pour le VRPTW — scoreur du prochain nœud.

    Architecture
    ------------
    1. Encodeur statique (Transformer)
       - Projection linéaire des features de nœuds → d_model
       - N couches de Multi-Head Self-Attention + Feed-Forward (pre-norm)
       - Chaque nœud produit un embedding qui capte les relations avec TOUS
         les autres nœuds simultanément — c'est l'avantage clé sur le MLP.

    2. Décodeur contextuel (une passe par décision)
       - Contexte de requête : [h_courant | h_moyen | proj(global)]
       - Clés enrichies : concat(h_i, action_features_i) projeté
       - Score d'attention : (q·k) / sqrt(d) clipé à [-10, 10] via tanh
         (convention AM, Kool et al. 2019)
       - Masquage des nœuds infaisables → -1e9 avant softmax

    Entrées
    -------
    node_features   : (B, N, node_dim)  — features statiques (inchangées)
    action_features : (B, N, action_dim) — features dynamiques par action
    current_node    : (B,)               — index du nœud courant
    dynamic_global  : (B, global_dim)    — état global dynamique
    feasible_mask   : (B, N)             — 1 si action faisable

    Sortie
    ------
    logits masqués : (B, N)
    """

    def __init__(
        self,
        node_dim: int = 7,
        action_dim: int = 5,
        global_dim: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── Encodeur statique ────────────────────────────────────────────────
        self.input_proj = nn.Linear(node_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LayerNorm — entraînement plus stable
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # évite les warnings PyTorch ≥ 2.0
        )

        # ── Décodeur contextuel ──────────────────────────────────────────────
        # Projection de l'état global dynamique
        self.global_proj = nn.Linear(global_dim, d_model)

        # Requête : [h_courant | h_moyen | g] → d_model
        self.context_proj = nn.Linear(3 * d_model, d_model)

        # Clés : [h_i | action_features_i] → d_model
        self.key_proj = nn.Linear(d_model + action_dim, d_model)

        # Projections finales d'attention (sans biais — convention AM)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

    # ────────────────────────────────────────────────────────────────────────
    # Méthodes publiques
    # ────────────────────────────────────────────────────────────────────────

    def encode(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encodeur statique : peut être appelé une seule fois par instance
        si on veut mettre l'embedding en cache.

        Paramètre
        ---------
        node_features : (B, N, node_dim)

        Retour
        ------
        h : (B, N, d_model)
        """
        x = self.input_proj(node_features)  # (B, N, d_model)
        return self.encoder(x)              # (B, N, d_model)

    def decode(
        self,
        h: torch.Tensor,
        action_features: torch.Tensor,
        current_node: torch.Tensor,
        dynamic_global: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Une passe de décodage : score tous les nœuds candidats.

        Paramètres
        ----------
        h               : (B, N, d_model) — sortie de l'encodeur
        action_features : (B, N, action_dim) — features dynamiques
        current_node    : (B,)
        dynamic_global  : (B, global_dim)
        feasible_mask   : (B, N)

        Retour
        ------
        logits : (B, N) — masqués sur les nœuds infaisables
        """
        batch_size = h.shape[0]
        batch_idx = torch.arange(batch_size, device=h.device)

        # Embedding du nœud courant
        h_current = h[batch_idx, current_node]    # (B, d_model)

        # Résumé global du graphe
        h_mean = h.mean(dim=1)                    # (B, d_model)

        # Contexte dynamique
        g = self.global_proj(dynamic_global)       # (B, d_model)

        # Vecteur de requête
        ctx = torch.cat([h_current, h_mean, g], dim=-1)   # (B, 3·d_model)
        query = self.context_proj(ctx)                      # (B, d_model)

        # Clés enrichies par les features d'action dynamiques
        keys = self.key_proj(
            torch.cat([h, action_features], dim=-1)
        )                                                   # (B, N, d_model)

        # Score d'attention
        q = self.W_q(query).unsqueeze(1)                   # (B, 1, d_model)
        k = self.W_k(keys)                                  # (B, N, d_model)

        scores = (q @ k.transpose(-1, -2)).squeeze(1)       # (B, N)
        scores = scores / (self.d_model ** 0.5)

        # Clipping AM : évite les gradients explosifs, stabilise l'entropie
        scores = 10.0 * torch.tanh(scores / 10.0)

        # Masquage des nœuds infaisables
        logits = scores.masked_fill(feasible_mask <= 0, -1e9)
        return logits

    def forward(
        self,
        node_features: torch.Tensor,
        action_features: torch.Tensor,
        current_node: torch.Tensor,
        dynamic_global: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Paramètres
        ----------
        node_features   : (B, N, node_dim) ou (N, node_dim)
        action_features : (B, N, action_dim) ou (N, action_dim)
        current_node    : (B,) ou scalaire
        dynamic_global  : (B, global_dim) ou (global_dim,)
        feasible_mask   : (B, N) ou (N,)

        Retour
        ------
        logits masqués : (B, N) ou (N,)
        """
        squeeze_output = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            current_node = (
                current_node.unsqueeze(0)
                if current_node.dim() == 0
                else current_node
            )
            dynamic_global = dynamic_global.unsqueeze(0)
            feasible_mask = feasible_mask.unsqueeze(0)
            squeeze_output = True

        h = self.encode(node_features)
        logits = self.decode(
            h, action_features, current_node, dynamic_global, feasible_mask
        )

        if squeeze_output:
            return logits.squeeze(0)
        return logits
