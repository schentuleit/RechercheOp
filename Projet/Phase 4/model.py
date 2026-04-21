"""
model.py — Architecture Kool adaptée VRPTW multi-véhicules
===========================================================
4 classes dans l'ordre du flux de données :

    MultiHeadAttention   ← bloc Q,K,V avec n_heads têtes parallèles
    EncoderLayer         ← MHA + résidu + LayerNorm + FFN + résidu + LayerNorm
    Encoder              ← N couches EncoderLayer empilées + projection initiale
    AttentionDecoder     ← contexte véhicule → scores → softmax masqué

Hyperparamètres fixés (accord projet) :
    d_h      = 128   dimension des embeddings
    n_heads  = 8     têtes d'attention
    d_k      = 16    dimension par tête (d_h / n_heads)
    n_layers = 3     couches encodeur
    d_ff     = 512   dimension intermédiaire FFN
    C        = 10    clipping tanh décodeur
    input_dim = 6    features par ville [x, y, d, a, b, s]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. MultiHeadAttention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Mécanisme d'attention multi-têtes.

    Chaque tête apprend ses propres projections W_Q, W_K, W_V
    dans un espace de dimension réduite d_k = d_h / n_heads.
    Les sorties des têtes sont concaténées puis reprojetées vers d_h.

    Paramètres
    ----------
    d_h     : int — dimension des embeddings (128)
    n_heads : int — nombre de têtes (8)
    """

    def __init__(self, d_h=128, n_heads=8):
        super().__init__()

        assert d_h % n_heads == 0, "d_h doit être divisible par n_heads"

        self.d_h     = d_h
        self.n_heads = n_heads
        self.d_k     = d_h // n_heads   # 16 — dimension par tête

        # Projections linéaires pour Q, K, V
        # On projette depuis d_h vers d_h (= n_heads × d_k)
        # puis on reshapera en (batch, n_heads, seq, d_k)
        self.W_Q = nn.Linear(d_h, d_h, bias=False)
        self.W_K = nn.Linear(d_h, d_h, bias=False)
        self.W_V = nn.Linear(d_h, d_h, bias=False)

        # Projection de sortie : concaténation des têtes → d_h
        self.W_O = nn.Linear(d_h, d_h, bias=False)

        # Facteur de normalisation des scores (évite saturation softmax)
        self.scale = self.d_k ** -0.5   # 1 / √16 ≈ 0.25

    def forward(self, query, key, value, mask=None):
        """
        Paramètres
        ----------
        query : (batch, seq_q, d_h)
        key   : (batch, seq_k, d_h)
        value : (batch, seq_k, d_h)
        mask  : (batch, seq_q, seq_k) bool — True = position masquée (→ -inf)

        Dans l'encodeur : query = key = value = H  (auto-attention)
        Dans le décodeur: query = contexte véhicule, key = value = H encodé

        Retourne
        --------
        out   : (batch, seq_q, d_h)
        """
        B, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        # ── Projections ────────────────────────────────────────────────
        # (B, seq, d_h) → (B, seq, n_heads, d_k) → (B, n_heads, seq, d_k)
        Q = self.W_Q(query).view(B, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key  ).view(B, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, seq_k, self.n_heads, self.d_k).transpose(1, 2)

        # ── Scores d'attention ─────────────────────────────────────────
        # Q @ K^T : (B, n_heads, seq_q, d_k) × (B, n_heads, d_k, seq_k)
        #         → (B, n_heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ── Masquage (optionnel) ───────────────────────────────────────
        # mask=True → position invalide → score = -inf → poids = 0 après softmax
        if mask is not None:
            # mask : (B, seq_q, seq_k) → (B, 1, seq_q, seq_k) pour broadcast sur têtes
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # ── Poids d'attention ──────────────────────────────────────────
        # softmax sur la dernière dim (seq_k) : somme = 1 pour chaque query
        attn_weights = F.softmax(scores, dim=-1)   # (B, n_heads, seq_q, seq_k)

        # ── Agrégation des values ──────────────────────────────────────
        # (B, n_heads, seq_q, seq_k) × (B, n_heads, seq_k, d_k)
        # → (B, n_heads, seq_q, d_k)
        context = torch.matmul(attn_weights, V)

        # ── Recombinaison des têtes ────────────────────────────────────
        # (B, n_heads, seq_q, d_k) → (B, seq_q, n_heads, d_k) → (B, seq_q, d_h)
        context = context.transpose(1, 2).contiguous().view(B, seq_q, self.d_h)

        # ── Projection de sortie ───────────────────────────────────────
        out = self.W_O(context)   # (B, seq_q, d_h)

        return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. EncoderLayer
# ══════════════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Une couche de l'encodeur = MHA + résidu + LayerNorm + FFN + résidu + LayerNorm.

    Structure identique au Transformer original (Vaswani 2017),
    utilisée par Kool 2019.

    Paramètres
    ----------
    d_h     : int — dimension embeddings (128)
    n_heads : int — têtes attention (8)
    d_ff    : int — dimension intermédiaire FFN (512)
    """

    def __init__(self, d_h=128, n_heads=8, d_ff=512):
        super().__init__()

        # Sous-couche 1 : auto-attention multi-têtes
        self.attention = MultiHeadAttention(d_h, n_heads)
        self.norm1     = nn.LayerNorm(d_h)

        # Sous-couche 2 : Feed Forward Network
        # dim 128 → 512 → 128, avec ReLU
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm2 = nn.LayerNorm(d_h)

    def forward(self, H):
        """
        Paramètres
        ----------
        H : (batch, N, d_h) — embeddings courants

        Retourne
        --------
        H : (batch, N, d_h) — embeddings enrichis
        """
        # ── Sous-couche 1 : auto-attention ────────────────────────────
        # Auto-attention : query = key = value = H
        # Chaque ville regarde toutes les autres (pas de masque ici)
        H_attn = self.attention(H, H, H)

        # Connexion résiduelle + normalisation
        # Le résidu H préserve l'information originale à travers les couches
        H = self.norm1(H + H_attn)

        # ── Sous-couche 2 : FFN ────────────────────────────────────────
        # Transformation non-linéaire ville par ville (indépendante)
        H_ff = self.ffn(H)

        # Connexion résiduelle + normalisation
        H = self.norm2(H + H_ff)

        return H


# ══════════════════════════════════════════════════════════════════════════════
# 3. Encoder
# ══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Encodeur complet = projection linéaire + N couches EncoderLayer.

    Transforme les features brutes (dim 6) en embeddings enrichis (dim 128)
    qui capturent le contexte global de l'instance.

    Tourne UNE SEULE FOIS par instance — résultat partagé entre tous les véhicules.

    Paramètres
    ----------
    input_dim : int — features par ville (6 : x,y,d,a,b,service)
    d_h       : int — dimension embeddings (128)
    n_heads   : int — têtes attention (8)
    n_layers  : int — nombre de couches (3)
    d_ff      : int — dimension FFN (512)
    """

    def __init__(self, input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512):
        super().__init__()

        # Projection initiale : 6 → 128
        # Même couche linéaire pour toutes les villes (poids partagés)
        self.input_projection = nn.Linear(input_dim, d_h)

        # N couches identiques empilées
        self.layers = nn.ModuleList([
            EncoderLayer(d_h, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.d_h = d_h

    def forward(self, node_features):
        """
        Paramètres
        ----------
        node_features : (batch, N, 6) — features normalisées [0,1]
                        N = n_clients + 1 (dépôt inclus en index 0)

        Retourne
        --------
        H     : (batch, N, d_h) — embeddings enrichis par ville
        h_bar : (batch, d_h)    — moyenne globale (résumé de l'instance)
        """
        # ── Projection initiale ────────────────────────────────────────
        # (batch, N, 6) → (batch, N, 128)
        H = self.input_projection(node_features)

        # ── N couches d'attention ──────────────────────────────────────
        # Après 3 couches, chaque h_i encode sa ville dans le contexte
        # de toute l'instance (géographie + fenêtres + demandes)
        for layer in self.layers:
            H = layer(H)

        # ── Moyenne globale ────────────────────────────────────────────
        # h_bar = résumé de l'instance entière
        # Utilisé dans le contexte du décodeur à chaque pas de temps
        h_bar = H.mean(dim=1)   # (batch, N, d_h) → (batch, d_h)

        return H, h_bar


# ══════════════════════════════════════════════════════════════════════════════
# 4. AttentionDecoder
# ══════════════════════════════════════════════════════════════════════════════

class AttentionDecoder(nn.Module):
    """
    Décodeur à attention — produit une distribution sur les villes à chaque pas.

    Reçoit le contexte du véhicule actif et les embeddings encodés,
    calcule un score de compatibilité pour chaque ville,
    applique le masque et retourne les log-probabilités.

    Appelé à CHAQUE décision (n × K fois par épisode environ).

    Paramètres
    ----------
    d_h : int — dimension embeddings (128)
    C   : float — clipping tanh (10)
    """

    def __init__(self, d_h=128, C=10.0):
        super().__init__()

        self.d_h = d_h
        self.C   = C
        self.d_k = d_h   # dans le décodeur, pas de multi-head : d_k = d_h

        # Projection du contexte véhicule vers l'espace query
        # Contexte = concat(h_bar, h_last, capa_proj, t_proj) → dim 4*d_h → d_h
        # On projette le concat depuis 4*d_h vers d_h
        self.context_proj = nn.Linear(4 * d_h, d_h, bias=False)

        # Projections Q et K pour le score de compatibilité
        self.W_Q = nn.Linear(d_h, d_h, bias=False)
        self.W_K = nn.Linear(d_h, d_h, bias=False)

        # Projections scalaires → d_h pour capa et temps
        # Un scalaire seul dans un concat serait écrasé — on le projette
        self.capa_proj = nn.Linear(1, d_h, bias=False)
        self.t_proj    = nn.Linear(1, d_h, bias=False)

        self.scale = d_h ** -0.5   # 1 / √128

    def forward(self, H, h_bar, h_last, capa_norm, t_norm, mask):
        """
        Paramètres
        ----------
        H        : (batch, N, d_h) — embeddings encodés (toutes les villes)
        h_bar    : (batch, d_h)    — moyenne globale de l'instance
        h_last   : (batch, d_h)    — embedding de la dernière ville visitée par k
        capa_norm: (batch, 1)      — capacité restante normalisée [0,1]
        t_norm   : (batch, 1)      — heure de dispo normalisée [0,1]
        mask     : (batch, N)      — True = ville invalide

        Retourne
        --------
        log_probs : (batch, N) — log π(ville | état)
                    -inf pour les villes masquées
        """
        # ── Construction du contexte véhicule ─────────────────────────
        # Projeter les scalaires capa et t vers dim d_h
        capa_emb = self.capa_proj(capa_norm)   # (batch, d_h)
        t_emb    = self.t_proj(t_norm)          # (batch, d_h)

        # Concaténer les 4 composantes du contexte
        # [h_bar | h_last | capa_emb | t_emb] → (batch, 4*d_h)
        ctx = torch.cat([h_bar, h_last, capa_emb, t_emb], dim=-1)

        # Projeter vers d_h — c'est le vecteur contexte final
        ctx = self.context_proj(ctx)   # (batch, d_h)

        # ── Score de compatibilité ─────────────────────────────────────
        # Query depuis le contexte véhicule : (batch, d_h)
        Q = self.W_Q(ctx)              # (batch, d_h)

        # Keys depuis les embeddings de villes : (batch, N, d_h)
        K = self.W_K(H)                # (batch, N, d_h)

        # Score = Q · K^T / √d_h  puis clipping tanh
        # Q : (batch, d_h) → (batch, 1, d_h) pour broadcast
        # K : (batch, N, d_h) → (batch, d_h, N) après transpose
        # résultat : (batch, 1, N) → squeeze → (batch, N)
        scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1)
        scores = scores * self.scale

        # Clipping : borne les logits dans [-C, C]
        # Empêche le softmax de saturer en début d'entraînement
        scores = self.C * torch.tanh(scores)

        # ── Masquage ───────────────────────────────────────────────────
        # mask=True → ville invalide → score = -inf → proba = 0
        scores = scores.masked_fill(mask, float('-inf'))

        # ── Log-probabilités ───────────────────────────────────────────
        # log_softmax = log(softmax(x)) — plus stable numériquement
        # que log(softmax(x)) calculé séparément
        log_probs = F.log_softmax(scores, dim=-1)   # (batch, N)

        return log_probs


# ══════════════════════════════════════════════════════════════════════════════
# 5. VRPTWModel — assemblage complet
# ══════════════════════════════════════════════════════════════════════════════

class VRPTWModel(nn.Module):
    """
    Modèle complet = Encoder + AttentionDecoder.

    Interface principale utilisée par train.py.

    Paramètres
    ----------
    input_dim : int   — features par ville (6)
    d_h       : int   — dimension embeddings (128)
    n_heads   : int   — têtes attention encodeur (8)
    n_layers  : int   — couches encodeur (3)
    d_ff      : int   — dimension FFN (512)
    C         : float — clipping tanh décodeur (10)
    """

    def __init__(self, input_dim=6, d_h=128, n_heads=8, n_layers=3, d_ff=512, C=10.0):
        super().__init__()
        self.encoder = Encoder(input_dim, d_h, n_heads, n_layers, d_ff)
        self.decoder = AttentionDecoder(d_h, C)
        self.d_h     = d_h

    def encode(self, node_features):
        """
        Encode une instance — appeler UNE SEULE FOIS par épisode.

        Paramètres
        ----------
        node_features : (batch, N, 6)

        Retourne
        --------
        H     : (batch, N, d_h)
        h_bar : (batch, d_h)
        """
        return self.encoder(node_features)

    def decode_step(self, H, h_bar, last_node, capa_norm, t_norm, mask):
        """
        Un pas de décodage — appeler à chaque décision du véhicule actif.

        Paramètres
        ----------
        H         : (batch, N, d_h)  — embeddings encodés (constants)
        h_bar     : (batch, d_h)     — moyenne globale (constante)
        last_node : (batch,) int     — index dernière ville du véhicule actif
        capa_norm : (batch,) float   — capacité restante normalisée
        t_norm    : (batch,) float   — heure de dispo normalisée
        mask      : (batch, N) bool  — True = ville invalide

        Retourne
        --------
        log_probs : (batch, N)
        """
        B = H.shape[0]

        # Récupérer l'embedding de la dernière ville visitée
        # last_node : (batch,) → index dans H
        # h_last[b] = H[b, last_node[b], :]
        idx    = last_node.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.d_h)
        h_last = H.gather(1, idx).squeeze(1)   # (batch, d_h)

        # Mettre les scalaires en forme (batch, 1) pour les projections
        capa = capa_norm.unsqueeze(-1)   # (batch, 1)
        t    = t_norm.unsqueeze(-1)      # (batch, 1)

        return self.decoder(H, h_bar, h_last, capa, t, mask)

    def forward(self, node_features, last_node, capa_norm, t_norm, mask):
        """
        Forward complet (encode + decode) — pour les tests unitaires.
        En entraînement réel, encode() et decode_step() sont appelés séparément.
        """
        H, h_bar  = self.encode(node_features)
        log_probs = self.decode_step(H, h_bar, last_node, capa_norm, t_norm, mask)
        return log_probs

    def select_action(self, log_probs, greedy=False):
        if greedy:
            action = log_probs.argmax(dim=-1)
        else:
            probs = log_probs.exp()
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / (probs_sum + 1e-8)
            # Clamp pour garantir une distribution valide
            probs = probs.clamp(min=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Clamp l'index pour éviter out-of-bounds sur GPU
        N = log_probs.shape[-1]
        action_clamped = action.clamp(0, N - 1)
        log_prob = log_probs.gather(1, action_clamped.unsqueeze(-1)).squeeze(-1)

        return action_clamped, log_prob