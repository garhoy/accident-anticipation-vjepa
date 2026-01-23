#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models for collision prediction with V-JEPA2 embeddings/tokens.

HEADS (ÚTILES):

- GRUPerFrame         (embeddings, frame-wise)
- TCNPerFrame         (embeddings, frame-wise)
- TransformerPerFrame (embeddings, frame-wise)

- BADASAttentiveProbe (tokens, frame-wise BADAS-style spatial attention)
- TemporalAttentionHead (tokens, spatial+temporal causal attention)
- CausalFiLMTCN       (tokens, FiLM + TCN causal con pooling espacial)

Factory: build_model(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torch.backends.cuda as cuda_backends
from torch.utils.checkpoint import checkpoint

torch.set_float32_matmul_precision("high")   # habilita TF32 (reduce mem/latencia)
cuda_backends.matmul.allow_tf32 = True

try:
    cuda_backends.enable_flash_sdp(True)
    cuda_backends.enable_mem_efficient_sdp(True)
    cuda_backends.enable_math_sdp(False)     # evita backend “math” que explota VRAM
except Exception:
    pass



class _FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.act(self.fc1(self.ln(x))))
        return x + self.drop(y)

def _init_anchor_grid(nq: int, scale: float = 0.6) -> torch.Tensor:
    """Grilla uniforme [-scale, scale] para romper simetrías de queries."""
    s = int(nq ** 0.5)
    s = max(1, s)
    xs = torch.linspace(-scale, scale, steps=s)
    ys = torch.linspace(-scale, scale, steps=s)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (s*s, 2)
    if grid.size(0) < nq:
        # si faltan, repite algunos
        rep = torch.cat([grid, grid[: (nq - grid.size(0))]], dim=0)
        return rep[:nq].unsqueeze(0)  # (1, Q, 2)
    return grid[:nq].unsqueeze(0)     # (1, Q, 2)

class _DeformableDecoderLayer(nn.Module):
    """
    Capa deformable 3D (causal) con Δ-bias.
    - Input q: (B,Q,D)
    - Volúmenes por nivel: (B,D,Dt,H_l,W_l) ya proyectados a D
    - Δ-map por nivel: (B,1,Dt,H_l,W_l) en [0,1]
    """
    def __init__(
        self,
        d_model: int, n_queries: int,
        num_points: int, num_levels: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        delta_lambda: float = 0.5,
        align_corners: bool = False,
        padding_mode: str = "border",     # mejor que 'zeros' en bordes
        memory_efficient: bool = False,   # menos VRAM, algo más lento
    ):
        super().__init__()
        self.D = d_model
        self.Q = n_queries
        self.P = num_points
        self.L = num_levels
        self.align_corners = bool(align_corners)
        self.padding_mode = padding_mode
        self.delta_lambda = float(delta_lambda)
        self.memory_efficient = bool(memory_efficient)

        self.ln_q = nn.LayerNorm(d_model)

        # Ancla base no idéntica (entrenable)
        self.anchor0 = nn.Parameter(_init_anchor_grid(self.Q, scale=0.6))  # (1,Q,2)

        # Ancla desplazable a partir de q
        self.spatial_anchor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )
        nn.init.uniform_(self.spatial_anchor[-1].weight, -0.01, 0.01)
        nn.init.uniform_(self.spatial_anchor[-1].bias,   -0.10, 0.10)

        # Offsets (dx,dy,dz) por nivel/punto condicionados con t_norm
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.L * self.P * 3)
        )

        # Logits de mezcla por punto (por nivel)
        self.weight_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.L * self.P)
        )
        self.attn_drop = nn.Dropout(attn_dropout)

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = _FFN(d_model, mult=4, drop=dropout)

    def forward(
        self,
        q: torch.Tensor,                    # (B,Q,D)
        vols: List[torch.Tensor],           # list[(B,D,Dt,Hl,Wl)]
        deltas: List[torch.Tensor],         # list[(B,1,Dt,Hl,Wl)]
        t_norm: torch.Tensor,               # (B,) en [-1,1]
    ) -> torch.Tensor:
        B, Q, D = q.shape
        qn = self.ln_q(q)                                 # (B,Q,D)

        # anclas: base + término dependiente de q
        anchors = (self.anchor0 + torch.tanh(self.spatial_anchor(qn))).clamp(-1, 1)  # (B,Q,2) via broadcast

        # offsets por nivel/punto condicionados con tiempo
        t_feat = t_norm.view(B, 1, 1).expand(B, Q, 1)
        off_in = torch.cat([qn, t_feat], dim=-1)          # (B,Q,D+1)
        offsets = torch.tanh(self.offset_mlp(off_in)).view(B, Q, self.L, self.P, 3)

        w_logits = self.weight_mlp(qn).view(B, Q, self.L, self.P)  # (B,Q,L,P)

        acc = torch.zeros_like(q)
        for l in range(self.L):
            V  = vols[l]    # (B,D,Dt,Hl,Wl)
            Dm = deltas[l]  # (B,1,Dt,Hl,Wl)

            off = offsets[:, :, l, :, :]                   # (B,Q,P,3)
            dx, dy, dz = off[..., 0], off[..., 1], off[..., 2]  # (B,Q,P)

            x = (anchors[..., 0].unsqueeze(-1) + dx).clamp(-1, 1)  # (B,Q,P)
            y = (anchors[..., 1].unsqueeze(-1) + dy).clamp(-1, 1)  # (B,Q,P)

            z_base = t_norm.view(B, 1).expand(B, Q)                 # (B,Q)
            z = (z_base.unsqueeze(-1) + dz).clamp(-1, 1)            # (B,Q,P)
            z = torch.minimum(z, z_base.unsqueeze(-1))              # causal: no mirar futuro

            grid = torch.stack([x, y, z], dim=-1)                   # (B,Q,P,3)

            if self.memory_efficient:
                # loop por Q (evita (B*Q,...) en VRAM)
                v_parts = []
                for qi in range(Q):
                    grid5 = grid[:, qi, :, :].unsqueeze(1)  # (B,P,1,1,3)
                    samp  = F.grid_sample(
                        V, grid5, mode='bilinear',
                        padding_mode=self.padding_mode,
                        align_corners=self.align_corners
                    ).squeeze(-1).squeeze(-1)              # (B,D,P)
                    d_samp = F.grid_sample(
                        Dm, grid5, mode='bilinear',
                        padding_mode=self.padding_mode,
                        align_corners=self.align_corners
                    ).squeeze(-1).squeeze(-1)              # (B,1,P)→(B,P)

                    logits_pts = w_logits[:, qi, l, :] + self.delta_lambda * d_samp  # (B,P)
                    attn = self.attn_drop(torch.softmax(logits_pts, dim=-1)).unsqueeze(1)  # (B,1,P)
                    vqi = (samp * attn).sum(dim=-1)        # (B,D)
                    v_parts.append(vqi)
                v_l = torch.stack(v_parts, dim=1)          # (B,Q,D)
            else:
                N = B * Q
                grid5 = grid.view(N, self.P, 1, 1, 3)
                Vbq   = V.repeat_interleave(Q, dim=0)      # (N,D,Dt,Hl,Wl)
                Dbq   = Dm.repeat_interleave(Q, dim=0)     # (N,1,Dt,Hl,Wl)

                samp = F.grid_sample(
                    Vbq, grid5, mode='bilinear',
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners
                ).view(B, Q, D, self.P)                    # (B,Q,D,P)
                d_samp = F.grid_sample(
                    Dbq, grid5, mode='bilinear',
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners
                ).view(B, Q, self.P)                       # (B,Q,P)

                logits_pts = w_logits[:, :, l, :] + self.delta_lambda * d_samp
                attn = self.attn_drop(torch.softmax(logits_pts, dim=-1))  # (B,Q,P)
                v_l = (samp * attn.unsqueeze(2)).sum(dim=-1)              # (B,Q,D)

            acc = acc + v_l

        q = q + self.drop(self.out_proj(acc))
        q = self.ffn(q)
        return q

class DeformableEventQueryHead(nn.Module):
    """
    API:
      forward(tokens:(B,L,D_in), token_mask:Optional(B,L)) -> (B,T,nW)  # si upsample
    Convención grid_sample 5D: (x→W, y→H, z→Dt) y align_corners=False.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        n_windows: int = 4,
        grid_h: int = 16,
        grid_w: int = 16,
        frames_per_clip: int = 16,
        tubelet_size: int = 2,
        n_queries: int = 12,
        num_points: int = 12,
        num_levels: int = 2,
        num_layers: int = 3,
        d_model: int = 256,
        dropout: float = 0.10,
        attn_dropout: float = 0.05,
        delta_lambda: float = 0.5,
        upsample_to_frames: bool = True,
        use_checkpoint: bool = True,
        align_corners: bool = False,
        padding_mode: str = "border",
        memory_efficient: bool = False,
    ):
        super().__init__()
        # geometría
        self.H = int(grid_h)
        self.W = int(grid_w)
        self.T = int(frames_per_clip)
        self.tube = int(tubelet_size)
        assert self.T % self.tube == 0, "frames_per_clip debe ser múltiplo de tubelet_size"
        self.Dt = self.T // self.tube

        self.D_in = int(embed_dim)
        self.D = int(d_model)
        self.Q = int(n_queries)
        self.P = int(num_points)
        self.L = int(num_levels)

        self.nW = int(n_windows)
        self.upsample_to_frames = bool(upsample_to_frames)
        self.use_checkpoint = bool(use_checkpoint)

        # Proyección única a d_model
        self.in_proj = nn.Linear(self.D_in, self.D)

        # Queries iniciales compartidas por tiempo
        self.q0 = nn.Parameter(torch.randn(1, self.Q, self.D) * 0.02)

        # Capas deformables
        self.layers = nn.ModuleList([
            _DeformableDecoderLayer(
                d_model=self.D, n_queries=self.Q,
                num_points=self.P, num_levels=self.L,
                dropout=dropout, attn_dropout=attn_dropout,
                delta_lambda=delta_lambda,
                align_corners=align_corners,
                padding_mode=padding_mode,
                memory_efficient=memory_efficient,
            ) for _ in range(num_layers)
        ])

        # Clasificador sobre Q*D
        mlp_in = self.Q * self.D
        mlp_h  = max(512, self.D * 2)
        self.cls = nn.Sequential(
            nn.LayerNorm(mlp_in),
            nn.Linear(mlp_in, mlp_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_h, mlp_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_h, self.nW),
        )

        # Coordenadas temporales normalizadas [-1,1]
        t_coords = torch.linspace(-1.0, 1.0, steps=self.Dt)
        self.register_buffer("t_coords", t_coords, persistent=False)

    # ---------------- helpers ----------------
    def _reshape_and_project(
        self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        tokens: (B, L, D_in) con L = Dt*H*W (+1 si CLS).
        token_mask: (B, L) opcional (True=válido) – si llega, enmascara antes de proyectar.
        return: X (B,D,Dt,H,W)
        """
        B, L, Din = tokens.shape
        expected = self.Dt * self.H * self.W

        if L == expected + 1:
            tokens = tokens[:, 1:, :]
            if token_mask is not None:
                token_mask = token_mask[:, 1:]
            L -= 1

        if token_mask is not None:
            tokens = tokens * token_mask.float().unsqueeze(-1)

        if L != expected:
            raise RuntimeError(
                f"[DeformableEventQueryHead] L={L}, esperado {expected} "
                f"(Dt={self.Dt}, HxW={self.H*self.W})"
            )

        x = self.in_proj(tokens)                                # (B, L, D)
        x = x.view(B, self.Dt, self.H, self.W, self.D).contiguous()
        x = x.permute(0, 4, 1, 2, 3).contiguous()               # (B,D,Dt,H,W)
        return x

    @staticmethod
    def _pyr_down(vol: torch.Tensor) -> torch.Tensor:
        # downsample espacial (temporal intacto)
        return F.avg_pool3d(vol, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    @staticmethod
    def _delta_map(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,D,Dt,H,W) → (B,1,Dt,H,W) en [0,1]
        """
        d = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).pow(2).mean(1, keepdim=True)  # (B,1,Dt-1,H,W)
        d = F.pad(d, (0, 0, 0, 0, 1, 0))                                           # pad t=0 → (B,1,Dt,H,W)
        # min-max por batch
        B = d.size(0)
        d_ = d.view(B, -1)
        d_min = d_.min(dim=1)[0].view(B, 1, 1, 1, 1)
        d_max = d_.max(dim=1)[0].view(B, 1, 1, 1, 1)
        d = ((d - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
        return d

    def _build_pyramid(self, X: torch.Tensor):
        vols = [X]
        for _ in range(1, self.L):
            vols.append(self._pyr_down(vols[-1]))
        deltas = [self._delta_map(v) for v in vols]
        return vols, deltas

    # ---------------- forward ----------------
    def forward(
        self,
        tokens: torch.Tensor,                # (B,L,D_in)
        token_mask: Optional[torch.Tensor] = None,  # (B,L) True=válido
    ) -> torch.Tensor:
        B = tokens.size(0)
        X = self._reshape_and_project(tokens, token_mask)  # (B,D,Dt,H,W)
        vols, deltas = self._build_pyramid(X)

        # precompute t_norm para todo el batch
        t_norm_vec = self.t_coords.unsqueeze(0).expand(B, -1).contiguous()  # (B, Dt)

        logits_dt = []
        for t in range(self.Dt):
            q = self.q0.expand(B, -1, -1)                 # (B,Q,D)
            t_norm_t = t_norm_vec[:, t]                   # (B,)

            for layer in self.layers:
                if self.use_checkpoint and q.requires_grad:
                    def _layer(_q, _tn):
                        return layer(_q, vols, deltas, _tn)
                    # q = checkpoint(_layer, q, t_norm_t)
                    q = checkpoint(_layer, q, t_norm_t, use_reentrant=False, preserve_rng_state=False)
                else:
                    q = layer(q, vols, deltas, t_norm_t)

            z = q.reshape(B, -1)                          # (B,Q*D)
            logits_dt.append(self.cls(z))                 # list[(B,nW)]

        logits_dt = torch.stack(logits_dt, dim=1)         # (B,Dt,nW)

        if not self.upsample_to_frames:
            return logits_dt

        # Upsample lineal Dt -> T
        out = F.interpolate(
            logits_dt.transpose(1, 2),  # (B,nW,Dt)
            size=self.T,
            mode="linear",
            align_corners=False
        ).transpose(1, 2)               # (B,T,nW)
        return out

# ============================================================
# AUX: MLP + ATTENTION BLOCKS
# ============================================================

class _MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class _AttnBlock(nn.Module):
    """
    Bloque Transformer con self/cross-attention.
    Compatible con key_padding_mask.
    """
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.ln1_q = nn.LayerNorm(dim)
        self.ln1_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mult=4, drop=drop)
        self.drop = nn.Dropout(drop)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is None or v is None:
            k = v = q

        q_norm = self.ln1_q(q)
        kv_norm = self.ln1_kv(k)

        q2, _ = self.attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + self.drop(q2)
        q = q + self.mlp(self.ln2(q))
        return q
class AttentivePoolerMinimal(nn.Module):
    """
    Pooler atencional con Q aprendibles:
      - Q ∈ R^{nQ×D} (parámetros)
      - cross-attn: Q -> tokens
      - (L-1) self-attn sobre Q
    Soporta key_padding_mask (True = PAD).
    Opcional: gradient checkpointing para reducir pico de VRAM.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 16,
        num_layers: int = 4,
        n_queries: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim debe ser divisible por n_heads"

        self.use_checkpoint = use_checkpoint

        # Q aprendibles
        self.q = nn.Parameter(torch.randn(1, n_queries, embed_dim) * 0.02)

        # 1 bloque de cross-attn (Q -> tokens)
        self.cross = _AttnBlock(embed_dim, n_heads, drop=dropout, attn_drop=attn_dropout)

        # (num_layers-1) bloques de self-attn sobre Q
        self.blocks = nn.ModuleList([
            _AttnBlock(embed_dim, n_heads, drop=dropout, attn_drop=attn_dropout)
            for _ in range(max(0, num_layers - 1))
        ])

        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,               # (B, N, D)
        token_mask: Optional[torch.Tensor] = None,  # (B, N) True = válido
    ) -> torch.Tensor:
        B, N, D = tokens.shape

        # key_padding_mask para MHA: True = PAD
        kpm = None
        if token_mask is not None:
            kpm = (~token_mask).contiguous()

        # expandir Q a batch
        q = self.q.expand(B, -1, -1)        # (B, nQ, D)

        # helpers para checkpoint (solo acepta tensores)
        def _cross(Q, K, V, M):
            return self.cross(Q, K, V, key_padding_mask=M)

        def _self(blk, Q):
            return blk(Q, Q, Q, key_padding_mask=None)

        if self.use_checkpoint and q.requires_grad:
            # si kpm es None, pasar máscara "todo válido" (todo False = no PAD)
            if kpm is None:
                kpm_cp = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            else:
                kpm_cp = kpm

            # ✅ CORRECTO: usa _cross con los parámetros correctos
            q = checkpoint(_cross, q, tokens, tokens, kpm_cp, use_reentrant=False, preserve_rng_state=False)
            for blk in self.blocks:
                # ✅ CORRECTO: captura blk correctamente con _blk=blk
                q = checkpoint(lambda Q, _blk=blk: _self(_blk, Q), q, use_reentrant=False, preserve_rng_state=False)
        else:
            q = _cross(q, tokens, tokens, kpm)
            for blk in self.blocks:
                q = _self(blk, q)

        return self.ln(q)

# ============================================================
# BADAS ATTENTIVE PROBE (TOKENS → FRAME-WISE)
# ============================================================
class BADASAttentiveProbe(nn.Module):
    """
    Adaptación frame-wise BADAS:
      - Reorganiza tokens en (Dt, H*W, D)
      - Pooler atencional espacial por step (con máscara)
      - MLP → logits por ventana
      - Upsample temporal a frames_per_clip
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        n_windows: int = 4,
        grid_h: int = 16,
        grid_w: int = 16,
        frames_per_clip: int = 16,
        tubelet_size: int = 2,
        M: int = 12,
        d: int = 64,
        n_heads: int = 16,
        num_attn_layers: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        upsample_to_frames: bool = True,
    ):
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.frames_per_clip = int(frames_per_clip)
        self.tubelet_size = int(tubelet_size)
        self.M = int(M)
        self.d = int(d)

        self.tokens_per_step = self.grid_h * self.grid_w
        self.steps = self.frames_per_clip // self.tubelet_size
        self.upsample_to_frames = upsample_to_frames

        self.pooler = AttentivePoolerMinimal(
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_attn_layers,
            n_queries=M,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.proj = nn.Linear(embed_dim, d)
        mlp_input = M * d
        mlp_hidden = 768
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input),
            nn.Linear(mlp_input, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_windows),
        )

        print(f"[BADASAttentiveProbe] Dt={self.steps}, HxW={self.tokens_per_step}, M={M}, d={d}")

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L, D) con L = Dt*HxW (+1 si hay CLS). Devuelve (B, Dt, HxW, D).
        """
        B, L, D = tokens.shape
        expected_L = (self.frames_per_clip // self.tubelet_size) * (self.grid_h * self.grid_w)
        if L == expected_L + 1:
            # Quitar CLS
            tokens = tokens[:, 1:, :]
            L -= 1
        if L != expected_L:
            raise RuntimeError(
                f"[BADASAttentiveProbe] L={L}, esperado {expected_L} "
                f"(Dt={self.frames_per_clip//self.tubelet_size}, HxW={self.grid_h*self.grid_w})"
            )
        return tokens.view(
            B,
            self.frames_per_clip // self.tubelet_size,
            self.grid_h * self.grid_w,
            D
        ).contiguous()

    def _reshape_token_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        """
        token_mask: (B, L) bool con True=válido. Devuelve (B, Dt, HxW).
        Si L = Dt*HxW + 1, descarta CLS.
        """
        assert token_mask.dim() == 2, f"token_mask debe ser (B, L); got {tuple(token_mask.shape)}"
        B, L = token_mask.shape
        expected = (self.frames_per_clip // self.tubelet_size) * (self.grid_h * self.grid_w)
        if L == expected + 1:
            token_mask = token_mask[:, 1:]
            L -= 1
        if L != expected:
            raise RuntimeError(
                f"[BADASAttentiveProbe] token_mask L={L} != {expected} "
                f"(Dt={self.steps}, HxW={self.tokens_per_step})"
            )
        return token_mask.view(B, self.steps, self.tokens_per_step).contiguous()

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: (B, L, D)
        token_mask: (B, L) bool (opcional)
        return: (B, T, nW) si upsample_to_frames, si no (B, Dt, nW)
        """
        B, L, D = tokens.shape
        x = self._reshape_tokens(tokens)             # (B, Dt, H*W, D)
        Dt = x.size(1)

        mask_seq = None
        if token_mask is not None:
            mask_seq = self._reshape_token_mask(token_mask)  # (B, Dt, H*W)

        logits_list = []
        for t in range(Dt):
            x_t = x[:, t, :, :]                          # (B, H*W, D)
            m_t = mask_seq[:, t, :] if mask_seq is not None else None
            q = self.pooler(x_t, token_mask=m_t)         # (B, M, D)
            q_proj = self.proj(q)                        # (B, M, d)
            z = q_proj.reshape(B, -1)                    # (B, M*d)
            logits_t = self.mlp(z)                       # (B, nW)
            logits_list.append(logits_t)

        logits_dt = torch.stack(logits_list, dim=1)      # (B, Dt, nW)

        if not self.upsample_to_frames:
            return logits_dt

        logits_t = F.interpolate(
            logits_dt.transpose(1, 2),           # (B, nW, Dt)
            size=self.frames_per_clip,           # → T
            mode="linear",
            align_corners=False
        ).transpose(1, 2)                        # (B, T, nW)
        return logits_t
# ============================================================
# TEMPORAL ATTENTION HEAD (TOKENS)
# ============================================================
class TemporalAttentionHead(nn.Module):
    """
    Tokens → pooler espacial atencional por step → TF Encoder temporal causal.
    Devuelve (B, T, n_windows) (upsample desde Dt a frames_per_clip).
    Máscara: soporta token_mask (B, L) con True=válido; se reordena a (B, Dt, HxW).
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        n_windows: int = 4,
        grid_h: int = 16,
        grid_w: int = 16,
        frames_per_clip: int = 16,
        tubelet_size: int = 2,
        M: int = 6,
        proj_dim: int = 32,
        n_heads_spatial: int = 8,
        n_heads_temporal: int = 2,
        tfm_layers: int = 1,
        dropout: float = 0.30,
        attn_dropout: float = 0.10,
        token_dropout_p: float = 0.10,
        temporal_dropout_p: float = 0.05,
        freeze_pooler: bool = True,
        upsample_to_frames: bool = True,
    ):
        super().__init__()
        assert embed_dim % n_heads_spatial == 0
        assert proj_dim % n_heads_temporal == 0

        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.frames_per_clip = int(frames_per_clip)
        self.tubelet_size = int(tubelet_size)
        self.tokens_per_step = self.grid_h * self.grid_w
        self.steps = self.frames_per_clip // self.tubelet_size
        self.upsample_to_frames = upsample_to_frames

        self.token_dropout_p = float(token_dropout_p)
        self.temporal_dropout_p = float(temporal_dropout_p)

        self.pooler = AttentivePoolerMinimal(
            embed_dim=embed_dim,
            n_heads=n_heads_spatial,
            num_layers=2,
            n_queries=M,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        if freeze_pooler:
            for p in self.pooler.parameters():
                p.requires_grad = False

        self.in_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads_temporal,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        enc_layer.self_attn.dropout = attn_dropout
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=tfm_layers)

        self.out_norm = nn.LayerNorm(proj_dim)
        self.out_drop = nn.Dropout(dropout)
        self.head = nn.Linear(proj_dim, n_windows)
        self.pe = nn.Parameter(torch.randn(1, self.steps, proj_dim) * 0.02)
        self.feat_drop = nn.Dropout(dropout)

        print("[TemporalAttentionHead] init")

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape
        expected_L = self.steps * self.tokens_per_step
        if L == expected_L + 1:
            tokens = tokens[:, 1:, :]  # descarta CLS si existe
            L -= 1
        if L != expected_L:
            raise RuntimeError(
                f"[TemporalAttentionHead] L={L} != {expected_L} "
                f"(Dt={self.steps}, HxW={self.tokens_per_step})"
            )
        return tokens.view(B, self.steps, self.tokens_per_step, D)

    def _reshape_token_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        """
        token_mask: (B, L) con True = válido. Soporta L = Dt*HxW (+1 si hay CLS).
        Devuelve (B, Dt, HxW).
        """
        assert token_mask.dim() == 2, f"token_mask debe ser (B, L); got {tuple(token_mask.shape)}"
        B, L = token_mask.shape
        expected_L = self.steps * self.tokens_per_step
        if L == expected_L + 1:
            token_mask = token_mask[:, 1:]  # descarta CLS si existe
            L -= 1
        if L != expected_L:
            raise RuntimeError(
                f"[TemporalAttentionHead] token_mask L={L} != {expected_L} "
                f"(Dt={self.steps}, HxW={self.tokens_per_step})"
            )
        return token_mask.view(B, self.steps, self.tokens_per_step).contiguous()

    def _token_dropout(self, x_step: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout_p <= 0.0:
            return x_step
        B, N, D = x_step.shape
        keep = torch.rand(B, N, device=x_step.device) > self.token_dropout_p
        # garantiza al menos 1 token
        if keep.sum(dim=1).min() == 0:
            idx = torch.randint(0, N, (B,), device=x_step.device)
            keep[torch.arange(B, device=x_step.device), idx] = True
        return x_step * keep.unsqueeze(-1)

    def _temporal_keep_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        if not self.training or self.temporal_dropout_p <= 0.0:
            return torch.ones(B, T, dtype=torch.bool, device=device)
        keep = torch.rand(B, T, device=device) > self.temporal_dropout_p
        keep[:, -1] = True
        return keep

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = tokens.shape
        x = self._reshape_tokens(tokens)             # (B, Dt, N, D)
        Dt = x.size(1)
        mask_seq = None
        if token_mask is not None:
            mask_seq = self._reshape_token_mask(token_mask)  # (B, Dt, N)

        pooled = []
        for t in range(Dt):
            x_t = x[:, t, :, :]                      # (B, N, D)
            x_t = self._token_dropout(x_t)
            m_t = mask_seq[:, t, :] if mask_seq is not None else None
            q = self.pooler(x_t, token_mask=m_t)     # (B, M, D) — usa máscara
            s_t = q.mean(dim=1)                      # (B, D)
            pooled.append(s_t)
        s = torch.stack(pooled, dim=1)               # (B, Dt, D)

        y = self.proj(self.in_norm(s))               # (B, Dt, d)
        y = self.feat_drop(y) + self.pe[:, :Dt, :]

        causal_mask = self._causal_mask(Dt, tokens.device)
        keep_mask = self._temporal_keep_mask(B, Dt, tokens.device)

        y = self.temporal(
            y,
            mask=causal_mask,
            src_key_padding_mask=(~keep_mask).contiguous(),
        )
        y = self.out_drop(self.out_norm(y))          # (B, Dt, d)
        logits_dt = self.head(y)                     # (B, Dt, nW)

        if not self.upsample_to_frames:
            return logits_dt

        logits_t = F.interpolate(
            logits_dt.transpose(1, 2), size=self.frames_per_clip,
            mode="linear", align_corners=False
        ).transpose(1, 2)
        return logits_t                           # (B, T, nW)


# ============================================================
# HEADS CON EMBEDDINGS (z_clip)
# ============================================================

class GRUPerFrame(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_windows: int,
        hidden: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.ln_inp = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * 2 if bidirectional else hidden
        self.ln_out = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_windows)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D), mask: (B, T) True=válido
        """
        x = self.ln_inp(x)
        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        y, _ = self.gru(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(
            y, batch_first=True, total_length=x.size(1)
        )
        y = self.drop(self.ln_out(y))
        return self.head(y)  # (B, T, nW)

class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, d: int = 1, drop: float = 0.3):
        super().__init__()
        self.pad = nn.ConstantPad1d(((k - 1) * d, 0), 0.0)
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.relu(self.conv(self.pad(x)), inplace=True))

class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, d: int = 1, drop: float = 0.3):
        super().__init__()
        self.c1 = CausalConv1d(c_in, c_out, k, d, drop)
        self.c2 = CausalConv1d(c_out, c_out, k, d, drop)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.norm = nn.GroupNorm(1, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c2(self.c1(x))
        return self.norm(y + self.res(x))

class TCNPerFrame(nn.Module):
    """
    TCN causal sobre (B,T,D).
    """
    def __init__(
        self,
        embed_dim: int,
        n_windows: int,
        channels: int = 256,
        levels: int = 5,
        k: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.inp = nn.Conv1d(embed_dim, channels, 1)
        blocks = []
        for i in range(levels):
            d = 2 ** i
            blocks.append(TCNBlock(channels, channels, k, d, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, n_windows, 1)

        rf_frames = (k - 1) * sum(2 ** i for i in range(levels))
        print(f"[TCNPerFrame] RF={rf_frames} frames")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D), mask: (B, T) (no se aplica aquí; se usa en loss/métricas)
        """
        x = x.transpose(1, 2)                 # (B, D, T)
        x = self.blocks(self.inp(x))
        logits = self.head(x).transpose(1, 2) # (B, T, nW)
        return logits

class TransformerPerFrame(nn.Module):
    """
    Transformer causal sobre (B,T,D).
    """
    def __init__(
        self,
        embed_dim: int,
        n_windows: int,
        d_model: int = 384,      # ↑
        n_heads: int = 6,        # ↑
        num_layers: int = 4,     # ↑
        ff_dim: int = 1536,      # ↑
        dropout: float = 0.10,   # ↓
        pe_dropout: float = 0.05,# ↓
        pe_type: str = "learned",
        max_len: int = 8192,
        attn_dropout: float = 0.05,  # ↓ (aplicado de verdad)
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.in_norm = nn.LayerNorm(embed_dim)
        self.inp = nn.Linear(embed_dim, d_model)

        if pe_type == "learned":
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            self.register_buffer("pe", self._sinusoidal_pe(max_len, d_model))

        self.pe_drop = nn.Dropout(pe_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,              # FFN/output dropout
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # aplicar attn_dropout específico
        enc_layer.self_attn.dropout = attn_dropout
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_windows)

    def _sinusoidal_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    @staticmethod
    def _subsequent_mask(S: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: (B, T) True=válido
        """
        assert mask.dtype == torch.bool and mask.dim() == 2 and mask.shape[:2] == x.shape[:2], \
            f"mask debe ser bool de shape (B,T); got {mask.dtype}, {mask.shape}"

        x = self.in_norm(x)
        x = self.inp(x)

        if hasattr(self, "pe"):
            x = x + self.pe[:, : x.size(1), :]
        x = self.pe_drop(x)

        T = x.size(1)
        src_mask = self._subsequent_mask(T, x.device) if self.causal else None
        key_pad_mask = (~mask).contiguous()   # True=PAD, contiguo

        y = self.enc(x, mask=src_mask, src_key_padding_mask=key_pad_mask)
        y = self.out_drop(self.out_norm(y))
        logits = self.head(y)  # (B, T, nW)
        return logits




# ============================================================
# FACTORY
# ============================================================

def build_model(model_type: str, embed_dim: int, n_windows: int, **kw) -> nn.Module:
    mt = model_type.lower()

    # TOKENS-BASED HEADS
    if mt in ("badas_attn", "badas", "badas_fw"):
        print("[Factory] BADASAttentiveProbe")
        return BADASAttentiveProbe(
            embed_dim=kw.get("embed_dim_tokens", embed_dim),
            n_windows=n_windows,
            grid_h=kw.get("grid_h", 16),
            grid_w=kw.get("grid_w", 16),
            frames_per_clip=kw.get("frames_per_clip", 16),
            tubelet_size=kw.get("tubelet_size", 2),
            M=kw.get("M", 12),
            d=kw.get("d", 64),
            n_heads=kw.get("n_heads", 16),
            num_attn_layers=kw.get("num_attn_layers", 4),
            dropout=kw.get("dropout", 0.1),
            attn_dropout=kw.get("attn_dropout", 0.0),
            upsample_to_frames=kw.get("upsample_to_frames", True),
        )

    if mt in ("deformable_event", "defev", "deformable"):
        print("[Factory] DeformableEventQueryHead")
        return DeformableEventQueryHead(
            embed_dim=kw.get("embed_dim_tokens", embed_dim),
            n_windows=n_windows,
            grid_h=kw.get("grid_h", 16),
            grid_w=kw.get("grid_w", 16),
            frames_per_clip=kw.get("frames_per_clip", 16),
            tubelet_size=kw.get("tubelet_size", 2),
            n_queries=kw.get("n_queries", 12),
            num_points=kw.get("num_points", 12),
            num_levels=kw.get("num_levels", 2),
            num_layers=kw.get("num_layers", 3),
            d_model=kw.get("d_model", 256),
            dropout=kw.get("dropout", 0.10),
            attn_dropout=kw.get("attn_dropout", 0.05),
            delta_lambda=kw.get("delta_lambda", 0.5),
            upsample_to_frames=kw.get("upsample_to_frames", True),
            use_checkpoint=kw.get("use_checkpoint", True),
            align_corners=kw.get("align_corners", False),         
            padding_mode=kw.get("padding_mode", "border"),        
            memory_efficient=kw.get("memory_efficient", False), 
        )

    if mt in ("temporal_attn", "temporal_attention", "temp_attn"):
        print("[Factory] TemporalAttentionHead")
        return TemporalAttentionHead(
            embed_dim=kw.get("embed_dim_tokens", embed_dim),
            n_windows=n_windows,
            grid_h=kw.get("grid_h", 16),
            grid_w=kw.get("grid_w", 16),
            frames_per_clip=kw.get("frames_per_clip", 16),
            tubelet_size=kw.get("tubelet_size", 2),
            M=kw.get("M", 6),
            proj_dim=kw.get("proj_dim", 32),
            n_heads_spatial=kw.get("n_heads_spatial", 8),
            n_heads_temporal=kw.get("n_heads_temporal", 2),
            tfm_layers=kw.get("tfm_layers", 1),
            dropout=kw.get("dropout", 0.30),
            attn_dropout=kw.get("attn_dropout", 0.10),
            token_dropout_p=kw.get("token_dropout_p", 0.10),
            temporal_dropout_p=kw.get("temporal_dropout_p", 0.05),
            freeze_pooler=kw.get("freeze_pooler", True),
            upsample_to_frames=kw.get("upsample_to_frames", True),
        )

    # EMBEDDING-BASED HEADS
    if mt in ("gru", "gru_pf", "frame_gru"):
        print("[Factory] GRUPerFrame")
        return GRUPerFrame(
            embed_dim=embed_dim,
            n_windows=n_windows,
            hidden=kw.get("hidden", 512),
            num_layers=kw.get("num_layers", 2),
            dropout=kw.get("dropout", 0.3),
            bidirectional=False,
        )

    if mt in ("tcn", "temporal_cnn"):
        channels = kw.get("channels", 256)
        levels = kw.get("levels", 5)
        k = kw.get("k", 3)
        drop = kw.get("dropout", 0.4)
        print(f"[Factory] TCNPerFrame(ch={channels}, levels={levels}, k={k})")
        return TCNPerFrame(
            embed_dim=embed_dim,
            n_windows=n_windows,
            channels=channels,
            levels=levels,
            k=k,
            dropout=drop,
        )

    if mt in ("transformer", "tfm"):
        d_model = kw.get("d_model", 384)
        n_heads = kw.get("n_heads", 6)
        num_layers = kw.get("num_layers", 4)
        ff_dim = kw.get("ff_dim", 1536)
        causal = kw.get("causal", True)
        print(f"[Factory] TransformerPerFrame(d_model={d_model}, heads={n_heads}, layers={num_layers}, causal={causal})")
        return TransformerPerFrame(
            embed_dim=embed_dim,
            n_windows=n_windows,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=kw.get("dropout", 0.10),
            pe_dropout=kw.get("pe_dropout", 0.05),
            pe_type=kw.get("pe_type", "learned"),
            max_len=kw.get("max_len", 8192),
            attn_dropout=kw.get("attn_dropout", 0.05),
            causal=causal,
        )

    raise ValueError(f"Modelo desconocido: '{model_type}'")
