#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
import torch
import torch.nn as nn
import math

# --------- Minimal Attentive Pooler (Q attends to tokens X) ----------
class MLP(nn.Module):
    def __init__(self, d, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, int(d*mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(d*mlp_ratio), d)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class CrossMHSA(nn.Module):
    """Q (learnable) attends to K,V = X tokens"""
    def __init__(self, d_model, n_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop_attn = nn.Dropout(attn_drop)
        self.drop_proj = nn.Dropout(proj_drop)

    def forward(self, q_tok, x):
        # q_tok: [B,Q,D], x: [B,L,D]
        B,Q,D = q_tok.shape
        L = x.size(1)
        q = self.q(q_tok).view(B,Q,self.h,self.dh).permute(0,2,1,3)   # [B,H,Q,Dh]
        k = self.k(x).view(B,L,self.h,self.dh).permute(0,2,1,3)       # [B,H,L,Dh]
        v = self.v(x).view(B,L,self.h,self.dh).permute(0,2,1,3)       # [B,H,L,Dh]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.dh)          # [B,H,Q,L]
        attn = attn.softmax(dim=-1)
        attn = self.drop_attn(attn)
        y = attn @ v                                                  # [B,H,Q,Dh]
        y = y.transpose(1,2).contiguous().view(B,Q,D)                 # [B,Q,D]
        y = self.drop_proj(self.o(y))                                 # [B,Q,D]
        return y

class AttentivePoolerMinimal(nn.Module):
    """
    Estilo Meta: queries aprendibles que 'leen' todos los tokens.
    depth = nº de bloques [LN -> CrossAttn -> +res -> LN -> MLP -> +res]
    """
    def __init__(self, embed_dim=1024, num_queries=1, num_heads=16,
                 mlp_ratio=4.0, depth=3, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(embed_dim),
                "ln_x": nn.LayerNorm(embed_dim),
                "attn": CrossMHSA(embed_dim, n_heads=num_heads,
                                  attn_drop=attn_drop, proj_drop=drop),
                "ln_o": nn.LayerNorm(embed_dim),
                "mlp":  MLP(embed_dim, mlp_ratio=mlp_ratio, drop=drop),
            }))

    def forward(self, x):
        # x: [B,L,D]
        B,L,D = x.shape
        q = self.q.expand(B, -1, -1).contiguous()   # [B,Q,D]
        for blk in self.blocks:
            q_norm = blk["ln_q"](q)
            x_norm = blk["ln_x"](x)
            q = q + blk["attn"](q_norm, x_norm)
            q = q + blk["mlp"](blk["ln_o"](q))
        return q  # [B,Q,D]

# ----------------------- I/O & main -----------------------
def iter_pt_files(in_dirs):
    for d in in_dirs:
        d = Path(d)
        if not d.exists(): continue
        for p in sorted(d.rglob("*.pt")):
            yield p

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clips-per-forward", type=int, default=64)
    # pooler HP (buenos defaults para ViT-L)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--queries", type=int, default=1)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn-dropout", type=float, default=0.0)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    dev = args.device

    n_ok = n_skip = 0
    for pt_path in iter_pt_files(args.in_dirs):
        data = torch.load(pt_path, map_location="cpu")
        tokens = data.get("tokens", None)
        if tokens is None:
            print(f"[skip] {pt_path} (no tokens)"); n_skip += 1; continue

        # tokens: [N_clip, L, D] (fp16)
        N,L,D = tokens.shape
        D = int(D)

        pooler = AttentivePoolerMinimal(
            embed_dim=D, num_queries=args.queries, num_heads=args.heads,
            mlp_ratio=args.mlp_ratio, depth=args.depth,
            drop=args.dropout, attn_drop=args.attn_dropout
        ).to(dev).eval()

        # procesa por lotes de clips para no quedarte sin VRAM
        z_list = []
        for i0 in range(0, N, args.clips-per-forward if hasattr(args,'clips-per-forward') else args.clips_per_forward):
            i1 = min(N, i0 + args.clips_per_forward)
            x = tokens[i0:i1].to(dev, non_blocking=True).float()  # [B,L,D]
            q = pooler(x)                                         # [B,Q,D]
            z = q.mean(dim=1)                                     # [B,D] (si Q>1)
            z_list.append(z.cpu())

        z_attn = torch.cat(z_list, dim=0)         # [N_clip, D]

        # Espeja el árbol de dirs bajo out_root
        rel = pt_path.relative_to(pt_path.anchor) if pt_path.drive else pt_path
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # preserva lo anterior y reemplaza z_clip por el nuevo
        data["z_clip_raw"] = data.get("z_clip", None)
        data["z_clip"] = z_attn.to(torch.float32)
        data.setdefault("meta", {})["z_source"] = "attn_pool_tokens_offline_v1"

        torch.save(data, out_path)
        print(f"[OK] {pt_path} -> {out_path}   (z_clip {tuple(z_attn.shape)})")
        n_ok += 1

    print(f"\nDone. ok={n_ok} skip(no tokens)={n_skip}  out_root={out_root}")

if __name__ == "__main__":
    main()
