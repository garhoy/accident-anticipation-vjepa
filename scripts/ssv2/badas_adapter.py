# badas_adapter.py
# Wrapper del backbone BADAS para extraer tokens [B, N, D] y, opcionalmente,
# concatenar tokens de capas intermedias (e.g., [17,19,21,23] para Jester).
# Si D != 1024 (ViT-L), proyecta a 1024 con una capa congelada.

import torch
import torch.nn as nn

try:
    from badas import BADASModel
except Exception as e:
    raise RuntimeError(
        "No se pudo importar BADASModel. Instala BADAS-Open o añade su ruta a PYTHONPATH."
    ) from e


class BADASBackbone(nn.Module):
    def __init__(self, checkpoint: str = None, device: str = "cuda", proj_to: int = 1024):
        super().__init__()
        self.device = device
        # Ajusta a la firma real de tu BADASModel; si no usa 'checkpoint', quita el arg.
        self.model = BADASModel(checkpoint=checkpoint, device=device) if checkpoint else BADASModel(device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._proj_to = proj_to
        self._proj = None

        # Descubre dim de salida con un forward dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 16, 3, 256, 256, device=self.device)
            toks = self._encode_tokens_internal(dummy)  # [1, N, D]
            self._in_dim = int(toks.shape[-1])

        if self._in_dim != self._proj_to:
            self._proj = nn.Linear(self._in_dim, self._proj_to, bias=False).to(self.device)
            self._proj.requires_grad_(False)

    @torch.no_grad()
    def _encode_tokens_internal(self, video: torch.Tensor) -> torch.Tensor:
        """
        Intenta extraer tokens del encoder de BADAS.
        Debe devolver [B, N, D].
        Ajusta esta función según tu implementación real.
        """
        # Caso 1: API directa
        if hasattr(self.model, "encode_tokens"):
            toks = self.model.encode_tokens(video)  # [B, N, D]
            return toks

        # Caso 2: API 'encode' devuelve [B,N,D] o [B,T,P,D]
        if hasattr(self.model, "encode"):
            toks = self.model.encode(video)
            if toks.ndim == 4:  # [B,T,P,D] -> [B,N,D]
                B, T, P, D = toks.shape
                toks = toks.view(B, T * P, D)
            return toks

        # Caso 3: Buscar un módulo tipo ViT con .blocks
        vit = None
        for _, m in self.model.named_modules():
            if hasattr(m, "blocks"):
                vit = m
                break
        if vit is None:
            raise RuntimeError(
                "No encuentro encoder tipo ViT en BADASModel. Expón encode()/encode_tokens() -> [B,N,D]."
            )

        toks = vit(video)
        if isinstance(toks, (list, tuple)):
            toks = toks[-1]
        if toks.ndim == 4:  # [B,T,P,D] -> [B,N,D]
            B, T, P, D = toks.shape
            toks = toks.view(B, T * P, D)
        return toks

    @torch.no_grad()
    def encode_tokens(self, video: torch.Tensor, layers=None) -> torch.Tensor:
        """
        video: [B,T,3,H,W]
        layers:
          - None → tokens de la última capa (SSv2 / K400)
          - lista de enteros → concat tokens de esas capas por N (e.g. [17,19,21,23] para Jester/Diving-48)
        return: [B, N_concat, D_proj] con D_proj=1024
        """
        video = video.to(self.device)

        if layers is None:
            toks = self._encode_tokens_internal(video)  # [B,N,D]
            if self._proj is not None:
                toks = self._proj(toks)
            return toks

        # Hooks para capas específicas (requiere backbone tipo ViT con .blocks)
        vit = None
        for _, m in self.model.named_modules():
            if hasattr(m, "blocks"):
                vit = m
                break
        if vit is None:
            raise RuntimeError("No encontré un módulo ViT con 'blocks' para registrar hooks de capas.")

        outputs = {}
        handles = []

        def make_hook(idx):
            def hook(_m, _inp, out):
                outputs[idx] = out
            return hook

        for li in layers:
            if li < 0 or li >= len(vit.blocks):
                raise ValueError(f"Índice de capa inválido: {li}. num_blocks={len(vit.blocks)}")
            handles.append(vit.blocks[li].register_forward_hook(make_hook(li)))

        # Ejecuta un forward normal por el camino que usa tu modelo habitualmente
        _ = self._encode_tokens_internal(video)

        for h in handles:
            h.remove()

        if len(outputs) != len(layers):
            missing = [i for i in layers if i not in outputs]
            raise RuntimeError(f"No se capturaron salidas de capas: {missing}")

        toks_list = []
        for li in layers:
            t = outputs[li]
            if t.ndim == 4:  # [B,T,P,D] -> [B,N,D]
                B, T, P, D = t.shape
                t = t.view(B, T * P, D)
            toks_list.append(t)

        toks = torch.cat(toks_list, dim=1)  # concat por N
        if self._proj is not None:
            toks = self._proj(toks)
        return toks
