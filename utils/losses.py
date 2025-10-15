import torch
import torch.nn.functional as F

# ---------- Núcleo: Symmetric Cross-Entropy (binaria) ----------
def symmetric_cross_entropy_binary(
    logits: torch.Tensor,          # (B,1,H,W)
    targets: torch.Tensor,         # (B,1,H,W) en {0,1}
    alpha: float = 0.1,
    beta: float = 1.0,
    pos_weight: torch.Tensor | None = None,  # solo para CE
    eps: float = 2e-2,             # ← suavizado de etiquetas (evita log(0/1))
    reduction: str = "mean",
    clamp_logit: float = 15.0,     # ← recorte de logits extremos
    use_float32: bool = True       # ← calcula en fp32 (estable con AMP)
) -> torch.Tensor:
    # tipos
    z = logits.float() if use_float32 else logits
    y = targets.to(dtype=z.dtype)          # {0,1} en float

    # CE estable
    if pos_weight is not None:
        pw = pos_weight.to(dtype=z.dtype, device=z.device)
        ce = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction=reduction)
    else:
        ce = F.binary_cross_entropy_with_logits(z, y, reduction=reduction)

    # RCE estable: p=σ(z_clamp), q=y suavizado en [eps,1-eps]
    z = z.clamp(-clamp_logit, clamp_logit)
    p = torch.sigmoid(z)
    q = (y * (1.0 - 2.0*eps) + eps).clamp(eps, 1.0 - eps)
    rce = -(p * torch.log(q) + (1.0 - p) * torch.log(1.0 - q))
    rce = rce.mean() if reduction == "mean" else rce.sum()

    return alpha * ce + beta * rce


# ---------- Dice (binaria) ----------
def dice_loss_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    num = 2.0 * (probs * targets).sum(dim=dims) + smooth
    den = probs.sum(dim=dims) + targets.sum(dim=dims) + smooth
    dice = 1.0 - (num / den)
    return dice.mean()

def sce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 1.0,
    pos_weight: torch.Tensor | None = None,
    eps: float = 2e-2,             # ← igual que arriba
    gamma: float = 0.5,            # peso de Dice
    smooth: float = 1.0
) -> torch.Tensor:
    # SCE robusta
    sce = symmetric_cross_entropy_binary(
        logits, targets, alpha=alpha, beta=beta,
        pos_weight=pos_weight, eps=eps,
        reduction="mean", clamp_logit=15.0, use_float32=True
    )
    # Dice estable (fp32)
    z = logits.float()
    y = targets.to(dtype=z.dtype)
    p = torch.sigmoid(z)
    num = 2.0 * (p * y).sum(dim=(2, 3)) + smooth
    den = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + smooth
    d = (1.0 - num/den).mean().to(dtype=logits.dtype)

    return (1.0 - gamma) * sce + gamma * d

def bce_dice_loss(
    logits: torch.Tensor,  # (B,1,H,W)
    targets: torch.Tensor,  # (B,1,H,W) binaria {0,1}
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: torch.Tensor | None = None,  # <- NUEVO
    eps: float = 1.0
) -> torch.Tensor:
    # BCE (con pos_weight si se provee)
    if pos_weight is not None:
        # asegurar dtype/device correctos
        pw = pos_weight.to(dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

    # Dice
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    num = 2 * (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1 - (num + eps) / (den + eps)
    dice = dice.mean()

    return bce_weight * bce + dice_weight * dice