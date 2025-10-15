import torch
import yaml
import pyproj
import numpy as np
import geopandas as gpd
import rasterio as rio


from typing import List, Literal, Tuple, Optional, Dict
from rasterio.mask import mask as rio_mask
from shapely.ops import transform
from scipy.stats import entropy  # KL Divergence
from torch.nn import functional as F
from trainers.pl_trainer import SRSegLit
from trainers.dlinknet_trainer import RoadTrainer
from trainers.unetpp_trainer import BuildingTrainer
from utils.cloud_detection import cloud_model

# =========================
# Default
# =========================
CLEAR_FRACTION_THRESH = 0.70          # Proportion of clear pixels in S2
KLD_DIVERGENCE_THRESHOLD = 0.20       # KL divergence threshold against global histogram
BAND_IDXS = [3, 2, 1, 7, 10, 11]      # B4, B3, B2, B8, B11, B12 (channels-first)
S2_SCALE = 10_000
NBINS = 200
HIST_RANGE = (0, 1)
EPS = 1e-10
CLOUD_CKPT_PATH = "/home/tidop/projects/RefImSR/RRSGAN/weight/UNetMobV2_V2.pt"

# =========================
# Filtering Utilities
# =========================
def align_pairs(s1_images, s2_images):
    """
    Zip by index (without assuming same dates) and preserve order.
    Returns a list of dicts: {'idx','date_s1','date_s2','s1','s2'}.
    """
    n = min(len(s1_images), len(s2_images))
    pairs = []
    for i in range(n):
        s1i, s2i = s1_images[i], s2_images[i]
        pairs.append({
            "idx": i,  # índice original para trazabilidad
            "date_s1": str(s1i.get("date", "")),
            "date_s2": str(s2i.get("date", "")),
            "s1": s1i["image"],
            "s2": s2i["image"],
        })
    return pairs


def filter_zeros_nans(
    pairs
):
    """
    Deletes pairs with NaNs or zeros.
    """
    kept, removed = [], []
    for p in pairs:
        s1, s2 = p["s1"], p["s2"]
        has_nan = np.isnan(s1).any() or np.isnan(s2).any()
        has_zero = (s1 == 0).any() or (s2 == 0).any()

        if has_nan or has_zero:
            removed.append({**p, "reason": f"NaNs/zeros (nan={has_nan}, zero={bool(has_zero)})"})
        else:
            kept.append(p)

    print(f"[Zeros/NaNs] Keep {len(kept)} | Remove {len(removed)}")
    return kept, removed


def filter_clouds(pairs, ckpt_path, clear_thresh=CLEAR_FRACTION_THRESH):
    """
    Elimina por nubosidad usando cloud_model -> (H,W) con 0=clear, >0=nube/sombra.
    Se evalúa SIEMPRE sobre S2.
    """
    kept, removed = [], []
    for p in pairs:
        s2 = p["s2"]
        cloud_mask = cloud_model(ckpt_path, s2)    # (H, W)
        clear_frac = (cloud_mask == 0).sum() / cloud_mask.size
        if clear_frac < clear_thresh:
            removed.append({**p, "reason": f"cloudy (clear_frac={clear_frac:.2f})"})
        else:
            kept.append(p)

    print(f"[Clouds] Keep {len(kept)} | Remove {len(removed)} (clear threshold={clear_thresh:.2f})")
    return kept, removed


def _s2_six_bands_norm(s2):
    """Extracts 6 bands and normalizes to [0,1] (channels-first: (C,H,W))."""
    x = s2[BAND_IDXS, :, :].astype(np.float32) / S2_SCALE
    return np.clip(x, 0.0, 1.0)


def compute_global_hist(pairs, nbins=NBINS, rng=HIST_RANGE):
    """
    Global histogram (density) over S2 in all remaining pairs.
    """
    hist = np.zeros(nbins, dtype=np.float64)
    bin_edges = None
    total = 0.0
    for p in pairs:
        s2 = _s2_six_bands_norm(p["s2"]).ravel()
        h, be = np.histogram(s2, bins=nbins, range=rng, density=True)
        hist += h
        bin_edges = be
        total += h.sum()
    if total <= 0:
        total = 1.0
    hist = hist / total  # renormalize
    return hist, bin_edges


def filter_by_kl(pairs, global_hist, nbins=NBINS, rng=HIST_RANGE, kld_thresh=KLD_DIVERGENCE_THRESHOLD):
    """
    Filters by distribution similarity (KL(p_i || p_global)) in S2.
    Preserves original order.
    """
    kept, removed = [], []
    for p in pairs:
        s2 = _s2_six_bands_norm(p["s2"]).ravel()
        s2_hist, _ = np.histogram(s2, bins=nbins, range=rng, density=True)
        norm = s2_hist.sum()
        if norm <= 0:
            removed.append({**p, "reason": "KL: empty"})
            continue
        s2_hist = s2_hist / norm
        kl = entropy(s2_hist, global_hist)

        if kl < kld_thresh:
            kept.append(p)
        else:
            removed.append({**p, "reason": f"KL={kl:.4f} ≥ {kld_thresh:.2f}"})
            print(f"Removing pair idx={p['idx']} date_s2={p['date_s2']} (KL={kl:.4f})")

    print(f"[KL] Keep {len(kept)} | Remove {len(removed)} (threshold={kld_thresh:.2f})")
    return kept, removed


def unpack_pairs(pairs):
    """
    Reconstructs original-style lists, preserving dates specific to S1 and S2.
    """
    s1_list = [{"date": p["date_s1"], "image": p["s1"]} for p in pairs]
    s2_list = [{"date": p["date_s2"], "image": p["s2"]} for p in pairs]
    return s1_list, s2_list


def _as_float(img: np.ndarray, scale_if_uint16: float | None = 10000.0) -> np.ndarray:
    """Convierte a float32. Si es uint16 y scale_if_uint16 no es None, divide por ese factor."""
    if img.dtype == np.uint16 and scale_if_uint16:
        return (img.astype(np.float32) / float(scale_if_uint16))
    return img.astype(np.float32)

def _robust_mean_std_per_band(img_bhw: np.ndarray,
                              clip_pct: float = 2.0,
                              mask: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    img_bhw: (B,H,W)
    Devuelve medias y std robustas por banda con recorte por percentiles.
    """
    B = img_bhw.shape[0]
    means = np.zeros(B, dtype=np.float64)
    stds  = np.zeros(B, dtype=np.float64)
    for b in range(B):
        x = img_bhw[b]
        m = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
        vals = x[m]
        if vals.size == 0:
            means[b] = 0.0
            stds[b]  = 1.0
            continue
        if clip_pct and clip_pct > 0:
            lo = np.nanpercentile(vals, clip_pct)
            hi = np.nanpercentile(vals, 100.0 - clip_pct)
            vals = vals[(vals >= lo) & (vals <= hi)]
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        means[b] = float(mu)
        stds[b]  = float(sd if sd > 1e-6 else 1.0)  # evita división por ~0
    return means, stds

def _make_blocks(h: int, w: int, blocks: int | Tuple[int,int]) -> Tuple[list, list]:
    """
    Devuelve listas de slices por filas y columnas que particionan la imagen en una grilla.
    Si blocks es int, arma una grilla ~cuadrada.
    """
    if isinstance(blocks, int):
        r = int(np.sqrt(blocks))
        c = int(np.ceil(blocks / r))
    else:
        r, c = blocks

    row_edges = np.linspace(0, h, r+1).astype(int)
    col_edges = np.linspace(0, w, c+1).astype(int)
    row_slices = [slice(row_edges[i], row_edges[i+1]) for i in range(r)]
    col_slices = [slice(col_edges[j], col_edges[j+1]) for j in range(c)]
    return row_slices, col_slices

def _expand_piecewise_constant_map(g_block: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    g_block: (rows, cols) → expande por repetición a (H,W).
    No interpola; bordes quedan "a escalón" (normalmente OK para corrección local).
    """
    rows, cols = g_block.shape
    r_sizes = np.diff(np.linspace(0, H, rows+1).astype(int))
    c_sizes = np.diff(np.linspace(0, W, cols+1).astype(int))
    out = np.empty((H, W), dtype=g_block.dtype)
    r0 = 0
    for i, rh in enumerate(r_sizes):
        c0 = 0
        for j, cw in enumerate(c_sizes):
            out[r0:r0+rh, c0:c0+cw] = g_block[i, j]
            c0 += cw
        r0 += rh
    return out

# -----------------------
# 1) Global regression
# -----------------------

def global_regression_series(
    series: List[Dict],
    *,
    scale_if_uint16: float | None = 10000.0,
    robust_clip_pct: float = 2.0,
    reference_mode: str = "median",      # "median" o "mean"
    exclude_bands: Optional[List[int]] = None,  # índices a NO ajustar (p.ej. [12] para la 13ª)
    clip_output: Optional[Tuple[float,float]] = None,  # p.ej. (0.0, 1.0)
) -> Tuple[List[Dict], Dict]:
    """
    Ajuste global afín por imagen/banda: y = a * x + b, igualando (μ,σ) a una referencia virtual.
    - reference_mode = "median" usa la mediana de medias/SD entre imágenes como referencia.
    Devuelve (serie_ajustada, modelo) con a/b por imagen y banda.
    """
    exclude_bands = set(exclude_bands or [])
    # Carga y apila a float
    imgs_f = [ _as_float(d["image"], scale_if_uint16=scale_if_uint16) for d in series ]
    stack = np.stack(imgs_f, axis=0)  # (N,B,H,W)
    N, B, H, W = stack.shape

    # Estadísticos por imagen/banda
    mu_i = np.zeros((N, B), dtype=np.float64)
    sd_i = np.zeros((N, B), dtype=np.float64)
    for i in range(N):
        mu_i[i], sd_i[i] = _robust_mean_std_per_band(stack[i], clip_pct=robust_clip_pct)

    # Referencia virtual por banda
    if reference_mode == "median":
        mu_ref = np.median(mu_i, axis=0)
        sd_ref = np.median(sd_i, axis=0)
    else:
        mu_ref = np.mean(mu_i, axis=0)
        sd_ref = np.mean(sd_i, axis=0)

    # a,b por imagen/banda
    a = np.zeros((N, B), dtype=np.float32)
    b = np.zeros((N, B), dtype=np.float32)
    for i in range(N):
        # bands a ajustar
        adj = [bidx for bidx in range(B) if bidx not in exclude_bands]
        if adj:
            a[i, adj] = (sd_ref[adj] / sd_i[i, adj]).astype(np.float32)
            b[i, adj] = (mu_ref[adj] - a[i, adj] * mu_i[i, adj]).astype(np.float32)
        # bandas excluidas: identidad
        for bidx in exclude_bands:
            a[i, bidx] = 1.0
            b[i, bidx] = 0.0

    # aplica
    out_series: List[Dict] = []
    for i, item in enumerate(series):
        img = stack[i]
        # broadcasting (B,1,1)
        y = (a[i, :, None, None] * img + b[i, :, None, None]).astype(np.float32)
        if clip_output is not None:
            y = np.clip(y, clip_output[0], clip_output[1])
        out_item = dict(item)  # copia superficial
        out_item["image"] = y
        out_series.append(out_item)

    model = {
        "mu_ref": mu_ref,
        "sd_ref": sd_ref,
        "mu_per_image": mu_i,
        "sd_per_image": sd_i,
        "a": a,
        "b": b,
        "exclude_bands": list(exclude_bands),
        "reference_mode": reference_mode,
        "robust_clip_pct": robust_clip_pct,
        "scale_if_uint16": scale_if_uint16,
    }
    return out_series, model

# -----------------------
# 2) Local block adjustment (lineal por bloques)
# -----------------------
def local_block_adjustment_series(
    series: List[Dict],
    *,
    blocks: int | Tuple[int,int] = 16,  # total o (filas,columnas)
    alpha: float = 1.0,                  # 0..1 empuje hacia la referencia local
    method: str = "linear",              # "linear" (ganancia por bloque)
    gain_clip: Tuple[float,float] = (0.5, 2.0),
    exclude_bands: Optional[List[int]] = None,
    valid_pixel_threshold: float = 0.0,  # fracción mínima válida por bloque (0=siempre)
) -> Tuple[List[Dict], Dict]:
    """
    Corrige localmente por bloques. Para cada bloque/banda:
      target_mu = (1-alpha)*mu_local + alpha*mu_ref  → y = g * x, con g = target_mu / mu_local.
    'method' por ahora implementa 'linear' (ganancia). 'gamma' no implementado aquí.
    """
    assert 0.0 <= alpha <= 1.0, "alpha debe estar en [0,1]"
    if method != "linear":
        raise NotImplementedError("Por simplicidad, aquí implemento solo method='linear'.")

    exclude_bands = set(exclude_bands or [])
    imgs = [item["image"].astype(np.float32, copy=False) for item in series]
    stack = np.stack(imgs, axis=0)  # (N,B,H,W)
    N, B, H, W = stack.shape
    row_slices, col_slices = _make_blocks(H, W, blocks)
    R, C = len(row_slices), len(col_slices)

    # Medias locales por imagen/banda/bloque
    mu_local = np.full((N, B, R, C), np.nan, dtype=np.float64)
    count_local = np.zeros((N, 1, R, C), dtype=np.int32)

    for i in range(N):
        for bi in range(B):
            for r, rs in enumerate(row_slices):
                for c, cs in enumerate(col_slices):
                    blk = stack[i, bi, rs, cs]
                    # válidos
                    m = np.isfinite(blk)
                    n_valid = int(m.sum())
                    count_local[i, 0, r, c] += n_valid
                    if n_valid == 0:
                        continue
                    mu_local[i, bi, r, c] = float(np.nanmean(blk[m]))

    # Umbral de válidos por bloque (sobre todas las bandas no QA)
    if valid_pixel_threshold > 0:
        area_block = np.array([ (rs.stop-rs.start)*(cs.stop-cs.start) for rs in row_slices for cs in col_slices ])
        area_block = area_block.reshape(R, C)
        valid_frac = (count_local.sum(axis=0)[0] / (area_block[None, :, :])).astype(np.float32)  # (R,C)
        valid_mask_rc = valid_frac >= float(valid_pixel_threshold)
    else:
        valid_mask_rc = np.ones((R, C), dtype=bool)

    # Referencia local por banda/bloque: media entre imágenes (ignorando NaN)
    mu_ref = np.nanmean(mu_local, axis=0)  # (B,R,C)

    # Aplicación: construye mapas de ganancia por imagen/banda/bloque
    gains_blocks = np.ones((N, B, R, C), dtype=np.float32)
    eps = 1e-6
    for i in range(N):
        for bi in range(B):
            if bi in exclude_bands:
                continue
            loc = mu_local[i, bi]            # (R,C)
            ref = mu_ref[bi]                 # (R,C)
            # mezcla hacia referencia
            target = (1.0 - alpha) * loc + alpha * ref
            # evita NaN / bloques sin válidos: deja g=1
            with np.errstate(divide='ignore', invalid='ignore'):
                g = np.where(np.isfinite(loc) & np.isfinite(target) & valid_mask_rc,
                             target / np.maximum(loc, eps), 1.0)
            g = np.clip(g.astype(np.float32), gain_clip[0], gain_clip[1])
            gains_blocks[i, bi] = g

    # Expande a mapas (B,H,W) por imagen y aplica
    out_series: List[Dict] = []
    for i, item in enumerate(series):
        img = stack[i]  # (B,H,W)
        out = np.empty_like(img, dtype=np.float32)
        for bi in range(B):
            if bi in exclude_bands:
                out[bi] = img[bi]  # sin cambios
                continue
            g_block = gains_blocks[i, bi]  # (R,C)
            g_full = _expand_piecewise_constant_map(g_block, H, W)  # (H,W)
            out[bi] = img[bi] * g_full
        out_item = dict(item)
        out_item["image"] = out
        out_series.append(out_item)

    info = {
        "mu_local": mu_local,     # (N,B,R,C)
        "mu_ref": mu_ref,         # (B,R,C)
        "gains_blocks": gains_blocks,  # (N,B,R,C)
        "blocks_grid": (R, C),
        "exclude_bands": list(exclude_bands),
        "alpha": alpha,
        "gain_clip": gain_clip,
        "method": method,
    }
    return out_series, info

def filtering_timeseries(
    s1_images,
    s2_images,
    normalize=False,                 # si True, aplica normalización global→local en S2
    normalize_params=None,           # ver llaves abajo
    local_blocks=16,               # si normalize=True, bloques para ajuste local

    # --- nubes y KLD ---
    cloud_ckpt_path=CLOUD_CKPT_PATH,
    clear_fraction_thresh=0.70,
    kld_thresh=0.20,
    nbins=200,
    hist_range=(0, 1),
):
    """
    Formatos esperados:
      - s1_images[i] = {'image': (B,H,W), 'date': str, 'cloud_fraction': ...}
      - s2_images[i] = {'image': (B,H,W), 'date': str, 'cloud_fraction': ...}

    normalize_params (opcionales):
      - scale_if_uint16 (def 10000.0)
      - robust_clip_pct (def 2.0)
      - reference_mode 'median'|'mean' (def 'median')
      - exclude_bands (def None)
      - clip_output (def None)
      - blocks (def 100)
      - alpha (def 1.0)
      - method (def 'linear')
      - gain_clip (def (0.5,2.0))
      - valid_pixel_threshold (def 0.0)
    """
    if normalize_params is None:
        normalize_params = {}

    # 1) Emparejar por índice
    pairs = align_pairs(s1_images, s2_images)
    print(f"[Align-index] Initial pairs: {len(pairs)}")

    # 2) NaNs/zeros
    pairs, removed_a = filter_zeros_nans(pairs)
    print(f"[Zeros/NaNs] Keep {len(pairs)} | Remove {len(removed_a)}")

    # 3) Filtro de nubes (S2)
    pairs, removed_b = filter_clouds(
        pairs,
        ckpt_path=cloud_ckpt_path,
        clear_thresh=clear_fraction_thresh
    )
    print(f"[Clouds] Keep {len(pairs)} | Remove {len(removed_b)} (clear threshold={clear_fraction_thresh:.2f})")

    # 4) KL vs histograma global (S2)  ← AHORA ANTES DE NORMALIZAR
    if pairs:
        global_hist, bin_edges = compute_global_hist(pairs, nbins, hist_range)
        pairs, removed_c = filter_by_kl(
            pairs,
            global_hist=global_hist,
            nbins=nbins,
            rng=hist_range,
            kld_thresh=kld_thresh
        )
        print(f"[KLD] Keep {len(pairs)} | Remove {len(removed_c)} (kld_thresh={kld_thresh:.2f})")
    else:
        bin_edges, removed_c = None, []

    # 5) Normalización global → local (solo S2 y solo si normalize=True)
    if normalize and pairs:
        # preparar lista para tus funciones: [{'image', 'date'}]
        s2_list = [{"image": p["s2"], "date": p["date_s2"]} for p in pairs]

        # GLOBAL
        s2_glob, _ = global_regression_series(
            s2_list
        )

        # LOCAL por bloques
        s2_corr, _ = local_block_adjustment_series(
            s2_glob, blocks=local_blocks
        )

        # escribir la imagen S2 corregida de vuelta en pairs
        for p, sc in zip(pairs, s2_corr):
            p["s2"] = sc["image"]

    # 6) Reconstruir salidas con el MISMO formato que las entradas
    s1_filtered, s2_filtered = [], []
    for p in pairs:
        i = p["idx"]
        s1_filtered.append({
            "image": p["s1"],
            "date": p["date_s1"],
            "cloud_fraction": s1_images[i].get("cloud_fraction", None),
        })
        s2_filtered.append({
            "image": p["s2"] / 10_000 if normalize is False else p["s2"],  # si no normaliza, escala a [0,1]
            "date": p["date_s2"],
            "cloud_fraction": s2_images[i].get("cloud_fraction", None),
        })

    logs = {
        "removed_zeros_nans": removed_a,
        "removed_clouds": removed_b,
        "removed_kl": removed_c,
        "bin_edges": bin_edges,
        "normalize_mode": bool(normalize),
        "cloud_ckpt_path": cloud_ckpt_path,
        "clear_fraction_thresh": clear_fraction_thresh,
        "kld_thresh": kld_thresh,
        "nbins": nbins,
        "hist_range": hist_range,
        "normalize_params_used": {
            k: normalize_params.get(k) for k in [
                "scale_if_uint16","robust_clip_pct","reference_mode","exclude_bands",
                "clip_output","blocks","alpha","method","gain_clip","valid_pixel_threshold"
            ]
        }
    }

    print(f"Original: S1={len(s1_images)} S2={len(s2_images)} | Final: {len(s2_filtered)} pairs")
    return s1_filtered, s2_filtered, logs

# =========================
# Semantic Segmentation Utilities
# =========================
def resize_array(arr: np.ndarray, shape_out: tuple, mode: str = 'bilinear') -> np.ndarray:
    """
    Resize a (H, W) or (C, H, W) array to new spatial size using interpolation.

    Args:
        arr (np.ndarray): Input array.
        shape_out (tuple): (H_out, W_out).
        mode (str): 'bilinear' for continuous data, 'nearest' for discrete.

    Returns:
        np.ndarray: Resized array.
    """
    is_single_band = arr.ndim == 2
    if is_single_band:
        arr = arr[np.newaxis, ...]  # → (1, H, W)

    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # → (1, C, H, W)
    resized = F.interpolate(tensor, size=shape_out, mode=mode, align_corners=False 
                            if mode == 'bilinear' else None)
    result = resized.squeeze(0).numpy()
    return result[0] if is_single_band else result

def save_features(s2_list: List[dict], 
                  s1_list: List[dict], 
                  profile: dict, 
                  shape: List[List[float]], 
                  model: Literal["multi_task", "one_task"] = "multi_task",
                  ) -> dict:
    
    """
    Save features extracted from S2 and S1 images. It includes the array list for the
    probability maps with cloud masks and the profile.    
    """

    # Initialize output arrays
    out_arrays = []
    for s1_arr, s2_arr in zip(s1_list, s2_list):
        s2_fullbands = np.array(s2_arr["image"])  # (13, H, W)
        s1_np = np.array(s1_arr["image"])  # (2, H, W)

        # Normalize S2 bands to 0-1
        s2_np = s2_fullbands[[3, 2, 1, 7, 10, 11]]  # RGB, NIR, SWIR1, SWIR2

        if model == "multi_task":
            # ============ MULTI-TASK MODEL ============
            CKPT_PATH = "../checkpoints/SRSegLit-50K_DLinkUnet++CSI_CBAM_best.ckpt"
            CFG_PATH = "../configs/br_config.yaml"
            
            with open(CFG_PATH, 'r') as f:
                cfg = yaml.safe_load(f)
                
            # Combine S2+S1 for input to model
            input_img = np.concatenate((s2_np, s1_np), axis=0)  # (C, H, W)
            mean = np.array(cfg['normalize']['mean']).reshape(-1, 1, 1)
            std = np.array(cfg['normalize']['std']).reshape(-1, 1, 1)
            input_img = (input_img - mean) / std
            input_tensor = torch.tensor(input_img).unsqueeze(0).float()

            # Load model
            ckpt = torch.load(CKPT_PATH, map_location='cpu')
            mt_model = SRSegLit(cfg)
            mt_model.load_state_dict(ckpt["state_dict"])
            mt_model.eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mt_model = mt_model.to(device).float()
            input_tensor = input_tensor.to(device).float()

            with torch.no_grad():
                build_logits, road_logits, *_ = mt_model(input_tensor)
                build_pred = torch.sigmoid(build_logits).squeeze(0).cpu().numpy()
                road_pred = torch.sigmoid(road_logits).squeeze(0).cpu().numpy()
                bg_pred = np.clip(1.0 - (build_pred[0] + road_pred[0]), 0, 1)

        elif model == "one_task":
            # ============ SEPARATE MODELS ============
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # --- ROAD MODEL ---
            ro_ckpt_path = "../checkpoints/dlinknet-bestmodel.ckpt"
            ro_cfg_path = "../configs/dlink_config.yaml"  

            with open(ro_cfg_path, 'r') as f:
                ro_cfg = yaml.safe_load(f)

            # Normalize S2 bands for road model
            s2_np_road = s2_fullbands[[3, 2, 1, 7, 10, 11]]   # RGB, NIR, SWIR1, SWIR2

            # Combine S2+S1 for road model
            input_img_road = np.concatenate((s2_np_road, s1_np), axis=0)  # (C, H, W)
            mean_road = np.array(ro_cfg['normalize']['mean']).reshape(-1, 1, 1)
            std_road = np.array(ro_cfg['normalize']['std']).reshape(-1, 1, 1)
            input_img_road = (input_img_road - mean_road) / std_road
            input_tensor_road = torch.tensor(input_img_road).unsqueeze(0).float()

            # Load and run road model
            ckpt_road = torch.load(ro_ckpt_path, map_location='cpu')
            road_model = RoadTrainer(ro_cfg)    
            road_model.load_state_dict(ckpt_road["state_dict"])
            road_model.eval()
            road_model = road_model.to(device).float()
            input_tensor_road = input_tensor_road.to(device).float()

            with torch.no_grad():
                road_logits = road_model(input_tensor_road)
                road_pred = torch.sigmoid(road_logits).squeeze(0).cpu().numpy()

            # --- BUILDING MODEL ---
            bu_ckpt_path = "../checkpoints/unetpp-bestmodel.ckpt"  # Ajusta la ruta
            bu_cfg_path = "../configs/unetpp_config.yaml"  # Ajusta la ruta

            with open(bu_cfg_path, 'r') as f:
                bu_cfg = yaml.safe_load(f)

            # Normalize S2 bands for building model
            s2_np_building = s2_fullbands[[3, 2, 1, 7, 10, 11]]   # RGB, NIR, SWIR1, SWIR2

            # Combine S2+S1 for building model
            input_img_building = np.concatenate((s2_np_building, s1_np), axis=0)  # (C, H, W)
            mean_building = np.array(bu_cfg['normalize']['mean']).reshape(-1, 1, 1)
            std_building = np.array(bu_cfg['normalize']['std']).reshape(-1, 1, 1)
            input_img_building = (input_img_building - mean_building) / std_building
            input_tensor_building = torch.tensor(input_img_building).unsqueeze(0).float()

            # Load and run building model
            ckpt_building = torch.load(bu_ckpt_path, map_location='cpu')
            building_model = BuildingTrainer(bu_cfg)    
            building_model.load_state_dict(ckpt_building["state_dict"])
            building_model.eval()
            building_model = building_model.to(device).float()
            input_tensor_building = input_tensor_building.to(device).float()

            with torch.no_grad():
                building_logits = building_model(input_tensor_building)
                build_pred = torch.sigmoid(building_logits).squeeze(0).cpu().numpy()

            # --- CALCULATE BACKGROUND ---
            bg_pred = np.clip(1.0 - (build_pred[0] + road_pred[0]), 0, 1)

        # ============ COMMON PROCESSING (UNIFIED) ============
        # At this point we have: build_pred, road_pred, bg_pred (same variables for both modes)

        # Upscale to 2.5m 
        H_pred, W_pred = build_pred.shape[1], build_pred.shape[2]
        upscale_shape = (H_pred, W_pred)  # target shape at 2.5m

        # Resize cloud mask
        cloud_mask = cloud_model(CLOUD_CKPT_PATH, s2_fullbands)
        cloud_mask_2_5 = resize_array(cloud_mask, upscale_shape, mode='nearest')

        # Stack all features at 2.5m
        features = np.stack([
            build_pred[0],
            road_pred[0],
            bg_pred,
            cloud_mask_2_5], axis=0)

        # --- Update profile to 2.5m ---
        new_transform = profile['transform'] * profile['transform'].scale(1/4, 1/4)
        new_height = features.shape[1]
        new_width = features.shape[2]

        out_profile = profile.copy()
        out_profile.update({
            "dtype": "float32",
            "count": features.shape[0],
            "nodata": np.nan,
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        # --- Reproject shape to image CRS ---
        shape_gdf = gpd.GeoSeries(shape).explode(index_parts=False)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", profile['crs'], always_xy=True).transform
        reprojected_shape = shape_gdf.apply(lambda geom: transform(transformer, geom))

        # --- Mask and store ---
        with rio.io.MemoryFile() as memfile:
            with memfile.open(**out_profile) as dataset:
                dataset.write(features)
            with memfile.open() as dataset:
                out_image, out_transform = rio_mask(dataset, reprojected_shape,
                                                    crop=True, all_touched=True,
                                                    nodata=np.nan)
                out_profile.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                out_arrays.append(out_image)

    features_dict = {
        'features': out_arrays,
        'profile': out_profile
    }
    return features_dict