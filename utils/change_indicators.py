# cdkliep_pipeline_smoothed.py
# -*- coding: utf-8 -*-
"""
Pipeline con integración de smoothing temporal para decisiones from→to.
- Se añaden overrides de fracciones suavizadas en compute_change_metrics (opcional).
- Se añade add_smoothed_labels(...) para re-etiquetar dominant_label con EMA.
- scan_temporal_consistencia puede usar etiquetas suavizadas sin tocar el resto.
"""

import numpy as np
import pandas as pd
import torch
import yaml

from typing import List, Dict, Any, Optional, Literal, Sequence

# ==== (Modelos originales; mantener imports si tu entorno los tiene) ====
from models.rrdbnet import RRDBNet
from trainers.unetpp_trainer import BuildingTrainer
from trainers.dlinknet_trainer import RoadTrainer
from trainers.pl_trainer import SRSegLit

from utils.pykliep import DensityRatioEstimator
from statsmodels.tsa.seasonal import seasonal_decompose


# =====================
# Utilidades base
# =====================
ARROW = "→"


def flatten_clean_pixels_masked(img, cloud_band_index=1):
    n_bands = img.shape[0]
    bands_idx = np.delete(np.arange(n_bands), cloud_band_index)
    arr = img[bands_idx].reshape(len(bands_idx), -1).T  # (hxw, c)
    arr = arr[~np.isnan(arr).any(axis=1)]
    cloud = img[cloud_band_index].ravel()
    valid_mask = ~np.isnan(cloud)
    cloud = cloud[valid_mask]
    pixels_cloud = (cloud != 0)
    percent_cloud = (100. * np.sum(pixels_cloud) / cloud.size)
    return arr, percent_cloud


def kliep_score(data, cloud_band_index=1, cloud_thresh=50, kliep_params=None, max_skip=5):
    if kliep_params is None:
        kliep_params = dict(max_iter=1000, sigmas=[0.5, 1, 2], verbose=0)
    n_times = data.shape[0]
    scores, t1s, t2s = [], [], []
    for t in range(n_times - 1):
        X_train, pct_cloud1 = flatten_clean_pixels_masked(data[t], cloud_band_index)
        if pct_cloud1 > cloud_thresh:
            continue  # Image t is too cloudy
        found = False
        for dt in range(1, min(max_skip, n_times - t)):
            X_test, pct_cloud2 = flatten_clean_pixels_masked(data[t + dt], cloud_band_index)
            if pct_cloud2 > cloud_thresh:
                continue  # Skip cloudy candidate
            # Both images pass the cloud threshold: proceed!
            try:
                kliep = DensityRatioEstimator(**kliep_params)
                kliep.fit(X_train, X_test)
                g_pred = kliep.predict(X_test)
                kl_score = np.mean(np.log(np.clip(g_pred, 1e-8, None)))
                scores.append(kl_score)
                t1s.append(t)
                t2s.append(t + dt)
            except Exception as e:
                print(f"KLIEP error t={t} t2={t+dt}: {e}")
            found = True
            break

        # Fallback: If no suitable image is found, calculate score with the image itself
        if not found:
            try:
                kliep = DensityRatioEstimator(**kliep_params)
                kliep.fit(X_train, X_train)
                g_pred = kliep.predict(X_train)
                kl_score = np.mean(np.log(np.clip(g_pred, 1e-8, None)))
                scores.append(kl_score)
                t1s.append(t)
                t2s.append(t)  # Pair with itself
            except Exception as e:
                print(f"KLIEP fallback error t={t}: {e}")

    return np.array(scores), np.array(t1s), np.array(t2s)


def _valid_mask(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Válidos = píxeles donde todos los canales son finitos y la suma por canal > 0
    en A1 y A2. Espera tensores (C,H,W).
    """
    assert a1.shape == a2.shape and a1.ndim == 3, "A1 y A2 deben tener shape (C,H,W)"
    v1 = np.all(np.isfinite(a1), axis=0) & (np.nansum(a1, axis=0) >= 0)
    v2 = np.all(np.isfinite(a2), axis=0) & (np.nansum(a2, axis=0) >= 0)
    return v1 & v2


def _select_dominant_idx(fracs: np.ndarray, bg_id: Optional[int], bg_full_tol: float) -> int:
    """
    Devuelve el índice de la clase dominante basándose en CONTEO de píxeles
    (y, por tanto, en el mayor porcentaje real en el área).
    Reglas:
      1) Si background (bg_id) cubre >= (1 - bg_full_tol), devuelve bg_id.
      2) Si no, elige la clase no-background con mayor CONTEO (empates: mayor fracción, luego menor índice).
      3) Si todo es inválido/no-finito, cae a bg_id (si existe) o -1.
    """
    fr = np.asarray(fracs, dtype=float)
    C = fr.size
    def _valid_bg(bg): return (bg is not None) and (0 <= bg < C)

    fr_sane = np.where(np.isfinite(fr) & (fr >= 0), fr, 0.0)
    s = float(fr_sane.sum())

    # ¿fracciones o conteos?
    if s <= 1.5:
        if s > 0:
            fr_norm = fr_sane / s
        else:
            fr_norm = fr_sane
        N = 100000
        cnt = np.rint(fr_norm * N).astype(int)
        diff = N - int(cnt.sum())
        if diff != 0:
            order = np.argsort(-fr_norm)  # prioriza clases con mayor fracción
            for k in order:
                if diff == 0:
                    break
                if diff > 0:
                    cnt[k] += 1; diff -= 1
                else:
                    if cnt[k] > 0:
                        cnt[k] -= 1; diff += 1
        fr_use = fr_norm
    else:
        cnt = np.rint(fr_sane).astype(int)
        total = cnt.sum()
        fr_use = (cnt / total) if total > 0 else np.zeros_like(cnt, dtype=float)

    if _valid_bg(bg_id):
        bg = int(bg_id)
        if np.isfinite(fr_use[bg]) and (fr_use[bg] >= 1.0 - bg_full_tol):
            return bg

    if _valid_bg(bg_id):
        cand = [k for k in range(C) if k != int(bg_id) and (cnt[k] > 0)]
        if len(cand) == 0:
            return int(bg_id)
    else:
        cand = [k for k in range(C) if (cnt[k] > 0)]
    if len(cand) == 0:
        return int(bg_id) if _valid_bg(bg_id) else -1

    cand_cnt = cnt[cand]
    if not np.isfinite(cand_cnt).any():
        return int(bg_id) if _valid_bg(bg_id) else -1
    arg = int(np.nanargmax(cand_cnt))
    best = [cand[arg]]

    ties = np.where(cand_cnt == cand_cnt[arg])[0]
    if ties.size > 1:
        tie_idxs = [cand[i] for i in ties]
        fr_ties = fr_use[tie_idxs]
        arg2 = int(np.nanargmax(fr_ties))
        best = [tie_idxs[arg2]]
        eq = np.where(fr_ties == fr_ties[arg2])[0]
        if eq.size > 1:
            best = [min([tie_idxs[i] for i in eq])]

    return int(best[0])


# =====================
# Residuales
# =====================
def generate_residuals(
    features: dict,
    t1s: np.ndarray,
    s2_f: List[dict],
    period: int = 7,
    model: str = "additive",
    fill_method: str = "interpolate",
    # nuevo comportamiento
    dominant_strategy: str = "auto",
    dominant_only: bool = True,
    low_q: float = 0.001,
    high_q: float = 0.999,
    min_std: float = 1e-3,
):
    """
    Calcula residuales de series (mean por canal) en los instantes t1.
    Devuelve: DataFrame con columnas
    ["date","ts1","residual_building","residual_roads","building_outlier","road_outlier"].
    """
    t1s = np.asarray(t1s, dtype=int)

    build_means, road_means, dates = [], [], []
    for t in t1s:
        feat = np.array(features["features"])[t]  # shape (C,H,W)
        build_means.append(np.nanmean(feat[0]))
        road_means.append(np.nanmean(feat[1]))
        dates.append(s2_f[t]["date"])

    df_series = pd.DataFrame({
        "date": dates,
        "build": np.asarray(build_means, dtype=float),
        "road":  np.asarray(road_means,  dtype=float),
        "ts1":   t1s,
    }).sort_values("date").reset_index(drop=True).set_index("date")

    if fill_method == "interpolate":
        df_series[["build","road"]] = df_series[["build","road"]].interpolate(limit_direction="both")
    elif fill_method in ("ffill","bfill"):
        df_series[["build","road"]] = getattr(df_series[["build","road"]], fill_method)()

    n = len(df_series)
    if n < max(2*period, period+1):
        return pd.DataFrame({
            "date": df_series.index.values,
            "ts1": df_series["ts1"].values.astype(int),
            "residual_building": np.full(n, np.nan),
            "residual_roads":    np.full(n, np.nan),
            "building_outlier":  np.full(n, False),
            "road_outlier":      np.full(n, False),
        })

    b_result = seasonal_decompose(df_series["build"], model=model, period=period, extrapolate_trend='freq')
    r_result = seasonal_decompose(df_series["road"],  model=model, period=period, extrapolate_trend='freq')

    b_resid = np.asarray(b_result.resid, dtype=float)
    r_resid = np.asarray(r_result.resid, dtype=float)

    x_idx = np.arange(n, dtype=float)
    def slope(y):
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return 0.0
        m, _ = np.polyfit(x_idx[mask], y[mask], 1)
        return m

    b_slope = slope(df_series["build"].to_numpy())
    r_slope = slope(df_series["road"].to_numpy())
    b_mean  = float(np.nanmean(df_series["build"]))
    r_mean  = float(np.nanmean(df_series["road"]))

    def pick_dominant(strategy="auto"):
        eps = 1e-12
        if strategy == "slope":
            if (b_slope - r_slope) > eps:
                return "building"
            elif (r_slope - b_slope) > eps:
                return "road"
            else:
                return "building" if (b_mean >= r_mean) else "road"
        elif strategy == "mean":
            return "building" if (b_mean >= r_mean) else "road"
        elif strategy == "auto":
            if max(b_slope, r_slope) > eps:
                return "building" if (b_slope >= r_slope) else "road"
            else:
                return "building" if (b_mean >= r_mean) else "road"
        else:  # "none"
            return "none"

    dominant = pick_dominant(dominant_strategy)

    def quantile_outlier_mask(x, low=0.05, high=0.95, tail="both", min_std=1e-3):
        x = np.asarray(x, dtype=float)
        if np.nanstd(x) < min_std:
            return np.zeros_like(x, dtype=bool)
        lo = np.nanquantile(x, low)
        hi = np.nanquantile(x, high)
        if tail == "both":
            return (x < lo) | (x > hi)
        elif tail == "positive":
            return x > hi
        elif tail == "negative":
            return x < lo
        else:
            raise ValueError("tail debe ser 'both', 'positive' o 'negative'.")

    tail_build = "positive" if (dominant == "building" and b_slope > 0) else "both"
    tail_road  = "positive" if (dominant == "road"     and r_slope > 0) else "both"

    b_mask = quantile_outlier_mask(b_resid, low=low_q, high=high_q, tail=tail_build, min_std=min_std)
    r_mask = quantile_outlier_mask(r_resid, low=low_q, high=high_q, tail=tail_road,  min_std=min_std)

    if dominant_only:
        if dominant == "building":
            r_mask = np.zeros_like(r_mask, dtype=bool)
        elif dominant == "road":
            b_mask = np.zeros_like(b_mask, dtype=bool)

    out = pd.DataFrame({
        "date": df_series.index.values,
        "ts1": df_series["ts1"].values.astype(int),
        "residual_building": b_resid,
        "residual_roads":    r_resid,
        "building_outlier":  b_mask,
        "road_outlier":      r_mask,
    })
    out["date"] = pd.to_datetime(out["date"])
    return out


# =====================
# Predicciones (igual)
# =====================
def get_predictions(s1_f, s2_f, method="multi_task", device=None):
    """
    Devuelve out_arrays con diccionarios:
      - 'date'     : str
      - 'rgb_pred' : np.ndarray (3,H,W) [build, road, bg]
      - 's2'       : np.ndarray (13,H,W)
      - 'sr'       : np.ndarray (3,4H,4W)  (siempre calculado)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_half = (device.type == 'cuda')

    out_arrays = []

    # --- SR común a ambos métodos (se carga una vez) ---
    logdir_hr = "../checkpoints/checkpoint.tar"
    net_hr = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
    net_hr.load_state_dict(torch.load(logdir_hr, map_location='cpu')['net_g_ema'])
    net_hr.eval().to(device)
    if use_half:
        net_hr.half()
    for p in net_hr.parameters():
        p.requires_grad = False

    _mt_cfg = _mt_model = None
    _ro_cfg = _ro_model = None
    _bu_cfg = _bu_model = None

    for i, (s1_arr, s2_arr) in enumerate(zip(s1_f, s2_f)):
        s2_full = np.asarray(s2_arr["image"])  # (13,H,W)
        s1_np   = np.asarray(s1_arr["image"])  # (2,H,W)
        s2_sel  = s2_full[[3, 2, 1, 7, 10, 11]]
        s2_rgb  = s2_sel[[0, 1, 2]]
        sr_in   = torch.from_numpy(s2_rgb).unsqueeze(0).to(device)
        sr_in   = sr_in.half() if use_half else sr_in.float()
        with torch.no_grad():
            sr_rgb_t = net_hr(sr_in)  # (1,3,4H,4W)
        sr_rgb = sr_rgb_t.squeeze(0).float().cpu().numpy().clip(0, 1)

        if method == "multi_task":
            if _mt_model is None:
                br_ckpt_path = "../checkpoints/SRSegLit-50K_DLinkUnet++CSI_CBAM_best.ckpt"
                cfg_path     = "../configs/br_config.yaml"
                with open(cfg_path, 'r') as f:
                    _mt_cfg = yaml.safe_load(f)
                _mt_model = SRSegLit(_mt_cfg)
                ckpt = torch.load(br_ckpt_path, map_location='cpu')
                _mt_model.load_state_dict(ckpt["state_dict"])
                _mt_model.eval().to(device).float()
                _mt_mean = np.array(_mt_cfg['normalize']['mean']).reshape(-1, 1, 1)
                _mt_std  = np.array(_mt_cfg['normalize']['std']).reshape(-1, 1, 1)

            x = np.concatenate([s2_sel, s1_np], axis=0)
            x = (x - _mt_mean) / _mt_std
            x = torch.from_numpy(x).unsqueeze(0).float().to(device)
            with torch.no_grad():
                build_logits, road_logits, *_ = _mt_model(x)
                build_pred = torch.sigmoid(build_logits).squeeze(0).cpu().numpy()
                road_pred  = torch.sigmoid(road_logits ).squeeze(0).cpu().numpy()

        elif method == "one_task":
            if _ro_model is None:
                ro_ckpt_path = "../checkpoints/dlinknet-bestmodel.ckpt"
                ro_cfg_path  = "../configs/dlink_config.yaml"
                with open(ro_cfg_path, 'r') as f:
                    _ro_cfg = yaml.safe_load(f)
                _ro_model = RoadTrainer(_ro_cfg)
                ckpt_ro = torch.load(ro_ckpt_path, map_location='cpu')
                _ro_model.load_state_dict(ckpt_ro["state_dict"])
                _ro_model.eval().to(device).float()
                _ro_mean = np.array(_ro_cfg['normalize']['mean']).reshape(-1, 1, 1)
                _ro_std  = np.array(_ro_cfg['normalize']['std']).reshape(-1, 1, 1)

            if _bu_model is None:
                bu_ckpt_path = "../checkpoints/unetpp-bestmodel.ckpt"
                bu_cfg_path  = "../configs/unetpp_config.yaml"
                with open(bu_cfg_path, 'r') as f:
                    _bu_cfg = yaml.safe_load(f)
                _bu_model = BuildingTrainer(_bu_cfg)
                ckpt_bu = torch.load(bu_ckpt_path, map_location='cpu')
                _bu_model.load_state_dict(ckpt_bu["state_dict"])
                _bu_model.eval().to(device).float()
                _bu_mean = np.array(_bu_cfg['normalize']['mean']).reshape(-1, 1, 1)
                _bu_std  = np.array(_bu_cfg['normalize']['std']).reshape(-1, 1, 1)

            x_ro = np.concatenate([s2_sel, s1_np], axis=0)
            x_ro = (x_ro - _ro_mean) / _ro_std
            x_ro = torch.from_numpy(x_ro).unsqueeze(0).float().to(device)
            with torch.no_grad():
                road_logits = _ro_model(x_ro)
                road_pred = torch.sigmoid(road_logits).squeeze(0).cpu().numpy()

            x_bu = np.concatenate([s2_sel, s1_np], axis=0)
            x_bu = (x_bu - _bu_mean) / _bu_std
            x_bu = torch.from_numpy(x_bu).unsqueeze(0).float().to(device)
            with torch.no_grad():
                build_logits = _bu_model(x_bu)
                build_pred = torch.sigmoid(build_logits).squeeze(0).cpu().numpy()

        else:
            raise ValueError(f"Método no soportado: {method}. Use 'multi_task' o 'one_task'.")

        b = build_pred[0] if build_pred.ndim == 3 else build_pred
        r = road_pred[0]  if road_pred.ndim  == 3 else road_pred
        bg = np.clip(1.0 - (b + r), 0.0, 1.0)
        rgb_pred = np.stack([b, r, bg], axis=0)  # (3,H,W)

        out_arrays.append({
            "date": s2_arr["date"],
            "image": rgb_pred,
            "s2": s2_full,
            "sr": sr_rgb
        })
        print(f"[{method}] {i+1}/{len(s2_f)} -> {s2_arr['date']}")

    return out_arrays


# =====================
# Smoothing temporal (AOI) y Overrides
# =====================
def _ema_vec(P: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    EMA por filas (tiempo) en matriz P[T, C]. Renormaliza cada fila para suma=1.
    """
    P = np.asarray(P, float)
    S = np.zeros_like(P)
    if len(P) == 0:
        return S
    S[0] = P[0]
    for t in range(1, len(P)):
        S[t] = alpha * S[t - 1] + (1 - alpha) * P[t]
        s = S[t].sum()
        S[t] = S[t] / (s if s > 1e-9 else 1.0)
    return S


def add_smoothed_labels(
    df_metrics: pd.DataFrame,
    alpha: float = 0.6,
    names: Sequence[str] = ("building", "road", "background"),
    label_col_out: str = "dominant_label_smooth",
) -> pd.DataFrame:
    """
    Reconstruye fracciones por fecha a partir de 'class_fracs_t1'/'class_fracs_t2'
    presentes en df_metrics, aplica EMA y re-etiqueta 'from → to' usando las fracciones
    suavizadas para cada par (t1, t2). Devuelve una copia de df con nueva columna.
    """
    dfw = df_metrics.reset_index(drop=True).copy()

    # Asegurar columnas dict presentes (si no, queda vacío y no se suaviza)
    if not (("class_fracs_t1" in dfw.columns) and ("class_fracs_t2" in dfw.columns)):
        # No hay fracciones para reconstruir → devolvemos tal cual
        dfw[label_col_out] = dfw.get("dominant_label", None)
        return dfw

    # Construir tabla de fracciones por fecha
    def _collect(col_name: str, time_col: str) -> Dict[int, np.ndarray]:
        out = {}
        for i, row in dfw.iterrows():
            d = row.get(col_name, None)
            if isinstance(d, dict) and time_col in row:
                t = int(row[time_col])
                vec = np.array([d.get(n, np.nan) for n in names], float)
                if t in out:
                    out[t] = np.nanmean(np.vstack([out[t], vec]), axis=0)
                else:
                    out[t] = vec
        return out

    map_t1 = _collect("class_fracs_t1", "t1")
    map_t2 = _collect("class_fracs_t2", "t2")
    all_t = sorted(set(map_t1.keys()) | set(map_t2.keys()))
    if len(all_t) == 0:
        dfw[label_col_out] = dfw.get("dominant_label", None)
        return dfw

    P = np.zeros((len(all_t), len(names)), float)
    for i, t in enumerate(all_t):
        v_list = []
        if t in map_t1:
            v_list.append(map_t1[t])
        if t in map_t2:
            v_list.append(map_t2[t])
        if len(v_list) == 0:
            continue
        v = np.nanmean(np.vstack(v_list), axis=0)
        # Relleno y normalización suave
        v = np.where(np.isfinite(v), v, 0.0)
        s = v.sum()
        P[i] = v / (s if s > 1e-9 else 1.0)

    # Suavizar por tiempo
    S = _ema_vec(P, alpha=alpha)

    # Índices inversos fecha→fila
    t2idx = {t: i for i, t in enumerate(all_t)}

    # Función auxiliar para decidir etiqueta usando fracs suavizadas
    def _decide_label_from_fracs(f1, f2, names=names, bg_full_tol=0.10, br_bias_eps=0.02, br_require_ratio_eps=0.05):
        names = list(names)
        def _idx(n): return names.index(n) if n in names else None
        b, r, bg = _idx("building"), _idx("road"), _idx("background")

        def _dom_tolerant(f):
            if bg is None:
                return int(np.argmax(f))
            # si BG domina casi todo, devolver BG; si no, el mayor no-bg
            if f[bg] >= 1.0 - bg_full_tol:
                return bg
            f2 = f.copy()
            f2[bg] = -1.0
            return int(np.argmax(f2))

        from_k = _dom_tolerant(f1)
        cand = [k for k in range(len(f1)) if k != from_k]

        def _prefer_BR(cands):
            if b is None or r is None: return None
            if (b in cands) and (r in cands):
                cb, cr = float(f2[b]), float(f2[r])
                if cb <= 0 and cr > 0: return r
                if cr <= 0 and cb > 0: return b
                return b if cb >= (1.0 - br_bias_eps) * cr else r
            return None

        def _enforce_strict(k):
            if k == r and b is not None:
                if f2[r] <= f2[b] * (1.0 + br_require_ratio_eps):
                    return b
            return k

        # prefer change_to por Δfracción positiva
        dfv = (f2 - f1).copy()
        dfv[dfv <= 0] = -np.inf
        k = int(np.nanargmax(dfv)) if np.isfinite(dfv).any() else -1
        if k not in cand or not np.isfinite(dfv[k]):
            k = _enforce_strict(_prefer_BR(cand) if _prefer_BR(cand) is not None else int(np.argmax(f2)))
        else:
            pref = _prefer_BR([k, b, r])
            k = _enforce_strict(pref if pref is not None else k)

        return names[from_k] + " \u2192 " + names[k]

    # Re-etiquetar cada fila
    new_labels = []
    for _, row in dfw.iterrows():
        t1 = int(row["t1"]); t2 = int(row["t2"])
        if (t1 not in t2idx) or (t2 not in t2idx):
            new_labels.append(row.get("dominant_label", None))
            continue
        f1 = S[t2idx[t1]]
        f2 = S[t2idx[t2]]
        new_labels.append(_decide_label_from_fracs(f1, f2))

    dfw[label_col_out] = new_labels
    return dfw


# =====================
# Métricas de cambio (con overrides opcionales)
# =====================
def compute_change_metrics(
    A1: np.ndarray,
    A2: np.ndarray,
    class_names: Sequence[str],
    *,
    bg_full_tol: float = 0.05,        # SOLO si background ocupa ≥ (1 - bg_full_tol) se puede elegir como dominante
    delta_frac: float = 0.01,         # incremento mínimo relativo (p.ej., 1% de n_valid)
    margin_frac: float = 0.005,       # margen mínimo relativo sobre la clase previa (p.ej., 0.5%)
    min_abs_delta_px: int = 5,        # incremento mínimo absoluto
    min_abs_margin_px: int = 3,       # margen mínimo absoluto
    presence_frac: float = 0.02,      # presencia mínima cuando el FROM es background (p.ej., 2% de n_valid)
    min_abs_presence_px: int = 10     # presencia mínima absoluta para FROM=background
) -> Dict[str, object]:
    """
    Determina la transición dominante 'from → to' usando SOLO conteos (argmax) y umbrales simples.
    Reglas (modificadas):
      - background queda SUPRIMIDO como dominante salvo que su fracción ≥ (1 - bg_full_tol)
      - from_k = dominante de T1 tras supresión de background
      - to*    = dominante de T2 tras supresión de background
      - Si from=background y to*≠background: se exige crecimiento real (Δ≥delta_px) y presencia mínima (counts2[to*]≥presence_px)
      - En otros casos (incluye pasar a background): se exige Δ≥delta_px y to* > from + margin_px
      - Si no se cumple, no hay cambio (to=from)
    """
    A1 = np.asarray(A1, float)
    A2 = np.asarray(A2, float)
    C, H, W = A1.shape
    names = np.asarray(class_names, dtype=object)

    # 1) Válidos
    valid = np.isfinite(A1).all(axis=0) & np.isfinite(A2).all(axis=0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "valid_mask": valid,
            "dominant_from": "unknown",
            "dominant_to": "unknown",
            "dominant_label": "unknown \u2192 unknown",
            "n_valid": 0
        }

    v = valid.ravel()
    flat1 = A1.reshape(C, -1)[:, v]
    flat2 = A2.reshape(C, -1)[:, v]

    # 2) Etiquetas por píxel y conteos
    y1 = np.argmax(flat1, axis=0)
    y2 = np.argmax(flat2, axis=0)
    counts1 = np.array([(y1 == k).sum() for k in range(C)], dtype=int)
    counts2 = np.array([(y2 == k).sum() for k in range(C)], dtype=int)
    f1 = counts1 / n_valid
    f2 = counts2 / n_valid

    # 3) Índice de background (si existe)
    bg_idx = int(np.where(names == "background")[0][0]) if np.any(names == "background") else None

    # 4) Dominante en T1 (from_k) con SUPRESIÓN de background salvo que sea casi total
    if bg_idx is not None and np.isfinite(f1[bg_idx]) and (f1[bg_idx] < 1.0 - bg_full_tol):
        # Prohibir background como candidato dominante en T1
        counts1_nb = counts1.copy()
        counts1_nb[bg_idx] = -1  # lo excluye del argmax
        from_k = int(np.argmax(counts1_nb))
    else:
        # Sin background o background casi total
        from_k = int(np.argmax(counts1))

    # 5) Candidato dominante en T2 (to_star) con la MISMA supresión de background
    if bg_idx is not None and np.isfinite(f2[bg_idx]) and (f2[bg_idx] < 1.0 - bg_full_tol):
        counts2_nb = counts2.copy()
        counts2_nb[bg_idx] = -1
        to_star = int(np.argmax(counts2_nb))
    else:
        to_star = int(np.argmax(counts2))

    # 6) Umbrales (relativos → absolutos)
    delta_px    = max(int(np.ceil(delta_frac    * n_valid)), int(min_abs_delta_px))
    margin_px   = max(int(np.ceil(margin_frac   * n_valid)), int(min_abs_margin_px))
    presence_px = max(int(np.ceil(presence_frac * n_valid)), int(min_abs_presence_px))

    # 7) Decisión (asimétrica para FROM=background)
    if to_star != from_k:
        if (bg_idx is not None) and (from_k == bg_idx) and (to_star != bg_idx):
            # background → (building|road): aparición suficiente + crecimiento real
            ok = ((counts2[to_star] - counts1[to_star]) >= delta_px) and (counts2[to_star] >= presence_px)
        else:
            # casos normales (incluye (building|road) → background)
            ok = ((counts2[to_star] - counts1[to_star]) >= delta_px) and (counts2[to_star] > counts2[from_k] + margin_px)
        to_k = to_star if ok else from_k
    else:
        to_k = from_k

    # 8) Derivados ligeros (diagnóstico)
    d = A2 - A1
    mag_map = np.full((H, W), np.nan, float)
    mag = np.sqrt(np.sum(d.reshape(C, -1)[:, v]**2, axis=0))
    mag_map[valid] = mag
    rate_change_global = float(np.mean(y1 != y2))
    changed_count = int(np.sum(y1 != y2))

    # Tasas y deltas por clase
    delta_counts = counts2 - counts1
    change_rates = {str(names[k]): float(delta_counts[k]) / float(n_valid) for k in range(C)}

    return {
        "valid_mask": valid,
        "magnitude_map": mag_map,
        "rate_change_global": rate_change_global,
        "changed_count": changed_count,
        "n_valid": n_valid,
        "class_counts_t1": {str(names[k]): int(counts1[k]) for k in range(C)},
        "class_counts_t2": {str(names[k]): int(counts2[k]) for k in range(C)},
        "class_fracs_t1": {str(names[k]): float(f1[k]) for k in range(C)},
        "class_fracs_t2": {str(names[k]): float(f2[k]) for k in range(C)},
        "dominant_from": str(names[from_k]),
        "dominant_to": str(names[to_k]),
        "dominant_label": f"{names[from_k]} \u2192 {names[to_k]}",
        "delta_px_used": int(delta_px),
        "margin_px_used": int(margin_px),
        "presence_px_used": int(presence_px),
        "change_rates": change_rates,
        "delta_counts": {str(names[k]): int(delta_counts[k]) for k in range(C)}
    }



# ==== DataFrame final con métricas de cambio ====
def build_results_dataframe_from_scalar(
    results,
    area_id,
    series_id,
    residuals_df: pd.DataFrame = None,
    round_decimals: int = 4
) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        cva_mag_mean = r.get("cva_mag_mean", None)
        if cva_mag_mean is None or not np.isfinite(cva_mag_mean):
            mag_map = r.get("magnitude", None)
            cva_mag_mean = float(np.nanmean(np.asarray(mag_map))) if mag_map is not None else np.nan

        def _r(x):
            try:
                x = float(x)
                return round(x, round_decimals) if np.isfinite(x) else np.nan
            except Exception:
                return np.nan

        rows.append({
            "nro": i,
            "series_id": series_id,
            "area_id": area_id,
            "rank": i,
            "t1": int(r["t1"]),
            "t2": int(r["t2"]),
            "date_change": r.get("date_change", None),
            "kliep_abs": _r(abs(float(r["score"]))) if ("score" in r) else np.nan,
            "cva_mag_mean": _r(cva_mag_mean),
            "dominant_label": r.get("dominant_label"),
            "rate_bu_change": _r(r.get("rate_bu_change", np.nan)),
            "rate_ro_change": _r(r.get("rate_ro_change", np.nan)),
            "rate_bk_change": _r(r.get("rate_bk_change", np.nan)),
            # *** Guardar fracciones para poder suavizar luego ***
            "class_fracs_t1": r.get("class_fracs_t1", None),
            "class_fracs_t2": r.get("class_fracs_t2", None),
        })

    df = pd.DataFrame(rows)

    if residuals_df is not None and not residuals_df.empty:
        residuals_df = residuals_df.copy()
        if "ts1" in residuals_df.columns:
            residuals_df = residuals_df.rename(columns={"ts1": "t1"})
        residuals_df["t1"] = residuals_df["t1"].astype(int)

        df = df.merge(
            residuals_df[["t1", "residual_building", "residual_roads"]],
            on="t1", how="left"
        )

    df = df.sort_values(by=["kliep_abs", "cva_mag_mean"], ascending=False, na_position="last").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    cols = [
        "nro","series_id","area_id","rank","t1","t2","date_change",
        "kliep_abs","cva_mag_mean","dominant_label",
        "rate_bu_change","rate_ro_change","rate_bk_change",
        "residual_building","residual_roads",
        "class_fracs_t1","class_fracs_t2"
    ]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    return df


# ─────────────────────────────────────────────
# Utils para ventanas
# ─────────────────────────────────────────────
def _parse_pair_label(lbl: str):
    """Parsea 'a → b' devolviendo (a,b); si no encaja, None."""
    if not isinstance(lbl, str) or ARROW not in lbl:
        return None
    a, b = [p.strip() for p in lbl.split(ARROW, 1)]
    return a, b


def _pick_representatives_by_t1(g: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve 1 fila representativa por t1 dentro de un grupo g.
    Prioridad: (t2 == t1+1) > |t2 - t1| mínimo > rank más bajo.
    Asume que g ya viene con columna '_pos' (posición en el DF reseteado).
    """
    g = g.copy()
    if "rank" not in g.columns:
        g["rank"] = np.arange(len(g), dtype=int) + 1
    g["delta"] = (g["t2"] - g["t1"]).abs()
    g["prefer_step"] = (g["t2"] == g["t1"] + 1)  # booleano
    sort_cols = ["t1", "prefer_step", "delta", "rank"]
    g = g.sort_values(sort_cols, ascending=[True, False, True, True], kind="mergesort")
    reps = (
        g.groupby("t1", as_index=False)
         .first()[["_pos", "t1", "t2", "rank"]]  # representante
    )
    return reps


def build_windows_by_t1_consecutive(
    df: pd.DataFrame,
    window_pairs: int = 7,
    step_pairs: int = 1,
    groupby_cols: Optional[List[str]] = None,
) -> List[List[int]]:
    """
    Ventanas con t1 estrictamente consecutivos: [b, b+1, ..., b+W-1].
    Usa 1 representante por t1. Devuelve índices POSICIONALES (df.reset_index).
    """
    assert {"t1", "t2"}.issubset(df.columns), "Faltan columnas 't1' y/o 't2'."
    dfw = df.copy()
    if "_pos" not in dfw.columns:
        dfw = dfw.reset_index(drop=True)
        dfw["_pos"] = np.arange(len(dfw), dtype=int)

    default_groups = [c for c in ["series_id", "area_id"] if c in dfw.columns]
    groupby_cols = groupby_cols or (default_groups if default_groups else [None])

    out: List[List[int]] = []
    groups = [("ALL", dfw)] if groupby_cols == [None] else dfw.groupby(groupby_cols, dropna=False)
    for _, g in groups:
        reps = _pick_representatives_by_t1(g)
        reps = reps.sort_values("t1", kind="mergesort").reset_index(drop=True)

        t1u = reps["t1"].to_numpy(int)
        pos = reps["_pos"].to_numpy(int)
        if len(t1u) < window_pairs:
            continue

        for start in range(0, len(t1u) - window_pairs + 1, step_pairs):
            base = t1u[start]
            expected = base + np.arange(window_pairs, dtype=int)
            if np.array_equal(t1u[start:start+window_pairs], expected):
                out.append(pos[start:start+window_pairs].tolist())
    return out


def build_windows_by_ordered_rows(
    df: pd.DataFrame,
    window_pairs: int = 7,
    step_pairs: int = 1,
    groupby_cols: Optional[List[str]] = None,
    order_cols: Optional[List[str]] = None,
    require_monotonic_t1: bool = True,
    require_t2_gt_t1: bool = False,
    drop_same_pairs: bool = True
) -> List[List[int]]:
    """
    Ventanas por filas ordenadas (robusto a t1 no consecutivos).
    Devuelve índices POSICIONALES (df.reset_index).
    """
    assert {"t1", "t2"}.issubset(df.columns), "Faltan columnas t1/t2."
    dfw = df.copy()
    if "_pos" not in dfw.columns:
        dfw = dfw.reset_index(drop=True)
        dfw["_pos"] = np.arange(len(dfw), dtype=int)

    if order_cols is None:
        order_cols = [c for c in ["series_id", "area_id"] if c in dfw.columns] + ["t1", "t2", "rank"]

    groups = [("ALL", dfw)] if not groupby_cols else dfw.groupby(groupby_cols, dropna=False)
    out: List[List[int]] = []

    for _, g in groups:
        g = g.sort_values(order_cols, kind="mergesort").copy()
        if drop_same_pairs:
            g = g[g["t2"] >= g["t1"]]
        if g.empty:
            continue
        g = g.reset_index(drop=True)

        n = len(g)
        if n < window_pairs:
            continue

        t1v = g["t1"].to_numpy(int)
        t2v = g["t2"].to_numpy(int)
        pos = g["_pos"].to_numpy(int)

        for start in range(0, n - window_pairs + 1, step_pairs):
            sl = slice(start, start + window_pairs)
            ok = True
            if require_monotonic_t1:
                ok &= np.all(np.diff(t1v[sl]) > 0)
            if require_t2_gt_t1:
                ok &= np.all(t2v[sl] >  t1v[sl])
            else:
                ok &= np.all(t2v[sl] >= t1v[sl])
            if ok:
                out.append(pos[sl].tolist())
    return out


# ─────────────────────────────────────────────
# Evaluación de una ventana (sin cambios de lógica fuerte)
# ─────────────────────────────────────────────
def evaluate_window_rules(
    df: pd.DataFrame,
    row_indices: List[int],
    *,
    residuals_df: Optional[pd.DataFrame] = None,
    repeats_required_pre: int = 3,
    repeats_required_post: int = 3,
    rate_change_threshold: float = 0.1,
    block_mode: Literal["at_least", "all"] = "at_least",
    anchor: Literal["center", "left", "argmax"] = "center",
    label_column: str = "dominant_label",   # NUEVO: permite usar etiqueta suavizada
) -> Dict[str, Any]:
    """
    Evalúa consistencia en una ventana:
    - prev/post blocks de clase
    - cambio significativo
    - residual_ok
    """
    dfw = df if ("_pos" in df.columns and df.index.equals(pd.RangeIndex(len(df)))) else df.reset_index(drop=True)

    sub = dfw.iloc[row_indices].copy()
    sub = sub.sort_values(["t1", "t2"] if "t2" in dfw.columns else ["t1"], kind="mergesort").reset_index(drop=True)
    n = len(sub)

    if anchor == "left":
        mid = 0
    elif anchor == "argmax" and "cva_mag_mean" in sub.columns:
        mid = int(sub["cva_mag_mean"].astype(float).to_numpy().argmax())
    else:
        mid = (n // 2) if (n % 2 == 1) else (n // 2 - 1)

    # Nombre de columna a usar (cruda o suavizada)
    if label_column not in sub.columns:
        label_column = "dominant_label"

    out = {
        "series_id": sub.loc[mid, "series_id"] if "series_id" in sub.columns else None,
        "area_id":   sub.loc[mid, "area_id"]   if "area_id"   in sub.columns else None,
        "start_t1":  int(sub.loc[0, "t1"]),
        "end_t2":    int(sub.loc[n-1, "t2"]),
        "first_label":  str(sub.loc[0,    label_column]),
        "end_label":    str(sub.loc[n-1,  label_column]),
        "center_label": str(sub.loc[mid,  label_column]),
        "center_t1": int(sub.loc[mid, "t1"]),
        "center_t2": int(sub.loc[mid, "t2"]),
        "from_class": "",
        "to_class": "",
        "prev_block_ok": False,
        "post_block_ok": False,
        "rate_sum_to": 0.0,
        "residual_min_center": np.nan,
        "residual_ok": False,
        "pair_t1s": sub["t1"].astype(int).tolist(),
        "pair_t2s": sub["t2"].astype(int).tolist(),
        "consistent": False,
        "reason": "",
    }

    parsed = [_parse_pair_label(x) for x in sub[label_column]]
    if any(p is None for p in parsed):
        out["reason"] = "invalid_labels"
        return out
    from_mid, to_mid = parsed[mid]
    out["from_class"], out["to_class"] = from_mid, to_mid

    prev_slice = parsed[max(0, mid - repeats_required_pre):mid]
    post_slice = parsed[mid + 1: mid + 1 + repeats_required_post]
    prev_ok = (len(prev_slice) >= repeats_required_pre) and all(pt == from_mid for (pf, pt) in prev_slice)
    post_ok = (len(post_slice) >= repeats_required_post) and all(pt == to_mid   for (pf, pt) in post_slice)
    
    if block_mode == "all":
        prev_ok = prev_ok and all(pt == from_mid for (pf, pt) in parsed[:mid])
        post_ok = post_ok and all(pt == to_mid   for (pf, pt) in parsed[mid+1:])

    out["prev_block_ok"] = prev_ok
    out["post_block_ok"] = post_ok

    rate_map = {"building": "rate_bu_change", "road": "rate_ro_change", "background": "rate_bk_change"}

    if to_mid == "background":
        other_classes = [cls for cls in rate_map.keys() if cls != "background"]
        rate_sum = 0.0
        for cls in other_classes:
            rate_col = rate_map.get(cls)
            if rate_col in sub.columns:
                val = pd.to_numeric(sub.loc[mid, rate_col], errors="coerce")
                if pd.notna(val) and val < 0:
                    rate_sum += abs(val)
        out["rate_sum_to"] = rate_sum
    else:
        rate_col = rate_map.get(to_mid)
        center_t1 = out["center_t1"]
        center_t2_consecutive = center_t1 + 1
        rate_pair_query = dfw[(dfw["t1"] == center_t1) & (dfw["t2"] == center_t2_consecutive)]
        if not rate_pair_query.empty and rate_col in rate_pair_query.columns:
            val = pd.to_numeric(rate_pair_query[rate_col].iloc[0], errors="coerce")
            out["rate_sum_to"] = float(val) if pd.notna(val) else 0.0
        elif rate_col in sub.columns:
            val = pd.to_numeric(sub.loc[mid, rate_col], errors="coerce")
            out["rate_sum_to"] = float(val) if pd.notna(val) else 0.0
        else:
            out["rate_sum_to"] = 0.0

    if residuals_df is not None and "ts1" in residuals_df.columns:
        t1_center = out["center_t1"]
        res_row = residuals_df[residuals_df["ts1"] == t1_center]
        if not res_row.empty:
            b_out = bool(res_row["building_outlier"].iloc[0]) if "building_outlier" in res_row.columns else False
            r_out = bool(res_row["road_outlier"].iloc[0])     if "road_outlier"     in res_row.columns else False
            out["residual_ok"] = not (b_out or r_out)
            rb = pd.to_numeric(res_row.get("residual_building", pd.Series([np.nan])).iloc[0], errors="coerce")
            rr = pd.to_numeric(res_row.get("residual_roads",    pd.Series([np.nan])).iloc[0], errors="coerce")
            out["residual_min_center"] = float(min(rb, rr)) if pd.notna(rb) and pd.notna(rr) else np.nan
        else:
            out["residual_ok"] = True
    else:
        out["residual_ok"] = True

    change_con = (from_mid != to_mid) or (out["rate_sum_to"] >= rate_change_threshold)
    out["consistent"] = prev_ok and post_ok and out["residual_ok"] and change_con

    if not out["consistent"]:
        if not change_con: out["reason"] = "no_significant_change"
        elif not prev_ok:  out["reason"] = "prev_block_fail"
        elif not post_ok:  out["reason"] = "post_block_fail"
        elif not out["residual_ok"]: out["reason"] = "residual_flagged"
        else: out["reason"] = "failed_some_check"
    else:
        out["reason"] = "ok"
    return out


# ─────────────────────────────────────────────
# Scan general (con soporte de smoothing)
# ─────────────────────────────────────────────
def scan_temporal_consistency(
    df: pd.DataFrame,
    *,
    window_pairs: int = 7,
    step_pairs: int = 1,
    groupby_cols: Optional[List[str]] = None,
    residuals_df: Optional[pd.DataFrame] = None,
    repeats_required_pre: int = 3,
    repeats_required_post: int = 3,
    rate_change_threshold: float = 0.1,
    block_mode: Literal["at_least", "all"] = "at_least",
    window_mode: Literal["row", "t1_consecutive"] = "row",
    anchor: Literal["center", "left", "argmax"] = "center",
    # NUEVO:
    smooth_alpha: Optional[float] = None,
    use_smoothed_labels: bool = True,
) -> pd.DataFrame:
    """
    Orquesta el escaneo temporal devolviendo un DataFrame de ventanas evaluadas.
    Si smooth_alpha se da, reconstruye fracciones por fecha, aplica EMA y añade
    la columna 'dominant_label_smooth' para usar en los bloques.
    """
    dfw = df.reset_index(drop=True).copy()
    dfw["_pos"] = np.arange(len(dfw), dtype=int)

    label_col = "dominant_label"
    if smooth_alpha is not None:
        dfw = add_smoothed_labels(dfw, alpha=smooth_alpha)
        if use_smoothed_labels and "dominant_label_smooth" in dfw.columns:
            label_col = "dominant_label_smooth"

    # Construcción de ventanas
    if window_mode == "row":
        wins = build_windows_by_ordered_rows(
            dfw, window_pairs, step_pairs,
            groupby_cols=groupby_cols,
            order_cols=[c for c in ["series_id", "area_id"] if c in dfw.columns] + ["t1", "t2", "rank"],
            require_monotonic_t1=True,
            require_t2_gt_t1=False,
            drop_same_pairs=True
        )
    else:
        wins = build_windows_by_t1_consecutive(dfw, window_pairs, step_pairs, groupby_cols)

    # Evaluación
    evals = [
        evaluate_window_rules(
            dfw, w,
            residuals_df=residuals_df,
            repeats_required_pre=repeats_required_pre,
            repeats_required_post=repeats_required_post,
            rate_change_threshold=rate_change_threshold,
            block_mode=block_mode,
            anchor=anchor,
            label_column=label_col,
        )
        for w in wins
    ]
    return pd.DataFrame(evals)


def summarize_top_changes(
    df_metrics: pd.DataFrame,
    df_windows: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Toma ventanas consistentes y resume Top-N eventos.
    La fecha de cambio se fija como la del par consecutivo siguiente.
    """
    cons = df_windows[df_windows["consistent"] == True].copy()
    if cons.empty:
        return pd.DataFrame(columns=[
            "series_id","area_id","center_t1","to_class",
            "kliep_abs","cva_mag_mean","overall_score","change_date"
        ])

    rows = []
    for _, r in cons.iterrows():
        ct1 = int(r["center_t1"])
        sid = r.get("series_id", None)
        aid = r.get("area_id", None)

        q = df_metrics[df_metrics["t1"] == ct1]
        if sid is not None and "series_id" in df_metrics.columns:
            q = q[q["series_id"] == sid]
        if aid is not None and "area_id" in df_metrics.columns:
            q = q[q["area_id"] == aid]
        if q.empty:
            continue

        q_pref = q[q["t2"] == (ct1 + 1)]
        if not q_pref.empty:
            base = q_pref.sort_values(["rank","cva_mag_mean"], ascending=[True, False]).iloc[0]
        else:
            q_next = q[q["t2"] > ct1].sort_values("t2", ascending=True)
            if not q_next.empty:
                base = q_next.sort_values(["t2","rank","cva_mag_mean"], ascending=[True, True, False]).iloc[0]
            else:
                base = q.sort_values(["rank","cva_mag_mean"], ascending=[True, False]).iloc[0]

        kliep = float(base["kliep_abs"]) if "kliep_abs" in base else float("nan")
        mag   = float(base["cva_mag_mean"]) if "cva_mag_mean" in base else float("nan")
        change_date = base.get("date_change", None)
        overall_score = (kliep + mag) / 2.0

        rows.append({
            "series_id": base.get("series_id", sid),
            "area_id":   base.get("area_id", aid),
            "center_t1": ct1,
            "to_class":  r.get("to_class", ""),
            "kliep_abs": kliep,
            "cva_mag_mean": mag,
            "overall_score": overall_score,
            "change_date": change_date,
        })

    top_df = pd.DataFrame(rows)
    if top_df.empty:
        return top_df
    top_df = top_df.sort_values("overall_score", ascending=False).head(top_n).reset_index(drop=True)
    return top_df
