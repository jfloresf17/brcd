import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class NCBAResult:
    q1: int             # change start (index of the middle sample in the 3-pt window)
    q2: int             # change end (stabilization)
    change_t: int       # time index of the largest increment between q1..q2
    change_mag: float   # mean(after q2) - mean(before q1)
    is_ncba: bool       # classification given the global threshold

def detect_ncba(
    F: np.ndarray,
    *,
    alpha: float = 1.0,
    require_q2_after_q1: bool = True,
) -> Tuple[List[NCBAResult], float, np.ndarray]:
    """
    Detect Newly Constructed Building Areas (NCBAs) from time-series features.

    Parameters
    ----------
    F : np.ndarray
        Array of shape (n_objects, n_times) with the building feature per object over time.
        Values may be in any scale; only differences are used.
    alpha : float
        Threshold = mean(change_mag) + alpha * std(change_mag). Default 1.0.
    require_q2_after_q1 : bool
        If True, q2 (min of D) is searched only after q1 (max of D).

    Returns
    -------
    results : List[NCBAResult]
        One result per object with q1, q2, change_t, change_mag, is_ncba.
    threshold : float
        The global threshold applied to change magnitudes.
    magnitudes : np.ndarray
        Vector of change magnitudes for all objects.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    n, T = F.shape
    if T < 3:
        raise ValueError("Need at least 3 time steps.")
    
    # Second discrete difference: D[t] = F[t+1] - 2F[t] + F[t-1]  (t = 1..T-2)
    D = F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]          # (n, T-2)

    results: List[NCBAResult] = []
    mags = np.empty(n, dtype=float)

    for i in range(n):
        fi = F[i]
        di = D[i]

        # Q1: index of max(D) → maps to "middle" of the 3-pt window → time index q1 = argmax(D)+1
        q1_mid = int(np.nanargmax(di))
        q1 = q1_mid + 1

        # Q2: index of min(D). By default search AFTER q1 so q2 > q1.
        if require_q2_after_q1 and q1_mid + 1 < di.size:
            q2_mid_local = int(np.nanargmin(di[q1_mid + 1:]))
            q2 = (q1_mid + 1) + q2_mid_local + 1
        else:
            q2_mid = int(np.nanargmin(di))
            q2 = q2_mid + 1

        # Enforce ordering and valid bounds
        q1 = max(1, min(q1, T - 2))
        q2 = max(q1 + 1, min(q2, T - 1))

        # Change point: largest increment estrictamente dentro de [q1 .. q2)
        inc = np.diff(fi)  # T-1

        left  = q1
        right = max(q1 + 1, q2 - 1)   # interior estricto
        seg = inc[left:right]         # inc[left..right-1] termina a lo sumo en q2-1

        if seg.size > 0:
            # opcional: solo aumentos
            # seg = np.where(seg > 0, seg, -np.inf)
            rel = int(np.nanargmax(seg))
            change_t = left + rel + 1     # ≤ q2-1
        else:
            # episodio mínimo; elige una convención coherente
            change_t = min(q1 + 1, q2 - 1) if (q2 - q1) >= 2 else q1 + 1
            
        # Change magnitude: mean(after q2) - mean(before q1)
        pre = fi[:q1]
        post = fi[q2 + 1:]
        pre_mean = float(np.nanmean(pre)) if pre.size else float(fi[q1])
        post_mean = float(np.nanmean(post)) if post.size else float(fi[q2])
        mag = post_mean - pre_mean
        mags[i] = mag

        results.append(NCBAResult(q1=q1, q2=q2, change_t=change_t, change_mag=mag, is_ncba=False))

    # Global threshold on change magnitude
    mu = float(np.nanmean(mags))
    sd = float(np.nanstd(mags, ddof=0))
    threshold = mu + alpha * sd

    for r, m in zip(results, mags):
        r.is_ncba = bool(m > threshold)

    return results, threshold, mags
