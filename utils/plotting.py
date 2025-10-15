import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import rasterio as rio
import geopandas as gpd
import pyproj

from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
from rasterio.transform import rowcol
from rasterio import features
from affine import Affine

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Bbox

from shapely.ops import transform as shp_transform

def plot_entire_timeline(array_list: list, code: str, type_image="sentinel2", show=True):
        
        ncol = 10
        N = len(array_list)
        nrow = int(np.ceil(N / ncol))

        # Tamaño por celda (ajusta tile si quieres miniaturas más grandes)
        tile = 3  # pulgadas por celda
        fig, axes = plt.subplots(nrow, ncol, figsize=(tile*ncol, tile*nrow), squeeze=False)

        # CERO huecos: quitamos layout automático y fijamos márgenes/espaciado mínimos
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02)

        for i, ax in enumerate(axes.flat):
            if i < N:
                if type_image == "sentinel2":
                    rgb = array_list[i]["image"][[3,2,1]].transpose(1, 2, 0) * (2.5 / 10_000)
                    label = f"{array_list[i]['date'].split()[0]} [{i}] - {(array_list[i]['cloud_fraction']*100):.1f}% clouds"
                    filename = f"../outputs/{code}/s2_{code}.png"

                elif type_image == "sentinel2_filtered":
                    rgb = array_list[i]["image"][[3,2,1]].transpose(1, 2, 0)  * 2.5 
                    label = f"{array_list[i]['date'].split()[0]} [{i}] - {(array_list[i]['cloud_fraction']*100):.1f}% clouds"
                    filename = f"../outputs/{code}/s2f_{code}.png"
                
                elif type_image == "predicted":                                   
                    rgb = array_list[i]["image"].transpose(1, 2, 0)
                    label = f"{array_list[i]['date'].split()[0]} [{i}]"
                    filename = f"../outputs/{code}/ss_{code}.png"
                
                elif type_image == "superres":
                    rgb = array_list[i]["sr"].transpose(1, 2, 0).astype(np.float32)
                    label = f"{array_list[i]['date'].split()[0]} [{i}]"
                    filename = f"../outputs/{code}/sr_{code}.png"
                
                # Etiqueta DENTRO del eje (no como título), así no se reserva espacio extra
                ax.imshow(np.clip(rgb, 0, 1))               
                ax.text(
                    0.02, 0.02, label,
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5, edgecolor="none")
                )
                ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            else:
                ax.remove()  # elimina ejes sobrantes sin dejar “parches” visibles
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)


def select_indices_separated(N, k=4, min_sep=4):
    idxs = []
    candidates = set(range(N))
    while len(idxs) < k and candidates:
        i = np.random.choice(list(candidates))
        idxs.append(i)
        # Elimina vecinos cercanos
        for j in range(i-min_sep, i+min_sep+1):
            candidates.discard(j)
    return sorted(idxs)


def plot_s2_timeline(s2_images: dict, code:str, selected_idx=None, show=True, type_image="sentinel2"):
    """
    Genera un timeline de imágenes Sentinel-2 con miniaturas y lo guarda en un archivo.

    Parámetros
    ----------
    s2_images : dict
        Diccionario con llaves "date" y "image", donde
        - "date"  : lista de strings o datetimes
        - "image" : lista de arrays numpy con bandas (CHW o HWC).
    code : str
        Código del área (usado para nombrar el archivo de salida).
    selected_idx : list, opcional
        Lista de índices a resaltar. Si None, se usan [1, 22, 40, 52].
    show : bool, opcional
        Si True, muestra la figura después de guardarla.
    type_image : str, opcional
        Tipo de imagen, por defecto "sentinel2". La otra opción es "predicted".
    """

    # ---------- 1) Preparar DataFrame ----------
    dates = [s2_image["date"] for s2_image in s2_images]
    imgs  = [s2_image["image"] for s2_image in s2_images]

    dt_index = pd.to_datetime(dates)
    df = pd.DataFrame({"image": imgs}, index=dt_index).sort_index()
    n = len(df)

    if selected_idx is None:
        selected_idx = [1, 22, 40, 52]
    idx = [min(k, n-1) for k in selected_idx if n > 0]
    selected = df.iloc[idx] if len(idx) > 0 else df.iloc[:0]

    # ---------- 2) Crear figura y eje ----------
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.32])  # franja inferior

    # ---------- 3) Parámetros ----------
    target_h = 128
    alpha_bg = 0.5
    y0 = 0.0

    # Línea base (spine)
    ax.spines['bottom'].set_position(('data', -0.01))
    ax.spines['bottom'].set_zorder(0)

    # ---------- 4) Miniaturas de fondo ----------
    for dt, row in df.iterrows():
        arr = row['image']
        if type_image == "predicted":
            rgb = arr.transpose(1, 2, 0)
        
        elif type_image == "sentinel2":
            rgb = arr[[3, 2, 1], :, :].transpose(1, 2, 0) * (2.5/10000)
        
        elif type_image == "sentinel2_filtered":
            rgb = arr[[3, 2, 1], :, :].transpose(1, 2, 0) * 2.5 
        
        elif type_image == "superres":
            rgb = arr.transpose(1, 2, 0).astype(np.float32)

        
        h, *_ = rgb.shape
        zoom = target_h / h
        thumb = OffsetImage(np.clip(rgb, 0, 1), zoom=zoom, alpha=alpha_bg)
        ab = AnnotationBbox(thumb, (dt, y0),
                            xycoords='data', box_alignment=(0.5, 0),
                            bboxprops=dict(edgecolor='black', linewidth=1),
                            frameon=True, pad=0, zorder=1)
        ax.add_artist(ab)

    # ---------- 5) Puntos de adquisición ----------
    ax.margins(x=0.015)
    ax.set_xlim(df.index.min(), df.index.max())
    y_min = ax.get_ylim()[0]
    ax.scatter(df.index, [y_min - 0.01]*n, s=30, color='skyblue',
               zorder=1, clip_on=False)

    # ---------- 6) Resaltar seleccionadas ----------
    for dt, row in selected.iterrows():
        arr = row['image']
        if type_image == "predicted":
            rgb = arr.transpose(1, 2, 0)
            output_path = f"../outputs/{code}/ss_ts_{code}.png"
        
        elif type_image == "sentinel2":
            rgb = arr[[3, 2, 1], :, :].transpose(1, 2, 0) * (2.5/10000)
            output_path = f"../outputs/{code}/s2_ts_{code}.png"
        
        elif type_image == "sentinel2_filtered":
            rgb = arr[[3, 2, 1], :, :].transpose(1, 2, 0) * 2.5 
            output_path = f"../outputs/{code}/s2f_ts_{code}.png"
        
        elif type_image == "superres":
            rgb = arr.transpose(1, 2, 0).astype(np.float32)
            output_path = f"../outputs/{code}/sr_ts_{code}.png"

        h, *_ = rgb.shape
        zoom = target_h / h
        thumb_sel = OffsetImage(np.clip(rgb, 0, 1), zoom=zoom, alpha=1)
        ab_sel = AnnotationBbox(thumb_sel, (dt, y0),
                                xycoords='data', box_alignment=(0.5, 0),
                                frameon=True, pad=0,
                                bboxprops=dict(edgecolor='red', linewidth=2.5),
                                zorder=3)
        ax.add_artist(ab_sel)
        ax.scatter([dt], [y_min - 0.01], s=25, facecolors='none',
                   edgecolors='red', linewidths=2, zorder=1, clip_on=False)

    # ---------- 7) Ejes y estilo ----------
    ax.set_xticks(pd.date_range(df.index.min(), df.index.max(), freq='MS'))
    ax.xaxis.set_major_formatter(DateFormatter("%b %y"))
    ax.tick_params(axis='x', rotation=0, labelsize=11)

    ax.set_yticks([])
    ax.set_ylim(y_min, 0.18)
    for sp in ['left', 'right', 'top']:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('')

    # ---------- 8) Guardar figura ----------
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)


def _to_geom(thing):
    if hasattr(thing, "geometry"):       # GeoDataFrame/GeoSeries
        geoms = list(thing.geometry)
    elif isinstance(thing, (list, tuple)):
        geoms = list(thing)
    else:
        geoms = [thing]
    return unary_union(geoms)


def crop_closeup_CHW(image_chw, polygon_like, transform,
                     buffer_dist=50, mask_outside=False):
    """
    Recorta un close-up (por bbox) alrededor de un polígono con buffer.
    Devuelve: cropped [C,h,w], (r0,r1,c0,c1), window_transform
    """
    H, W = image_chw.shape[1], image_chw.shape[2]
    geom = _to_geom(polygon_like)
    geom_buf = geom.buffer(buffer_dist)

    # bbox en coords del CRS
    minx, miny, maxx, maxy = geom_buf.bounds

    # esquinas a (fila, col) con el transform global
    r0, c0 = rowcol(transform, minx, maxy)  # sup-izq
    r1, c1 = rowcol(transform, maxx, miny)  # inf-der

    # ordenar y acotar
    r0, r1 = sorted((r0, r1)); c0, c1 = sorted((c0, c1))
    r0 = max(0, r0); c0 = max(0, c0)
    r1 = min(H, r1); c1 = min(W, c1)

    cropped = image_chw[:, r0:r1, c0:c1]

    # transform de la ventana recortada
    a, b, c, d, e, f = transform[:6]
    x0, y0 = transform * (c0, r0)
    window_transform = Affine(a, b, x0, d, e, y0)

    if not mask_outside:
        return cropped, (r0, r1, c0, c1), window_transform

    # máscara opcional
    mask = features.rasterize(
        [(geom_buf, 1)], out_shape=(r1 - r0, c1 - c0),
        transform=window_transform, fill=0, all_touched=False, dtype=bool
    )
    out = np.zeros_like(cropped)
    for i in range(cropped.shape[0]):
        out[i] = np.where(mask, cropped[i], 0)
    return out, (r0, r1, c0, c1), window_transform


def plot_polygon_in_window(ax, polygon_like, window_transform, **kw):
    """Dibuja los contornos en coords de la ventana (no del raster completo)."""
    geom = _to_geom(polygon_like)
    def _plot_one(poly):
        x, y = poly.exterior.xy
        rr, cc = rowcol(window_transform, x, y)  # ¡ojo: transform de la ventana!
        ax.plot(cc, rr, **kw)
        for ring in poly.interiors:
            xi, yi = ring.xy
            rri, cci = rowcol(window_transform, xi, yi)
            ax.plot(cci, rri, **kw)

    if isinstance(geom, MultiPolygon):
        for g in geom.geoms: _plot_one(g)
    elif isinstance(geom, Polygon):
        _plot_one(geom)
    else:
        # por si llega un GeometryCollection
        try:
            for g in geom.geoms: 
                if isinstance(g, (Polygon, MultiPolygon)):
                    plot_polygon_in_window(ax, g, window_transform, **kw)
        except Exception:
            pass

def chw_to_hwc(a):
                """Convierte [C,H,W] -> [H,W,C] si aplica."""
                return a.transpose(1, 2, 0) if (a.ndim == 3 and a.shape[0] in (1, 3, 4)) else a

def s2_to_rgb(chw):
    """Escala S2 (CHW) a [0,1] y HWC."""
    im = chw_to_hwc(chw) * 2.5
    return np.clip(im, 0, 1)

def sr_to_rgb(chw):
    """SR (CHW) a HWC [0,1] (asumiendo ya float); aplica clip por seguridad."""
    im = chw_to_hwc(chw.astype(np.float32))
    return np.clip(im, 0, 1)

# ------------------------------------------------
# ==== helpers t1 -> posición en out_arrays ====
def _build_t1_to_pos(out_arrays):
    mapping = {}
    for pos, it in enumerate(out_arrays):
        if isinstance(it, dict):
            if 't1' in it and it['t1'] is not None:
                mapping[int(it['t1'])] = pos
            elif 'date_idx' in it and it['date_idx'] is not None:
                mapping[int(it['date_idx'])] = pos
    return mapping

def _pos_from_t1(t1, t1_to_pos, out_len):
    if t1_to_pos:
        if t1 in t1_to_pos:
            return t1_to_pos[t1]
        try:
            t1i = int(t1)
            if t1i in t1_to_pos:
                return t1_to_pos[t1i]
        except Exception:
            pass
    if isinstance(t1, (int, np.integer)) and 0 <= t1 < out_len:
        return int(t1)
    raise KeyError(
        f"No se pudo mapear t1={t1} a una posición de out_arrays; "
        f"añade 't1' o 'date_idx' a cada item, o pasa panel_t1s compatibles."
    )


# ==== función principal ====
def unified_closeup_and_ts(
    out_arrays, s2_profile, shape, idx: int,
    df_metrics, top_df,
    panel_t1s=None,                      # lista de t1; se eligen 4 centrados en idx (por valor)
    highlight_col: int = 1,              # recalculado según idx
    factor: float = 1.0,               # factor de brillo para todas las imágenes (SR y S2)
    type: str = "building",            # "building" o "road" (solo afecta al time-series)
    circle_size: float = 210,
    circle_ec: str = "#d7263d",
    circle_lw: float = 2.2,

    # círculo opcional en closeups
    top_circle_xy: tuple | None = None,
    top_circle_r: float | None = None,
    top_circle_kwargs: dict | None = None,
    plot_polygon_in_closeups: bool = True,

    figsize=(12, 9), savepath=None, dpi=300,

    # proporciones dentro del panel superior (misma fila)
    overview_frac: float = 0.35,  # ancho relativo del overview (0..1)
    label_frac: float = 0.10,     # fracción del resto para la columna de etiquetas (0..1)

    # altura relativa de los paneles: el top será más alto que el bottom
    top_panel_frac: float = 0.65, # 0.65 arriba / 0.35 abajo (ajústalo a tu gusto)

    # ajustes del timeseries
    xleft_pad: float = 0.4,       # margen izquierdo extra (negativo) en el eje X
    show_overview_coords: bool = True,  # mostrar Lat/Lon en el overview
    show: bool = True
):
    fig = plt.figure(figsize=figsize)

    # proporción panel superior / inferior
    top_panel_frac = float(np.clip(top_panel_frac, 0.2, 0.9))
    bottom_frac = 1.0 - top_panel_frac
    gs = fig.add_gridspec(2, 1, height_ratios=[top_panel_frac, bottom_frac], hspace=0.12)

    # ===== TOP =====
    # configurar anchos de columnas según overview_frac y label_frac
    overview_frac = float(np.clip(overview_frac, 0.05, 0.85))
    label_frac    = float(np.clip(label_frac,    0.005, 0.08))

    remaining = max(1e-6, 1.0 - overview_frac)
    label_w  = remaining * label_frac
    col_w    = remaining * (1.0 - label_frac) / 4.0
    width_ratios = [overview_frac, label_w, col_w, col_w, col_w, col_w]

    # 3 filas × 6 columnas: overview (0), etiquetas (1), 4 closeups (2..5)
    gst = gs[0].subgridspec(
        3, 6,
        wspace=0.0,
        hspace=0.0,                 # CERO separación vertical entre BRS/S2/SR
        width_ratios=width_ratios
    )

    axes = np.empty((3, 6), dtype=object)
    for r in range(3):
        for c in range(1, 6):  # col 0 se usa aparte para overview
            axes[r, c] = fig.add_subplot(gst[r, c])

    # etiquetas simples (columna 1)
    for r, lab in enumerate(["BRS", "S2", "SR"]):
        ax_lbl = axes[r, 1]
        ax_lbl.axis("off")
        ax_lbl.text(0.98, 0.5, lab, transform=ax_lbl.transAxes,
                    ha="right", va="center", fontsize=11, fontweight="bold")

    # reproyección
    transform_x1 = s2_profile['transform']
    transform_x4 = rio.Affine(transform_x1.a / 4, transform_x1.b, transform_x1.c,
                              transform_x1.d, transform_x1.e / 4, transform_x1.f)
    shp = gpd.GeoSeries(shape, crs="EPSG:4326").explode(index_parts=False).reset_index(drop=True)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", s2_profile['crs'], always_xy=True).transform
    reprojected_shape = shp.apply(lambda geom: shp_transform(transformer, geom))

    # ===== selección por VALOR t1 =====
    t1_to_pos = _build_t1_to_pos(out_arrays)
    if panel_t1s is None or len(panel_t1s) == 0:
        t1_sorted = np.asarray(df_metrics.sort_values('t1')['t1'], dtype=int)
    else:
        t1_sorted = np.unique(np.asarray(panel_t1s, dtype=int))

    if idx in t1_sorted:
        pos_c = int(np.where(t1_sorted == idx)[0][0])
    else:
        pos_c = int(np.argmin(np.abs(t1_sorted - idx)))

    left = max(0, pos_c - 1)
    right = min(len(t1_sorted), left + 10)
    if right - left < 10:
        left = max(0, right - 10)
        right = min(len(t1_sorted), left + 10)

    panel_t1s_sel = t1_sorted[left:right].tolist()
    panel_pos = [_pos_from_t1(t1, t1_to_pos, len(out_arrays)) for t1 in panel_t1s_sel]

    if idx in panel_t1s_sel:
        highlight_col = int(panel_t1s_sel.index(idx))
    else:
        highlight_col = idx

    # ===== crops =====
    seg_imgs, seg_tfs = [], []
    s2_imgs,  s2_tfs  = [], []
    sr_imgs,  sr_tfs  = [], []
    titles = []
    for p, t1 in zip(panel_pos, panel_t1s_sel):
        (stn_crop, _, tf_stn) = crop_closeup_CHW(out_arrays[p]['image'], reprojected_shape, transform_x4, 10, False)
        (s2_crop,  _, tf_s2 ) = crop_closeup_CHW(out_arrays[p]['s2'][[3,2,1]], reprojected_shape, transform_x1, 40, False)
        (sr_crop,  _, tf_sr ) = crop_closeup_CHW(out_arrays[p]['sr'],              reprojected_shape, transform_x4, 10, False)
        seg_imgs.append(chw_to_hwc(stn_crop)); seg_tfs.append(tf_stn)
        s2_imgs.append(s2_to_rgb(s2_crop));    s2_tfs.append(tf_s2)
        sr_imgs.append(sr_to_rgb(sr_crop));    sr_tfs.append(tf_sr)
        titles.append(f"T{t1}: {str(out_arrays[p]['date']).split(' ')[0]}")

    # ===== OVERVIEW (SR) + inset (segmentación) =====
    p_central = panel_pos[highlight_col+1]
    full_sr = sr_to_rgb(out_arrays[p_central]['sr'])  # overview con SR
    ax_over = fig.add_subplot(gst[:, 0])              # ocupa 3 filas
    ax_over.imshow(full_sr * factor, interpolation='nearest', aspect='auto')
    ax_over.set_xticks([]); ax_over.set_yticks([])

    # bbox rojo del polígono (coords ×4)
    inv4 = ~transform_x4
    if type == "road":
        minx, miny, maxx, maxy = reprojected_shape.total_bounds
        px0, py0 = inv4 * (minx, miny)
        px1, py1 = inv4 * (maxx, maxy)
        x0, x1 = min(px0, px1), max(px0, px1)
        y0, y1 = min(py0, py1), max(py0, py1)
        ax_over.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, ec='crimson', lw=2.0))
    
    # --- helper: dibuja exterior (y opcionalmente agujeros) de un shapely polygon ---
    def _plot_polygon_pixels(ax, poly, inv_affine, ec='black', lw=1.6):
        # exterior
        ext = np.array(poly.exterior.coords)
        ext_px = np.array([inv_affine * (x, y) for x, y in ext])
        ax.add_patch(MplPolygon(ext_px, closed=True, fill=False, edgecolor=ec, linewidth=lw))

    # dibujar el/los polígonos en negro
    for geom in reprojected_shape.geometry:
        if geom is None:
            continue
        g = geom  # puede venir como (Multi)Polygon o colección
        # Multiparte
        if getattr(g, "geom_type", "") == "MultiPolygon":
            for sub in g.geoms:
                _plot_polygon_pixels(ax_over, sub, inv4, ec='black', lw=1.6)
        elif getattr(g, "geom_type", "") == "Polygon":
            _plot_polygon_pixels(ax_over, g, inv4, ec='black', lw=1.6)
        else:
            # si fuera LineString/GeometryCollection: ignora o adapta aquí
            try:
                for sub in g.geoms:
                    if sub.geom_type == "Polygon":
                        _plot_polygon_pixels(ax_over, sub, inv4, ec='black', lw=1.6)
            except Exception:
                pass


    for sp in ax_over.spines.values():
        sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1.2)

    # Lat/Lon del centroide del AOI
    if show_overview_coords:
        centroid_ll = gpd.GeoSeries(shape, crs="EPSG:4326").unary_union.centroid
        lat, lon = centroid_ll.y, centroid_ll.x
        txt = f"Lat: {lat:.4f}\nLon: {lon:.4f}"
        ax_over.text(0.02, 0.98, txt, transform=ax_over.transAxes,
                     ha="left", va="top",
                     bbox=dict(facecolor="white", edgecolor="black",
                               boxstyle="round,pad=0.25", alpha=0.9),
                     fontsize=9, zorder=5)

    # inset: segmentación (esquina inferior izquierda)
    ax_in = inset_axes(ax_over, width="30%", height="30%", loc="lower left", borderpad=0.8)
    ax_in.imshow(chw_to_hwc(out_arrays[p_central]['image']), interpolation='nearest', aspect='auto')
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1.0)

    # ===== closeups SIN espacio vertical =====
    def _frame_axes(ax):
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1.0)

    def _inline_col_title(ax, txt):
        ax.text(0.5, 1.001, txt, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, clip_on=False)

    def _imshow_full(ax, img):
        ax.imshow(img, interpolation='nearest', aspect='auto')  # ocupa toda la celda
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_anchor('N')  # pega arriba

    for j in range(4):
        # BRS
        ax = axes[0, 2 + j]
        _imshow_full(ax, seg_imgs[j])
        if plot_polygon_in_closeups:
            plot_polygon_in_window(ax, reprojected_shape, seg_tfs[j], color='black', linewidth=1.2)
        _inline_col_title(ax, titles[j]); _frame_axes(ax)
        if top_circle_xy and top_circle_r:
            ax.add_patch(Circle(top_circle_xy, top_circle_r, fill=False, ec='r', lw=1.6, **(top_circle_kwargs or {})))
        # S2
        ax = axes[1, 2 + j]
        _imshow_full(ax, s2_imgs[j] * factor)
        if plot_polygon_in_closeups:
            plot_polygon_in_window(ax, reprojected_shape, s2_tfs[j], color='black', linewidth=1.2)
        _frame_axes(ax)
        if top_circle_xy and top_circle_r:
            ax.add_patch(Circle(top_circle_xy, top_circle_r, fill=False, ec='r', lw=1.6, **(top_circle_kwargs or {})))
        # SR
        ax = axes[2, 2 + j]
        _imshow_full(ax, sr_imgs[j] * factor)
        if plot_polygon_in_closeups:
            plot_polygon_in_window(ax, reprojected_shape, sr_tfs[j], color='black', linewidth=1.2)
        _frame_axes(ax)
        if top_circle_xy and top_circle_r:
            ax.add_patch(Circle(top_circle_xy, top_circle_r, fill=False, ec='r', lw=1.6, **(top_circle_kwargs or {})))

    def add_frame_around_columns(fig, axes3x6, start_col, num_cols=2, color="#d7263d", lw=3.0, pad=0.005):
        """
        Añade un marco rojo alrededor de múltiples columnas consecutivas.
        
        Args:
            fig: Figura de matplotlib
            axes3x6: Array de ejes (3 filas x 6 columnas)
            start_col: Columna inicial (0-indexed)
            num_cols: Número de columnas a enmarcar (default: 2)
            color: Color del marco
            lw: Grosor de línea
            pad: Padding alrededor del marco
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        # Recoger bounding boxes de todas las columnas especificadas
        all_bbs = []
        for col in range(start_col, start_col + num_cols):
            if col < axes3x6.shape[1]:  # Verificar que la columna existe
                col_bbs = [axes3x6[r, col].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                        for r in range(3)]
                all_bbs.extend(col_bbs)
        
        if all_bbs:
            # Unir todas las bounding boxes
            bb = Bbox.union(all_bbs)
            bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            rect = Rectangle((bb.x0, bb.y0), bb.width, bb.height,
                            transform=fig.transFigure, fill=False, ec=color,
                            lw=lw, zorder=1000, clip_on=False)
            fig.add_artist(rect)

    fig.subplots_adjust(left=0.03, right=0.995, top=0.98, bottom=0.12)
    add_frame_around_columns(fig, axes, start_col=2 + highlight_col)

    # ===== BOTTOM: time-series (círculo y tick rojo atados al idx) =====
    def _nearest_t1(series, target):
        """Devuelve (valor_mas_cercano, indice_fila) en una serie 1D."""
        s = np.asarray(series, dtype=float)
        j = int(np.argmin(np.abs(s - float(target))))
        return series.iloc[j], j

    ax = fig.add_subplot(gs[1])
    dfm = df_metrics.sort_values(by='t1').reset_index(drop=True)
    x = dfm['t1']                 # t1 de referencia
    y = dfm['kliep_abs']
    d = dfm['cva_mag_mean']

    # curvas
    ax.plot(x, y, marker='o', label='KLIEP Score', zorder=2)
    ax.plot(x, d / max(d.max(), 1e-9) * max(y.max(), 1e-9), marker='s',
            label='Mean CVA Magnitude', zorder=2)

    # (opcional) vlines de cambios detectados - independientes del tick/círculo
    for cp in sorted(set(top_df['center_t1'].tolist())):
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8, zorder=1)

    # === clave: usar idx para fijar círculo y tick ===
    # si idx no existe exactamente, toma el t1 más cercano (xi)
    xi, i = _nearest_t1(dfm['t1'], idx)
    yi = y.iloc[i]

    # círculo en (xi, yi)
    ax.scatter([xi], [yi], s=circle_size, facecolors="none",
            edgecolors=circle_ec, linewidths=circle_lw, zorder=6)

    # asegurar ticks enteros e incluir xi
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_xticks()
    if xi not in ticks:
        ax.set_xticks(sorted(np.unique(np.append(ticks, xi))))

    # reset todos los ticks a negro y luego marca SOLO xi en rojo
    for tick in ax.get_xticklabels():
        tick.set_color('black'); tick.set_fontweight('normal')
    for tick, val in zip(ax.get_xticklabels(), ax.get_xticks()):
        if int(round(val)) == int(xi):
            tick.set_color('crimson'); tick.set_fontweight('bold')

    # Calcular el máximo valor en el eje x
    x_max = x.max()  # Suponiendo que `x` es una Serie o un array con los valores del eje x

    # Ajustar los límites del eje x
    ax.set_xlim(left=-abs(float(xleft_pad)), right=x_max + abs(float(xleft_pad)))

    ax.set_ylabel('KLIEP Score')
    cp_proxy  = Line2D([0], [0], color='red', linestyle='--', label='Detected change(s)')
    kl_proxy  = Line2D([0], [0], marker='o', color='w', label='KLIEP Score',
                    markerfacecolor='tab:blue', markersize=8)
    cva_proxy = Line2D([0], [0], marker='s', color='w', label='Mean CVA Magnitude',
                    markerfacecolor='tab:orange', markersize=8)
    ax.legend(handles=[kl_proxy, cva_proxy, cp_proxy], loc='best')
    ax.grid(True, alpha=0.25)

    if savepath:
       fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
       plt.show()
       plt.close(fig)
    
    elif not show:
       plt.close(fig)
