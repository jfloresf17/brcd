import pathlib
import pyproj
import leafmap
import rasterio as rio
import numpy as np
import torch
import onnxruntime as ort

from model.RRSNet_arch import RRSNet 

def process_superresolution(folder_path: pathlib.Path, s2_path: pathlib.Path, return_array=False):
    """
    Realiza todo el flujo de trabajo de superresolución: genera la imagen de referencia
    a partir de la imagen Sentinel-2, procesa las imágenes y guarda o devuelve la imagen superresuelta.

    Args:
        folder_path (pathlib.Path): Carpeta donde se guardarán los resultados.
        s2_path (pathlib.Path): Ruta de la imagen Sentinel-2.
        return_array (bool): Si es True, devuelve el array en lugar de guardar el archivo.

    Returns:
        np.ndarray: Array de la imagen superresuelta (si `return_array` es True).
    """
    # Abrir la imagen Sentinel-2 para obtener los límites y el CRS
    with rio.open(s2_path) as s2_src:
        s2_minx, s2_miny, s2_maxx, s2_maxy = s2_src.bounds
        crs = s2_src.crs  # Obtener el CRS de la imagen Sentinel-2
        profile = s2_src.profile
        print(crs)

    # Transformar los límites al sistema de coordenadas geográficas (EPSG:4326)
    inv_transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minlat, minlon = inv_transformer.transform(s2_minx, s2_miny)
    maxlat, maxlon = inv_transformer.transform(s2_maxx, s2_maxy)

    bbox = [minlat, minlon, maxlat, maxlon]
    ref_path = folder_path / "ref_satellite.tif"

    # Asegurarte de que el CRS esté en formato EPSG
    crs_epsg = crs.to_epsg()
    if crs_epsg is None:
        raise ValueError("El CRS no tiene un código EPSG válido.")

    if ref_path.exists():
        print("La imagen de referencia ya existe. Usando la imagen existente.")
    else:
        leafmap.map_tiles_to_geotiff(str(ref_path), bbox, zoom=20, source="SATELLITE", crs=f"EPSG:{crs_epsg}")
    
    # Procesar las imágenes
    with rio.open(ref_path) as src, rio.open(s2_path) as s2_src:
        # Generar una ventana para la imagen Sentinel-2
        window = rio.windows.from_bounds(s2_minx, s2_miny, s2_maxx, s2_maxy, src.transform)

        # Leer la imagen satelital
        satimage = src.read(window=window)
        satimage_res = resample_image(satimage, size=(256, 256)) / 255.0
        sat_image_down = resample_image(satimage_res, size=(64, 64))
        sat_image_up = resample_image(sat_image_down, size=(256, 256))

        # Leer la imagen Sentinel-2
        s2 = s2_src.read()
        s2 = np.clip(s2 / 10000, 0, 1)[[3, 2, 1]]

        # Superresolver la imagen
        super_img = superresolve_image(s2)
        print(s2.shape, super_img.shape, satimage_res.shape, sat_image_up.shape)

        # Verificar valores NaN
        if check_nan([s2, super_img, satimage_res, sat_image_up]):
            print("NaN values detected in the inputs. Skipping model execution...")
            return None

        # Ejecutar la red
        fake_HR = run_network(s2, super_img, satimage_res, sat_image_up)

        # Guardar o devolver la imagen superresuelta
        if return_array:
            return fake_HR
        else:
            save_superresolved_image(fake_HR, profile, src.crs, folder_path)


def resample_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Reescala una imagen a un tamaño específico.
    """
    return (
        torch.nn.functional.interpolate(
            torch.tensor(image[None]).float(), size=size, mode="bilinear", align_corners=True, antialias=True
        )
        .squeeze()
        .numpy()
    )


def superresolve_image(s2: np.ndarray) -> np.ndarray:
    """
    Superresuelve una imagen Sentinel-2 utilizando un modelo ONNX.
    """
    srmodel = ort.InferenceSession("weights/sr_han.onnx")

    # Añadir padding de 16 píxeles
    matched_image_tors = np.pad(s2, ((0, 0), (16, 16), (16, 16)), mode="edge").astype(np.float32)

    # Ejecutar el modelo
    input_name = srmodel.get_inputs()[0].name
    sr_img = srmodel.run(None, {input_name: matched_image_tors[None]})[0].squeeze()

    # Remover el padding
    return sr_img[:, 64:-64, 64:-64]


def check_nan(inputs: list) -> bool:
    """
    Verifica si hay valores NaN en las entradas.
    """
    return any(np.isnan(inp).any() for inp in inputs)


def run_network(s2, super_img, satimage_res, sat_image_up):
    """
    Ejecuta la red neuronal con las imágenes procesadas.
    """
    path_weight_G = "./weights/RRSGAN_G.pth"
    network = RRSNet(ngf=64, n_blocks=16)
    load_net = torch.load(path_weight_G)
    network.load_state_dict(load_net, strict=False)
    network.eval()
    network.cpu()

    with torch.no_grad():
        fake_HR = network(
            torch.tensor(s2[None]).float(),
            torch.tensor(super_img[None]).float(),
            torch.tensor(satimage_res[None]).float(),
            torch.tensor(sat_image_up[None]).float(),
        )
    return fake_HR.squeeze().cpu().numpy()


def save_superresolved_image(fake_HR, profile, crs, folder_path):
    """
    Guarda la imagen superresuelta en un archivo GeoTIFF.

    Args:
        fake_HR (np.ndarray): Imagen superresuelta.
        profile (dict): Perfil de la imagen original.
        crs (str): Sistema de referencia de coordenadas.
        folder_path (pathlib.Path): Carpeta donde se guardará la imagen.
    """
    # Get transform from the profile
    transform = profile["transform"]
    new_a = transform.a /4
    new_e = transform.e /4
    new_transform = transform._replace(a=new_a, e=new_e)
    print(new_transform)
    profile.update(transform=new_transform, width=256, height=256, dtype=rio.float32)

    with rio.open(folder_path / "sr_image.tif", "w", **profile) as dst:
        dst.write(fake_HR, indexes=[1, 2, 3])