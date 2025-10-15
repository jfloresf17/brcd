import ee
ee.Initialize(project='ee-jfloresf')
import io
import torch
import pyproj
import pathlib
import numpy as np
import pandas as pd
import rasterio as rio
import segmentation_models_pytorch as smp

from typing import List, Tuple
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

def get_s1_col(lat: float, 
                lon: float, 
                start_date: str, 
                end_date: str, 
                patch_size: int = 64,
                orbit_pass: str = 'DESCENDING'
                ) -> ee.ImageCollection:
    """
    Función para obtener una colección de imágenes de Sentinel-2 sin nubes.
    Parámetros:
        lat (float): Latitud del punto de interés.
        lon (float): Longitud del punto de interés.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        cloud_cover (float): Porcentaje de nubes permitido (0-1).
        patch_size (int): Tamaño del parche en metros.

    Devuelve:
        ee.ImageCollection: Colección de imágenes de Sentinel-2 sin nubes.
    """
    # Crear un punto y generar un buffer de 320 m
    geometry = ee.Geometry.Point(lon, lat)
    buffered_geometry = geometry.buffer(patch_size * 10 / 2) 

    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterBounds(buffered_geometry) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                    .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass)) \
                    .select(['VV', 'VH']) \
                    .map(lambda img: img.set("date", ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')))
    print(f"Original collection size: {s1_collection.size().getInfo()}")

    data_unique_per_day = s1_collection.distinct("date")
    print(f"Filtered collection size: {data_unique_per_day.size().getInfo()}")
    
    ## Get the Sentinel-2 image collection
    return data_unique_per_day


def get_s2cloudless_collection(lat: float, 
                             lon: float, 
                             start_date: str, 
                             end_date: str, 
                             clean_treshold: float = 0.8, 
                             patch_size: int = 64) -> ee.ImageCollection:
    """
    Función para obtener una colección de imágenes de Sentinel-2 sin nubes.
    Parámetros:
        lat (float): Latitud del punto de interés.
        lon (float): Longitud del punto de interés.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        cloud_cover (float): Porcentaje de nubes permitido (0-1).
        patch_size (int): Tamaño del parche en metros.

    Devuelve:
        ee.ImageCollection: Colección de imágenes de Sentinel-2 sin nubes.
    """

    # Threshold para cloudscore
    # threshold_cloud = 1 - cloud_cover # A nivel de pixel (default)
    # threshold_cloud = clean_cover # A nivel de pixel (default)

    # Crear un punto y generar un buffer de 320 m
    geometry = ee.Geometry.Point(lon, lat)
    buffered_geometry = geometry.buffer(patch_size * 10 / 2) 

    ## Use the image collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(buffered_geometry) \
                    .filterDate(start_date, end_date) \
                    .map(lambda img: img.set("date", ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')))

    s2_unique_per_day = s2_collection.distinct("date")
    print(f"Original collection size: {s2_unique_per_day.size().getInfo()}")

    ## Apply the cloud score
    csPlus_collection = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED') \
                .filterBounds(buffered_geometry) \
                .filterDate(start_date, end_date) \
                .select('cs_cdf')

    ## Join the cloud score to the image collection
    s2_with_cloud_score = ee.Join.saveFirst('cloud_score').apply(
        primary=s2_unique_per_day,
        secondary=csPlus_collection,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    )

    print(f"Collection with cloud score size: {s2_with_cloud_score.size().getInfo()}")

    # Filtrar imágenes con fracción de nubes menor al umbral
    def get_cloud_fraction(image):
        cloud_score = ee.Image(image.get('cloud_score'))
        cleanmask = cloud_score.gte(clean_treshold) 

        cleanPixelCount = cleanmask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffered_geometry,
            scale=10,
            maxPixels=1e13
        ).values().get(0)

        totalPixelCount = cloud_score.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=buffered_geometry,
            scale=10,
            maxPixels=1e13
        ).values().get(0)

        clean_fraction = ee.Algorithms.If(
                            ee.Number(totalPixelCount).gt(0),
                            ee.Number(cleanPixelCount).divide(totalPixelCount), # If True
                            0 # If False
                        )

        return image.set('clean_fraction', clean_fraction)


    s2_without_clouds = ee.ImageCollection(s2_with_cloud_score.map(get_cloud_fraction))

    # Obtener el tamaño de la colección filtrada
    s2_without_clouds = s2_without_clouds.filter(ee.Filter.gte('clean_fraction', clean_treshold))

    print(f"Filtered collection size: {s2_without_clouds.size().getInfo()}")
    return s2_without_clouds


def get_matching_s1_s2(s1_collection: ee.ImageCollection,
                       s2_collection: ee.ImageCollection) -> Tuple[List[ee.Image], List[ee.Image], List[str]]:
    """
    Matches Sentinel-2 images to the closest Sentinel-1 images (by available timestamps), keeping only those
    with equal image count per matched timestamp.
    """

    # STEP 1: Pull timestamps ONCE for both collections
    s1_times = s1_collection.aggregate_array("system:time_start").getInfo()
    s2_times = s2_collection.aggregate_array("system:time_start").getInfo()

    s1_dates = [pd.to_datetime(t, unit='ms') for t in s1_times]
    s2_dates = [pd.to_datetime(t, unit='ms') for t in s2_times]

    # STEP 2: Match each S2 date to the closest S1 date
    valid_pairs = []
    matching_dates_list = []

    for s2_dt in s2_dates:
        closest_s1_dt = min(s1_dates, key=lambda s1_dt: abs((s2_dt - s1_dt).total_seconds()))

        # Convert back to ms timestamp for filtering
        s1_ts = int(closest_s1_dt.timestamp() * 1000)
        s2_ts = int(s2_dt.timestamp() * 1000)

        valid_pairs.append((s1_ts, s2_ts))
        matching_dates_list.append(f"S1:{closest_s1_dt.date()}_S2:{s2_dt.date()}")

    # STEP 3: Create matching image lists without .getInfo() inside loop
    s1_images = [s1_collection.filter(ee.Filter.eq("system:time_start", pair[0])).first() for pair in valid_pairs]
    s2_images = [s2_collection.filter(ee.Filter.eq("system:time_start", pair[1])).first() for pair in valid_pairs]

    # STEP 4: Filter out any nulls (if .first() returns empty)
    s1_images_clean = []
    s2_images_clean = []
    matching_clean_dates = []

    for i in range(len(s1_images)):
        s1_img = s1_images[i]
        s2_img = s2_images[i]
        if s1_img is not None and s2_img is not None:
            s1_images_clean.append(s1_img)
            s2_images_clean.append(s2_img)
            matching_clean_dates.append(matching_dates_list[i])

    final_size = min(len(s1_images_clean), len(s2_images_clean))

    if final_size > 0:
        matching_final = matching_clean_dates[:final_size]

        print(f"Found {final_size} valid image pairs.")
        print(f"Matching dates: {matching_final}")

        return list(s1_images_clean[:final_size]), list(s2_images_clean[:final_size]), matching_final

    print("No valid image pairs found.")
    return [], [], []


def get_images(lat: float,
               lon: float,
               image_list: List[ee.Image],
               bands: List[str] = None,
               patch_size: int = 64,
               include_clouds: bool = False) -> Tuple[list, dict]:
    """
    Descarga parches de imágenes de Sentinel a partir de una lista de ee.Image.
    """

    # Get timestamps for all images
    dates = [img.get("system:time_start").getInfo() for img in image_list]
    dates_str = [
        pd.to_datetime(date, unit="ms").strftime("%Y-%m-%d %H:%M:%S") for date in dates
    ]

    # Optionally get cloud_fraction from each image
    if include_clouds:
        cloud_fractions = [1 - (float(img.get("clean_fraction").getInfo())) for img in image_list]
        print(f"cloud_fractions: {cloud_fractions}")

    # Get UTM CRS using pyproj
    utm_list = query_utm_crs_info(datum_name='WGS 84', area_of_interest=AreaOfInterest(lon, lat, lon, lat))
    utm_crs = utm_list[0].code
    print(f"UTM CRS: {utm_crs}")

    transformer = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{utm_crs}', always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    scale_x, scale_y = 10, 10
    ul_x = center_x - scale_x * patch_size / 2
    ul_y = center_y + scale_y * patch_size / 2

    length = len(image_list)
    indices_to_download = list(range(length))

    # List to hold image data
    image_arrays_list = []

    for i in indices_to_download:
        image = image_list[i]

        request = {
            'expression': image,
            'fileFormat': 'GEO_TIFF',
            'bandIds': bands,
            'grid': {
                'dimensions': {
                    'width': patch_size,
                    'height': patch_size
                },
                'affineTransform': {
                    'scaleX': scale_x,
                    'shearX': 0,
                    'translateX': ul_x,
                    'shearY': 0,
                    'scaleY': -scale_y,
                    'translateY': ul_y
                },
                'crsCode': f'EPSG:{utm_crs}'
            }
        }

        image_png = ee.data.computePixels(request)

        with rio.open(io.BytesIO(image_png), 'r') as src:
            image_array = src.read()
            profile = src.profile

            image_arrays_list.append({
                "image": image_array,
                "date": dates_str[i],
                "cloud_fraction": cloud_fractions[i] if include_clouds else None,
            })

    return image_arrays_list, profile


def cloud_model(cloud_model_path: str, data: np.ndarray) -> np.ndarray:
    """
    Cloud model to detect clouds in the image

    Args:
        cloud_model_path (str): Path to the cloud model.
        data (np.ndarray): Image data.
    
    Returns:
        np.ndarray: Returns the mask with the clouds.
    """

    # Load the weights into the model
    model_v2 = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, classes=4, in_channels=13)
    model_v2.load_state_dict(torch.load(cloud_model_path, map_location=torch.device('cpu')))

    # Desactivate the gradient estimation
    for param in model_v2.parameters():
        param.requires_grad = False

    # To eval model
    model_v2 = model_v2.eval()

    # Get the data
    data_torch = torch.from_numpy(data)[None]

    logits = model_v2(data_torch.float())
    output = logits.argmax(dim=1).squeeze().numpy().astype(np.uint8)

    return output
