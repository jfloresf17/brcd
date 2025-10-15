# from typing import List
# import torch
# import numpy as np
# from torchvision import transforms
# import segmentation_models_pytorch as smp

# def preprocess_image_for_inference(image: np.ndarray, 
#                                    normalize: bool=False, 
#                                    mean: List[float]=None, 
#                                    std: List[float]=None
#     ) -> torch.Tensor:
#     """
#     Preprocess an image for inference with a PyTorch model.

#     Args:
#     image (np.ndarray): Input image as a numpy array.
#     normalize (bool): If True, normalize the image. Default: False.
#     mean (List[float]): List with the mean values for normalization.
#     std (List[float]): List with the standard deviation values for normalization.

#     Returns:
#     torch.Tensor: Preprocessed image as a PyTorch tensor.
#     """

#     image = image.transpose(1,2,0)

#     # Define la pipeline de preprocesamiento
#     preprocess_pipeline = transforms.Compose([
#         transforms.ToTensor()  # Convierte la imagen a tensor
#     ])

#     if normalize and mean is not None and std is not None:
#         preprocess_pipeline.transforms.append(transforms.Normalize(mean=mean, std=std))

#     # Aplica el preprocesamiento
#     image = preprocess_pipeline(image).float()
#     image = image.unsqueeze(0)  # Añade la dimensión de batch

#     return image


# def br_inference(checkpoint_path: str, 
#                     normalize: bool, 
#                     mean: List[float],
#                     std: List[float], 
#                     image: np.ndarray) -> np.ndarray:

#     """
#     Inference with a pre-trained segmentation model(for building and roads) for a given image.

#     Args:
#     checkpoint_path (str): Path to the model checkpoint.
#     normalize (bool): If True, normalize the image.
#     mean (List[float]): List with the mean values for normalization.
#     std (List[float]): List with the standard deviation values for normalization.
#     image (np.ndarray): Input image as a numpy array.

#     Returns:
#     np.ndarray: Output of the model as a probability map.
#     """

#     # Load the Student model
#     model = smp.Unet(encoder_name="mit_b1", in_channels=3, classes=1)
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#     filter_ckpt = {k: v for k, v in checkpoint.items()}
#     model.load_state_dict(filter_ckpt)
#     model = model.cpu()
#     model.eval()

#     # Load the image
#     image = preprocess_image_for_inference(image, normalize=normalize, 
#                                            mean=mean, std=std).cpu()
    
#     # Inference
#     with torch.no_grad():
#         logit = model(image)
#         probs = torch.sigmoid(logit)

#     output = probs.squeeze().numpy()
    
#     return output

# import cv2
# from super_image import HanModel
# from models.RRSNet import RRSNet

# def rrsgan_inference(s2_arr, s2_profile):
#     with rio.open("../satellite.tif") as src:
#         transform = s2_profile['transform']
#         print(f"Transform: {transform}")
#         # Calculate center coordinates
#         ulx, uly = transform.c, transform.f
#         center_x = ulx + (s2_profile['width'] * transform.a) / 2
#         center_y = uly - (s2_profile['height'] * transform.a) / 2

#         print(f"Center coordinates: {center_x}, {center_y}")

#         # Get the bounds considering a patch size of 64 pixels and a resolution of 10m
#         patch_size = 64
#         s2_minx = center_x - (patch_size * transform.a) / 2
#         s2_miny = center_y - (patch_size * transform.a) / 2
#         s2_maxx = center_x + (patch_size * transform.a) / 2
#         s2_maxy = center_y + (patch_size * transform.a) / 2

#         print(f"Bounds: {s2_minx}, {s2_miny}, {s2_maxx}, {s2_maxy}")

#         window = rio.windows.from_bounds(s2_minx, s2_miny, s2_maxx, s2_maxy, src.transform)

#         ## Read the satellite image
#         satimage = src.read(window=window)

#         ## Resample the image to get a pixel of 2.5m
#         satimage_res = cv2.resize(satimage.transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_LINEAR)
#         satimage_res = satimage_res.transpose(2, 0, 1) / 255.

#         ## Downsample the image to 64x64
#         sat_image_down = cv2.resize(satimage_res.transpose(1, 2, 0), (64, 64), interpolation=cv2.INTER_LINEAR)
#         sat_image_down = sat_image_down.transpose(2, 0, 1)

#         ## Upsample the image to 256x256
#         sat_image_up = cv2.resize(sat_image_down.transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_LINEAR)
#         sat_image_up = sat_image_up.transpose(2, 0, 1)

#         ## Read s2 image
#         s2 = s2_arr[[3, 2, 1]] / 10000
#         s2 = np.clip(s2, 0, 1)

#         ## Superresolve the image
#         srmodel = HanModel.from_pretrained('eugenesiow/han', scale=4)
#         srmodel.eval()

#         ## Add a padding of 16 pixels
#         matched_image_tors = np.pad(s2, ((0,0),(16, 16),(16, 16)), mode="edge")
#         matched_image_tors = matched_image_tors

#         image_torch = torch.from_numpy(matched_image_tors).float()

#         ## Apply the super resolution
#         with torch.no_grad():
#             sr_img = srmodel(image_torch[None]).squeeze()

#         ## Remove the padding
#         super_img = sr_img.numpy()
#         super_img = super_img[:,64:-64,64:-64]


#         path_weight_G = "/home/tidop/projects/RefImSR/RRSGAN/exp/experiments/023_RRSGAN/experiments/023_RRSGAN/models/latest_G.pth"
#         network = RRSNet(ngf=64, n_blocks=16)
#         load_net = torch.load(path_weight_G)
#         network.load_state_dict(load_net, strict=False)
#         network.eval()
#         network.cpu()    
        
#         ## Run the network
#         with torch.no_grad():
#             fake_HR = network(torch.tensor(s2[None]).float(), torch.tensor(super_img[None]).float(), torch.tensor(satimage_res[None]).float(), 
#                               torch.tensor(sat_image_up[None]).float())
#         fake_HR = fake_HR.squeeze().cpu().numpy()

#     building_mean = [0.6105628,  0.56201386, 0.4756266]
#     building_std = [0.13386504, 0.1379912,  0.15555547]

#     road_mean = [0.6039244,  0.57160616, 0.48706716]
#     road_std = [0.12059106, 0.11974796, 0.13146031]

#     buiulding_ckpt_path = "/home/tidop/projects/srseg/checkpoints/mitb1_building_wcedice_unet_best_model.pth"
#     road_ckpt_path = "/home/tidop/projects/srseg/checkpoints/mitb1_roads_ce_unet_best_model.pth"

#     building_output = br_inference(buiulding_ckpt_path, True, building_mean, building_std, fake_HR)
#     road_output = br_inference(road_ckpt_path, True, road_mean, road_std, fake_HR)

#     background_output = np.clip(1.0 - (building_output + road_output), 0, 1)

#     rgb_pred = np.stack([building_output, road_output, background_output], axis=0)  # (3, H, W)
#     return rgb_pred

# rrsgan_arrays = []
# for i, s2_arr in enumerate(s2_images):
#     rgb_pred = rrsgan_inference(s2_arr["image"], s2_profile)
#     rrsgan_arrays.append({
#         "date": s2_arr["date"],
#         "pred": rgb_pred,
#         "profile": s2_profile
#     })
#     print(f"Processed RRS image {i+1}/{len(s2_images)}: {s2_arr['date']}")
    
# import geopandas as gpd
# import pyproj
# import rasterio as rio
# from shapely.ops import transform
# from rasterio.transform import rowcol

# # Plot all predictions with ncol = 10
# # Plot all S2 images with ncol = 10
# ncol = 10
# nrow = (len(s2_images) + ncol - 1) // ncol

# transform_full = out_arrays[0]['profile']['transform']
# # Change the resolution to 2.5
# transform_full = rio.Affine(transform_full.a / 4, transform_full.b, transform_full.c, 
#                             transform_full.d, transform_full.e / 4, transform_full.f)


# # Explode and reproject geometry
# shape = gpd.GeoSeries(shape, crs="EPSG:4326").explode(index_parts=False).reset_index(drop=True)
# transformer = pyproj.Transformer.from_crs("EPSG:4326", out_arrays[0]['profile']['crs'], always_xy=True).transform
# reprojected_shape = shape.apply(lambda geom: transform(transformer, geom))

# fig, axes = plt.subplots(nrow, ncol, figsize=(20, 10))
# for i, ax in enumerate(axes.flat):
#     if i < len(s2_images):
#         ax.imshow(rrsgan_arrays[i]["pred"].transpose(1, 2, 0))
#         for geom in reprojected_shape.geometry:
#             if geom.geom_type == 'Polygon':
#                 x, y = geom.exterior.xy
#                 rows, cols = rowcol(transform_full, x, y)
#                 ax.plot(cols, rows, color='black', linewidth=1)
#             elif geom.geom_type == 'MultiPolygon':
#                 for poly in geom.geoms:
#                     x, y = poly.exterior.xy
#                     rows, cols = rowcol(transform_full, x, y)
#                     ax.plot(cols, rows, color='black', linewidth=1)
#         ax.set_title(rrsgan_arrays[i]["date"].split(" ")[0] + f"[{str(i)}]")
#         ax.axis('off')
#     else:
#         ax.axis('off')