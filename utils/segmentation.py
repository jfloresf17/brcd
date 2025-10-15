import pathlib
import re
import numpy as np
import rasterio as rio
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
import io
import base64
import segmentation_models_pytorch as smp
from typing import List
from torchvision import transforms


def preprocess_image_for_inference(image: np.ndarray, 
                                   normalize: bool=False, 
                                   mean: List[float]=None, 
                                   std: List[float]=None
    ) -> torch.Tensor:
    image = image.transpose(1,2,0)
    preprocess_pipeline = transforms.Compose([transforms.ToTensor()])

    if normalize and mean is not None and std is not None:
        preprocess_pipeline.transforms.append(transforms.Normalize(mean=mean, std=std))
    image = preprocess_pipeline(image).float()
    image = image.unsqueeze(0)
    return image


def onnx_inference_segmentation(onnx_model_path: str, image: np.ndarray, mean: List[float], std: List[float], normalize: bool=True) -> np.ndarray:
    # image = preprocess_image_for_inference(image, normalize=normalize, mean=mean, std=std).numpy()
    # session = ort.InferenceSession(onnx_model_path)
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # result = session.run([output_name], {input_name: image})[0]
    # return result
    # Load the Student model
    model = smp.Unet(encoder_name="mit_b1", in_channels=3, classes=1)
    checkpoint = torch.load(onnx_model_path, map_location=torch.device("cpu"))
    filter_ckpt = {k: v for k, v in checkpoint.items()}
    model.load_state_dict(filter_ckpt)
    model = model.cpu()
    model.eval()

    # Load the image
    image = preprocess_image_for_inference(image, normalize=normalize, 
                                           mean=mean, std=std).cpu()
    
    # Inference
    with torch.no_grad():
        logit = model(image)
        probs = torch.sigmoid(logit)

    output = probs.squeeze().numpy()
    
    return output


def save_image_to_base64(image_array, save_path, cmap=None):
    plt.imshow(image_array, cmap=cmap)
    plt.axis("off")

    # Guardar imagen en disco
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Convertir imagen a Base64
    buf = io.BytesIO()
    plt.imsave(buf, image_array, cmap=cmap, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def get_segmentation(folder_path: pathlib.Path):
    building_mean = [0.6105628,  0.56201386, 0.4756266]
    building_std = [0.13386504, 0.1379912,  0.15555547]

    road_mean = [0.6039244,  0.57160616, 0.48706716]
    road_std = [0.12059106, 0.11974796, 0.13146031]

    images_plot_b = []
    images_plot_r = []
    dates_imgs = []

    for sr_path in list(folder_path.glob("sr*.tif")):
        date_str = sr_path.name.split("_")[1].split(".")[0]
        with rio.open(sr_path) as src:
            image = src.read()
            # building_output = onnx_inference_segmentation("weights/buildings.onnx", image, building_mean, building_std)
            # road_output = onnx_inference_segmentation("weights/roads.onnx", image, road_mean, road_std)
            building_output = onnx_inference_segmentation("/home/tidop/projects/srseg/checkpoints/mitb1_building_wcedice_unet_best_model.pth", image, building_mean, building_std)
            road_output = onnx_inference_segmentation("/home/tidop/projects/srseg/checkpoints/mitb1_roads_ce_unet_best_model.pth", image, road_mean, road_std)

            # Save the images
            img_np_b = np.nan_to_num(building_output, nan=0.0)
            s2_img_b64_b = save_image_to_base64(img_np_b, folder_path / f"b_{date_str}.png")

            img_np_r = np.nan_to_num(road_output, nan=0.0)
            s2_img_b64_r = save_image_to_base64(img_np_r, folder_path / f"r_{date_str}.png")

            images_plot_b.append(s2_img_b64_b)
            images_plot_r.append(s2_img_b64_r)
            dates_imgs.append(date_str)

            # Export the images
            profile = {}
            transform = rio.transform.from_bounds(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, src.width, src.height)
            profile.update(crs=src.crs, transform=transform, driver="GTiff", width=src.width, height=src.height, count=1, dtype=rio.float32)

            with rio.open(folder_path / f"b_{date_str}.tif", "w", **profile) as dst:
                dst.write(building_output.squeeze(), indexes=1)

            with rio.open(folder_path / f"r_{date_str}.tif", "w", **profile) as dst:
                dst.write(road_output.squeeze(), indexes=1)
    
    return { "images_b": images_plot_b, "images_r": images_plot_r, "dates": dates_imgs }
