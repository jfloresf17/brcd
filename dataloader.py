import torch
import pathlib
import numpy as np
import rasterio as rio
from torchvision import transforms
import pytorch_lightning as pl

# from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# For building or road segmentation tasks, we define a single-task dataset:
# For multi-task segmentation (e.g. building + road), we define a dataset that returns a dict of masks:
class MultiTaskDataset(Dataset):
    """
    Dataset multi-task (building + road).  
    Devuelve (img_tensor, bld_mask, road_mask).
    """
    def __init__(
        self,
        s1_list, s2_list,
        building_list, road_list,
        aug=None,
        normalize=False, mean=None, std=None,
    ):
        self.s1, self.s2 = sorted(s1_list), sorted(s2_list)
        self.buildings, self.roads = sorted(building_list), sorted(road_list)
        self.aug = aug

        # construcción de normalización con PyTorch para after-augment
        if normalize:
            self.norm = transforms.Normalize(mean=mean, std=std)
        else:
            self.norm = None

    def __len__(self):
        return len(self.s2)

    def __getitem__(self, idx):
        # 1) Leer imágenes
        with rio.open(self.s1[idx]) as src1, rio.open(self.s2[idx]) as src2:
            img_s1 = src1.read().astype('float32')       # (2,256,256)
            img_s2 = (src2.read() / 10_000).astype('float32')  # (6,256,256)
        img = np.concatenate([img_s2, img_s1], axis=0)      # (8,256,256)

        # 2) Leer máscaras
        with rio.open(self.buildings[idx]) as sb:
            bld = sb.read(1).astype('uint8')
        with rio.open(self.roads[idx]) as sr:
            road = sr.read(1).astype('uint8')

        # # 3) Data augmentation
        # if self.aug:
        #     # Combinar img y masks en un diccionario para Albumentations
        #     augmented = self.aug(
        #         image=img.transpose(1,2,0),   # Alb expects H×W×C
        #         bld_mask=bld,
        #         road_mask=road
        #     )
        #     img = augmented['image'].transpose(2,0,1)  # de vuelta a C×H×W
        #     bld = augmented['bld_mask']
        #     road = augmented['road_mask']

        # 4) Normalización post-augment
        if self.norm:
            img = self.norm(torch.from_numpy(img))

        return img, torch.from_numpy(bld), torch.from_numpy(road)


# For the LightningDataModule that handles both single- and multi-task datasets:
class SegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        ds = cfg['dataset']
        self.batch_size  = cfg['training']['batch_size']
        self.num_workers = cfg['training']['num_workers']
        self.normalize   = cfg['normalize']['apply']
        self.mean        = cfg['normalize']['mean']
        self.std         = cfg['normalize']['std']
        self.val_size    = ds['val_size']
        self.seed        = ds['seed']

        # Paths
        self.s2_path = pathlib.Path(ds['dataroot_S2'])
        self.s1_path = pathlib.Path(ds['dataroot_S1'])
        self.b_path  = pathlib.Path(ds['dataroot_BU'])
        self.r_path  = pathlib.Path(ds['dataroot_RO'])

    def setup(self, stage=None):
        # 1) Obtener listas y emparejar stems
        all_s2 = sorted(self.s2_path.glob('*.tif'))
        all_s1 = sorted(self.s1_path.glob('*.tif'))
        all_b  = sorted(self.b_path.glob('*.tif'))
        all_r  = sorted(self.r_path.glob('*.tif'))
        stems = set(p.stem.split("_")[1] for p in all_s2) & \
                set(p.stem.split("_")[1] for p in all_s1) & \
                set(p.stem.split("_")[1] for p in all_b) & \
                set(p.stem.split("_")[1] for p in all_r)
        # Filtrar listas
        def filter_by(stems, lst):
            return [p for p in lst if p.stem.split("_")[1] in stems]
        all_s2 = filter_by(stems, all_s2)
        all_s1 = filter_by(stems, all_s1)
        all_b  = filter_by(stems, all_b)
        all_r  = filter_by(stems, all_r)

        # 2) Train/val split
        (self.tr_s2, self.val_s2,
         self.tr_s1, self.val_s1,
         self.tr_b,  self.val_b,
         self.tr_r,  self.val_r) = train_test_split(
            all_s2, all_s1, all_b, all_r,
            test_size=self.val_size,
            random_state=self.seed,
            shuffle=True
        )

        # 3) Crear augmentations (70% geométricos, 30% crop centrado)
        # self.train_aug = A.Compose([
        # A.OneOf([
        #     # Crop centrado en edificio/carret (implementado con RandomCrop+PadIfNeeded)
        #     A.Sequential([
        #         A.RandomCrop(128, 128, p=1.0),
        #         A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, p=1.0),
        #     ], p=0.3),
        #     # Crop random más suave: extrae patch 192×192 y pad al tamaño original
        #     A.Sequential([
        #         A.RandomCrop(192, 192, p=1.0),
        #         A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, p=1.0),
        #     ], p=0.3),
        #     # No crop
        #     A.NoOp(p=0.4),
        # ], p=1.0),
        # # Distorsiones geométricas y fotométricas
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.Affine(rotate=30, translate_percent=0.05, scale=(0.9,1.1), p=0.5),
        # A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        # A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.GaussianNoise(p=0.3),
        # A.OneOf([
        #     A.MultiplicativeNoise(multiplier=(0.8,1.2), p=0.5),
        #     A.GlassBlur(sigma=1.0, max_delta=2, p=0.2),
        # ], p=0.3),
        # ], additional_targets={'bld_mask':'mask', 'road_mask':'mask'}, 
        # # deshabilitamos chequeo de shapes para mayor flexibilidad
        # is_check_shapes=False
        # )
            
        # 4) Construir datasets
        self.train_ds = MultiTaskDataset(
            self.tr_s1, self.tr_s2,
            self.tr_b,  self.tr_r,
            # aug=self.train_aug,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std
        )
        self.val_ds = MultiTaskDataset(
            self.val_s1, self.val_s2,
            self.val_b,   self.val_r,
            # aug=None,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std
        )

        # # 5) Pre-calculemos patch_perc para el sampler
        # patch_perc = []
        # for _, bld_mask, road_mask in self.train_ds:
        #     # % building + road en el parche
        #     frac = ((bld_mask>0).float() + (road_mask>0).float()).mean().item()
        #     patch_perc.append(frac)
        # patch_perc = torch.tensor(patch_perc, dtype=torch.float32)

        # # Pesos inversos normalizados
        # ε = 1e-3
        # raw_w = 1.0 / (patch_perc + ε)
        # self.patch_weights = raw_w / raw_w.sum()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(
    #         weights=self.patch_weights,
    #         num_samples=len(self.patch_weights),
    #         replacement=True
    #     )
    #     return DataLoader(
    #         self.train_ds,
    #         batch_size=self.batch_size,
    #         sampler=sampler,
    #         num_workers=self.num_workers,
    #         pin_memory=True
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True
    #     )
    
# class SegDataModule(pl.LightningDataModule):
#     """
#     LightningDataModule handling single- and multi-task segmentation.
#     """
#     def __init__(
#         self,
#         cfg: dict
#     ):
#         super().__init__()
#         ds = cfg['dataset']
#         seed = ds['seed']
#         self.normalize = cfg['normalize']['apply']
#         self.mean      = cfg['normalize']['mean']
#         self.std       = cfg['normalize']['std']
#         self.batch_size  = cfg['training']['batch_size']
#         self.num_workers = cfg['training']['num_workers']

#         # Split dataset into train, val, test using sklearn's train_test_split
#         building_path = pathlib.Path(ds['dataroot_BU'])
#         road_path = pathlib.Path(ds['dataroot_RO'])
#         s2_path = pathlib.Path(ds['dataroot_S2'])
#         s1_path = pathlib.Path(ds['dataroot_S1'])


#         all_s2 = sorted(list(s2_path.glob('*.tif')))
#         all_s1 = sorted(list(s1_path.glob('*.tif')))
#         all_buildings = sorted(list(building_path.glob('*.tif')))
#         all_roads = sorted(list(road_path.glob('*.tif')))

#         # Match the counts of all datasets, get the minimum length
#         common_stems = set(p.stem.split("_")[1] for p in all_s2) & \
#                set(p.stem.split("_")[1] for p in all_s1) & \
#                set(p.stem.split("_")[1] for p in all_buildings) & \
#                set(p.stem.split("_")[1] for p in all_roads)
        
#         all_s2 = [p for p in all_s2 if p.stem.split("_")[1] in common_stems]
#         all_s1 = [p for p in all_s1 if p.stem.split("_")[1] in common_stems]
#         all_buildings = [p for p in all_buildings if p.stem.split("_")[1] in common_stems]
#         all_roads = [p for p in all_roads if p.stem.split("_")[1] in common_stems]

#         assert len(all_s2) == len(all_s1) == len(all_buildings) == len(all_roads), \
#             "Counts of Sentinel-2, Sentinel-1, building masks, and road masks must match"
        
#         # Split into train, val, test
#         self.train_s2, self.val_s2, self.train_s1, self.val_s1, self.train_bld, self.val_bld, self.train_road, self.val_road = \
#             train_test_split(all_s2, all_s1, all_buildings, all_roads, test_size=ds['val_size'], random_state=seed, shuffle=True)
        

#     def setup(self, stage=None):
#         # choose dataset class
#         self.train_dataset = MultiTaskDataset(
#             self.train_s1,
#             self.train_s2,
#             self.train_bld,
#             self.train_road,
#             normalize=self.normalize,
#             mean=self.mean,
#             std=self.std
#         )
#         self.val_dataset = MultiTaskDataset(
#             self.val_s1,
#             self.val_s2,
#             self.val_bld,
#             self.val_road,
#             normalize=self.normalize,
#             mean=self.mean,
#             std=self.std
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )
