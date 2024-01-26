import os, tqdm, cv2, rasterio
import torch
from pathlib import Path
from glob import glob

import torch
from torchvision.io import read_image
from torchvision import transforms as T


__all__ = [
    "LevirCDDataset",
    "LevirCDPlusDataset",
    "CD3DDataset",
    "make_cropped_dataset",
    "make_cropped_tiff_dataset",
]


def _normalize_cfg(partition = "train"):
    dataset_mean_std = {
        "train": {
            "A_128_128": {
                "mean": [0.45057096281223236, 0.4470393464069983, 0.3815948267089663],
                "std": [0.21715168747967356, 0.20408517208762272, 0.1867427639490323],
            },
            "B_128_128": {
                "mean": [0.3463029653739229, 0.3389253688082883, 0.2895384042782596],
                "std": [0.15800775701640743, 0.15240661814615453, 0.14486230966885502],
            }
        },
    }
    return (
        T.Normalize(
            torch.Tensor(dataset_mean_std[partition]["A_128_128"]["mean"]),
            torch.Tensor(dataset_mean_std[partition]["A_128_128"]["std"])
        ), 
        T.Normalize(
            torch.Tensor(dataset_mean_std[partition]["B_128_128"]["mean"]),
            torch.Tensor(dataset_mean_std[partition]["B_128_128"]["std"])
        ),
    ) if partition in dataset_mean_std.keys() else None


def make_cropped_dataset(ds_path, crop_size = (128, 128), stride = (64, 64), img_format="png"):
    for folder in ('A', 'B', 'label'):
        img_folder = str(ds_path / folder)
        cropping_folder_path = f"{ds_path}/{folder}_{crop_size[0]}_{crop_size[1]}"
        os.makedirs(cropping_folder_path, exist_ok=True)
        print(f"Working on {folder} images")
        for img_path in tqdm.tqdm(glob(img_folder + f"/*.{img_format}")):
            img = cv2.imread(img_path)
            org_width, org_height, _ = img.shape
            for x in range(0, org_width, stride[0]):
                cropped_width = x + crop_size[0]
                for y in range(0, org_height, stride[1]):
                    cropped_height = y + crop_size[1]
                    if cropped_width < org_width and cropped_height < org_height:
                        cropped_img = img[x:cropped_width, y:cropped_height, :]
                        basename = os.path.basename(img_path).split(".")[0]
                        cv2.imwrite(
                            f"{cropping_folder_path}/{basename}_{x}_{cropped_width}_{y}_{cropped_height}.{img_format}",
                            cropped_img
                        )


def make_cropped_tiff_dataset(ds_path, crop_size = (128, 128), stride = (64, 64), img_format="tif"):
    for folder in ('2010', '2017', '2D', '3D'):
        img_folder = str(ds_path / folder)
        cropping_folder_path = f"{ds_path}/{folder}_{crop_size[0]}_{crop_size[1]}"
        os.makedirs(cropping_folder_path, exist_ok=True)
        print(f"Working on {folder} TIFF images")
        handler = imageio.v3 if folder != "3D" else tiff
        for img_path in tqdm.tqdm(glob(img_folder + f"/*.{img_format}")):
            img = handler.imread(img_path)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
              
            org_width, org_height, _ = img.shape
            for x in range(0, org_width, stride[0]):
                cropped_width = x + crop_size[0]
                for y in range(0, org_height, stride[1]):
                    cropped_height = y + crop_size[1]
                    if cropped_width < org_width and cropped_height < org_height:
                        basename = os.path.basename(img_path).split(".")[0]
                        with rasterio.open(
                                f"{cropping_folder_path}/{basename}_{x}_{cropped_width}_{y}_{cropped_height}.tif", 
                                "w", **profile
                            ) as dst:
                            for k in range(0, bands):
                                dst.write(img[k, y:cropped_height, x:cropped_width], indexes=k+1)
                        


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]) 
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor


class LevirCDDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, partition = "train", crop_size: tuple = (128, 128), limit: int = None, img_format: str = "png"):
        assert partition in ("train", "val", "test")
        self.imgs_path = {}
        self.len_imgs = -1
        self.scaler = PyTMinMaxScalerVectorized()

        for folder in ("A", "B", "label"):
            path = str(ds_path / partition / f"{folder}_{crop_size[0]}_{crop_size[1]}")
            self.imgs_path[folder] = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(f".{img_format}")]
            self.imgs_path[folder].sort()
            if limit:
                self.imgs_path[folder] = self.imgs_path[folder][:limit]

            if self.len_imgs == -1:
                self.len_imgs = len(self.imgs_path[folder])
            else:
                assert self.len_imgs == len(self.imgs_path[folder])

    def __len__(self):
        return self.len_imgs
    
    def __getitem__(self, index):
        """
        Returns stacked A and B and the label image as well
        """
        img1 = read_image(self.imgs_path["A"][index]) / 255.
        img2 = read_image(self.imgs_path["B"][index]) / 255.
        img = torch.cat([img1, img2], axis=0)
        # img = torch.concat(
        #     [self.scaler(img1.float()), self.scaler(img2.float())], 
        #     axis=0
        # )
        label = T.Grayscale()(read_image(self.imgs_path["label"][index])) / 255.
        label = torch.concat([1-label, label], axis=0)
        return img, label

class LevirCDPlusDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, partition = "train", crop_size: tuple = (128, 128), limit: int = None, img_format: str = "png"):
        assert partition in ("train", "val", "test")
        self.imgs_path = {}
        self.len_imgs = -1
        self.scaler = PyTMinMaxScalerVectorized()

        for folder in ("A", "B", "label"):
            path = str(ds_path / partition / f"{folder}_{crop_size[0]}_{crop_size[1]}")
            self.imgs_path[folder] = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(f".{img_format}")]
            self.imgs_path[folder].sort()
            if limit:
                self.imgs_path[folder] = self.imgs_path[folder][:limit]

            if self.len_imgs == -1:
                self.len_imgs = len(self.imgs_path[folder])
            else:
                assert self.len_imgs == len(self.imgs_path[folder])

    def __len__(self):
        return self.len_imgs
    
    def __getitem__(self, index):
        """
        Returns stacked A and B and the label image as well
        """
        img1 = read_image(self.imgs_path["A"][index]) / 255.
        img2 = read_image(self.imgs_path["B"][index]) / 255.
        img = torch.cat([img1, img2], axis=0)
        # img = torch.concat(
        #     [self.scaler(img1.float()), self.scaler(img2.float())], 
        #     axis=0
        # )
        label = T.Grayscale()(read_image(self.imgs_path["label"][index])) / 255.
        label = torch.concat([1-label, label], axis=0)
        return img, label



class CD3DDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, partition = "train", crop_size: tuple = (128, 128), limit: int = None, img_format: str = "png"):
        assert partition in ("train", "val", "test")
        self.imgs_path = {}
        self.len_imgs = -1
        self.scaler = PyTMinMaxScalerVectorized()

        for folder in ("A", "A_time", "B", "B_time", "label"):
            path = str(ds_path / partition / f"{folder}_{crop_size[0]}_{crop_size[1]}")
            self.imgs_path[folder] = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(f".{img_format}")]
            self.imgs_path[folder].sort()
            if limit:
                self.imgs_path[folder] = self.imgs_path[folder][:limit]

            if self.len_imgs == -1:
                self.len_imgs = len(self.imgs_path[folder])
            else:
                assert self.len_imgs == len(self.imgs_path[folder])

    def __len__(self):
        return self.len_imgs
    
    def __getitem__(self, index):
        """
        Returns stacked A and B and the label image as well
        """
        img1 = read_image(self.imgs_path["A"][index]) / 255.
        img2 = read_image(self.imgs_path["B"][index]) / 255.
        img = torch.cat([img1, img2], axis=0)
        # img = torch.concat(
        #     [self.scaler(img1.float()), self.scaler(img2.float())], 
        #     axis=0
        # )
        label = imageio.v3.imread(self.imgs_path["2D"][index]) / 255.
        label = torch.from_numpy(label)
        label = torch.concat([1-label, label], axis=0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        img = torch.cat([img1, img2], axis=0)
        
        return img, label
