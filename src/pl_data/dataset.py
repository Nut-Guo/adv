import json
from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import json
import numpy as np
import torch
import os
from omegaconf import ValueNode
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.common.utils import PROJECT_ROOT
import albumentations as A
from albumentations.pytorch import ToTensor
import cv2


class DHDDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, image_size: ValueNode, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        with open(os.path.join(self.path, 'dhd_coco.json')) as f:
            self.data = json.load(f)
        self.images = list(sorted(os.listdir(os.path.join(self.path, "Images"))))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        img_name = self.images[index]
        img_path = os.path.join(self.path, "Images", img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return {
            "image": img,
            "bbox": self.data[img_name]
        }

    def __repr__(self) -> str:
        return f"DHDDataset({self.name}, {self.path})"


class PersonDataset(object):
    def __init__(self, name: ValueNode, path: ValueNode, image_size: ValueNode, max_size: ValueNode = None, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.image_size = image_size
        annotations = path + '_annotations.json'
        with open(annotations, 'r') as f:
            self.anno = json.load(f)
        self.imgs = list(self.anno.keys())
        # self.imgs = list(sorted(os.listdir(self.path)))
        if max_size:
            self.imgs = self.imgs[:max_size]
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((image_size, image_size)),
        # ])
        self.transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(),
            A.RandomBrightness(),
            # A.ShiftScaleRotate(),
            # A.Resize(image_size, image_size)
            A.RandomSizedBBoxSafeCrop(image_size, image_size),
            ToTensor()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['probs']))

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_path = os.path.join(self.path, name)
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = np.array(self.anno[name]['boxes']).clip(0, 1).tolist()
        confidence = self.anno[name]['confidence']
        # boxes = boxes.clamp(0, 1) * self.image_size
        # confidence = torch.Tensor(self.anno[name]['confidence'])
        # if self.transforms is not None:
        #     img = self.transforms(img)
        imgs = []
        bboxes = []
        probs = []
        for i in range(8):
            transformed = self.transforms(image=img, bboxes=boxes, probs=confidence)
            imgs.append(transformed["image"])
            bboxes.append(torch.tensor(transformed["bboxes"]) * self.image_size)
            probs.append(torch.tensor(transformed["probs"]))
        # print(transformed)
        return {
            "image": torch.stack(imgs),#transformed["image"],
            "boxes": torch.stack(bboxes),#torch.tensor(transformed["bboxes"]) * self.image_size,
            "classprobs": torch.stack(probs)#torch.tensor(transformed["probs"])
        }

    def __len__(self):
        return len(self.imgs)

    def __repr__(self) -> str:
        return f"PersonDataset({self.name}, {self.path})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: DHDDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, 
    )
    print(dataset[0]['image'].shape)


if __name__ == "__main__":
    main()
