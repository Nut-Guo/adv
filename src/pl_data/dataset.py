import json
from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import os
from omegaconf import ValueNode
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.common.utils import PROJECT_ROOT


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
    def __init__(self, name: ValueNode, path: ValueNode, image_size: ValueNode, max_size: ValueNode=None,**kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.imgs = list(sorted(os.listdir(self.path)))
        if max_size:
            self.imgs = self.imgs[:max_size]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return {
            "image": img,
        }

    def __len__(self):
        return len(self.imgs)

    def __repr__(self) -> str:
        return f"PersonDataset({self.name}, {self.path})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: DHDDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    print(dataset[0]['image'].shape)


if __name__ == "__main__":
    main()
