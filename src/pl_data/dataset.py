import json
from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import os
from omegaconf import ValueNode
from torch.utils.data import Dataset
from PIL import Image
from src.common.utils import PROJECT_ROOT

class DHDDataset(object):
    def __init__(self, root = 'data/DHD', transforms = None):
        self.root = root
        self.transforms = transforms
        with open(os.path.join(DHD_PATH, 'dhd_coco.json')) as f:
            self.data = json.load(f)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img#, self.data[self.imgs[idx]]

    def __len__(self):
        return len(self.imgs)

class DHDDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        with open(os.path.join(self.path, 'dhd_coco.json')) as f:
            self.data = json.load(f)
        self.images = list(sorted(os.listdir(os.path.join(self.path, "Images"))))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: DHDDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
