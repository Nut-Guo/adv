from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from src.common.utils import PROJECT_ROOT


class PatchAdvNet(pl.LightningModule):
    def __init__(self, model_name='yolov4-tiny', dataset=WiderPersonDataset, patch_size=100, batch_size=64,
                 datamodule=ImageDataModule):
        super().__init__()
        self.yolo, self.config = get_model(model_name)
        target_size = (self.config.width, self.config.height)
        self.patch_size = patch_size
        self.automatic_optimization = True
        self.reset_patch()
        self.batch_size = batch_size
        # self.datamodule = datamodule(dataset)
        self.lastpatch = None
        # self.bce = torch.nn.BCELoss()
        # self.train_dataloader, self.val_dataloader = datamodule.train_dataloader, datamodule.val_dataloader
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(target_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dset = dataset(transforms=self.transforms)
        train_size = int(float(len(dset)) * 0.9)
        val_size = len(dset) - train_size
        self.train_set, self.val_set = random_split(dset, [train_size, val_size])

    def reset_patch(self):
        self.patch = PatchNet(self.patch_size)

    def _infer(self, image):
        self.yolo.eval()
        detections = self.yolo(image)
        detections = self.yolo._split_detections(detections)
        detections = self.yolo._filter_detections(detections)
        return list(zip(detections['boxes'], detections['scores'], detections['labels']))

    def forward(self, x):
        x = self.patch(x)
        x = self._infer(x)
        return x

    def get_loss(self, results, target='person'):
        confidences = []
        for i, result in enumerate(results):
            bbox, confidence, id = result
            if target:
                # mask = torch.tensor([NAME2ID[i] for i in target],dtype = torch.long).unsqueeze(0).T
                # mask = torch.any(id.eq(mask), 0)
                mask = id.eq(NAME2ID[target])
                confidence = confidence[mask]
                confidences.append(confidence)
            # confidences.append(confidence)
        confidences = torch.cat(confidences).mean() if confidences else torch.tensor(0, dtype=torch.float32)
        # target = torch.zeros_like(confidences)
        # return self.bce(confidences, target)
        return confidences.sum()

    def training_step(self, batch, batch_idx):
        # if type(batch) is tuple:
        #     x, position = batch
        #     x = self.patch(x, position)
        # else:
        x = batch
        x = self.patch(x)
        x = self._infer(x)
        loss = self.get_loss(x)
        self.log('train_loss', loss)
        if (batch_idx % 1000 == 999):
            patch = self.patch.patch.clone().detach().cpu().permute((1, 2, 0))
            if self.lastpatch != None:
                delta = patch - self.lastpatch
                print(delta.max())
            self.lastpatch = patch
            plt.figure()
            plt.axis('off')
            plt.imshow(patch)
            plt.show()
        return loss

    def validation_step(self, batch, batch_idx):
        # if type(batch) is tuple:
        #     x, position = batch
        #     x = self.patch(x, position)
        # else:
        x = batch
        x = self.patch(x)
        x = self._infer(x)
        loss = self.get_loss(x)
        self.log('val_loss', loss)
        if (batch_idx % 1000 == 999):
            img = x.clone().detach().cpu().permute((1, 2, 0))
            plt.figure()
            plt.axis('off')
            plt.imshow(x)
            plt.show()
        return loss


class PatchNet(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        raise NotImplementedError

    def step(self, batch: Any, batch_idx: int):

        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss},
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
