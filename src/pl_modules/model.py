from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from omegaconf import DictConfig
from torch.optim import Optimizer
from torchvision.utils import draw_bounding_boxes
from src.common.utils import PROJECT_ROOT
from src.pl_modules.yolo import *
from src.pl_modules.patch import *
import wandb
import matplotlib.pyplot as plt
# from src.pl_modules.median_pool import *


class PatchNet(pl.LightningModule):
    def __init__(self, yolo_version, patch_size=100, init_patch='random', alpha=0.1, log_interval=100,
                 patch_transforms=None, *args, **kwargs) -> None:
        super().__init__()
        self.yolo, self.yolo_config = get_yolo(yolo_version)
        self.patch_size = patch_size
        self.patch_applier = PatchApplier()
        self.alpha = alpha
        self.patch_transformer = PatchTransformer(self.yolo_config.height, self.patch_size, **patch_transforms)
        self.pred_extractor = PredExtractor('person')
        self.total_variation = TotalVariation()
        self.log_interval = log_interval
        self.patch = self.generate_patch(init_patch)
        self.register_parameter(name='patch', param=self.patch)
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

    def generate_patch(self, patch_type):
        """
        Generate a random patch as a starting point for optimization.

        :param patch_type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if patch_type == 'gray':
            adv_patch = nn.Parameter(torch.full((3, self.patch_size, self.patch_size), 0.5), requires_grad=True)
        elif patch_type == 'random':
            adv_patch = nn.Parameter(torch.rand((3, self.patch_size, self.patch_size)), requires_grad=True)

        return adv_patch

    def get_patch(self):
        trans = transforms.ToPILImage()
        return trans(self.patch.clone().detach())

    def forward(self, img_batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        raise NotImplementedError

    def step(self, batch: Any, batch_idx: int):
        image_batch = batch['image']
        with torch.no_grad():
            self.patch.data = self.patch.data.clamp(0.001, 0.999)
            # gt = self.yolo(image_batch)
            # gt_output = self.pred_extractor(gt)
        adv_batch = self.patch_transformer(self.patch, batch['boxes'])
        image_batch = self.patch_applier(image_batch, adv_batch)
        # image_batch = F.interpolate(image_batch, (self.yolo_config.height, self.yolo_config.width))
        self.yolo.eval()
        detections = self.yolo(image_batch)
        pred = self.pred_extractor(detections)
        tv = self.total_variation(self.patch)
        tv_loss = tv * self.alpha
        if pred['classprobs'][0].nelement() != 0:
            det_loss = torch.mean(torch.cat(pred['scores']) * (-torch.log(1 - (torch.cat(pred['classprobs'])))))
            self.log("confidence", pred['classprobs'][0][0])
        else:
            det_loss = torch.tensor(0.)

        image = image_batch[0].clone().clamp(0, 1).detach().cpu()
        image = (image * 255).byte()
        boxes = pred['boxes'][0].clone().clamp(0, 1).detach().cpu().byte()
        bbox_image = draw_bounding_boxes(image, boxes, width=10,
                                         font="LiberationMono-Bold.ttf", font_size=30)
        if batch_idx % self.log_interval == 0:
            self.logger.experiment.log({
                'patch': wandb.Image(self.patch.clone().detach()),
                'adv_patch': wandb.Image(adv_batch[0].clone().detach()),
                'patched_img': wandb.Image(bbox_image.float().clone().detach())
            })
        # self.log_dict(
        #     {
        #         'det_loss': det_loss,
        #         'tv_loss': tv_loss,
        #     }
        # )
        loss = det_loss + torch.max(tv_loss, torch.tensor(0.1))
        losses = {'loss': loss, "det_loss": det_loss, "tv_loss": tv_loss}
        return losses

    # def training_epoch_end(self, outputs: List[Any]) -> None:
    #     mean_loss = torch.mean(torch.stack(outputs['det_loss']))
    #     self.get_patch().save()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": losses['loss'],
                "train_det_loss": losses['det_loss'],
                "train_tv_loss": losses['tv_loss'],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return losses['loss']

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": losses['loss'],
                "val_det_loss": losses['det_loss'],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return losses['loss']

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": losses['loss'],
                "test_det_loss": losses['det_loss'],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return losses['loss']

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
            self.hparams.optim.optimizer, params=[self.patch]
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
    )


if __name__ == "__main__":
    main()
