from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from omegaconf import DictConfig
from torch.optim import Optimizer
from src.common.utils import PROJECT_ROOT
from src.pl_modules.yolo import *
from src.pl_modules.patch import *
import wandb
import matplotlib.pyplot as plt
# from src.pl_modules.median_pool import *


class PatchNet(pl.LightningModule):
    def __init__(self, yolo_version, patch_size=100, init_patch='random', alpha=0.1, log_interval=100, patch_transformer=None,
                 pred_extractor=None, thresh_hold=0.5,  *args, **kwargs) -> None:
        super().__init__()
        self.yolo, self.yolo_config = get_yolo(yolo_version)
        self.patch_applier = PatchApplier()
        self.alpha = alpha
        self.thresh_hold = thresh_hold
        self.patch_transformer = patch_transformer
        self.pred_extractor = pred_extractor
        self.total_variation = TotalVariation()
        self.log_interval = log_interval
        self.patch_size = patch_size
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
            adv_patch = nn.Parameter(torch.full((3, self.patch_size[0], self.patch_size[1]), 0.5), requires_grad=True)
        elif patch_type == 'random':
            adv_patch = nn.Parameter(torch.rand((3, self.patch_size[0], self.patch_size[1])), requires_grad=True)

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
        image_batch = batch['image'][0]
        bboxes = batch['boxes'][0]
        with torch.no_grad():
            self.patch.data = self.patch.data.clamp(0.001, 0.999)
            # gt = self.yolo(image_batch)
            # gt_output = self.pred_extractor(gt)
        adv_batch = self.patch_transformer(self.patch, bboxes)  # batch['boxes'])
        patched_batch = self.patch_applier(image_batch, adv_batch)
        # image_batch = F.interpolate(image_batch, (self.yolo_config.height, self.yolo_config.width))
        self.yolo.eval()
        detections = self.yolo(patched_batch)
        pred = self.pred_extractor(detections)
        tv = self.total_variation(self.patch)
        tv_loss = tv * self.alpha
        det_loss = torch.sum(torch.cat(pred['scores']) * (-torch.log(1 - (torch.cat(pred['classprobs'])))))
        if pred['classprobs'][0].nelement() != 0:
            # self.log("confidence", pred['classprobs'][0][0])
            if pred['classprobs'][0][0] > self.thresh_hold:
                self.logger.agg_and_log_metrics({"success_rate": 0})
            else:
                self.logger.agg_and_log_metrics({"success_rate": 1})
        else:
            # self.log("confidence", torch.tensor(0, device='cuda'))
            self.logger.agg_and_log_metrics({"success_rate": 1})
        adv_mask = adv_batch != 0
        attentions = torch.zeros_like(image_batch)
        for attention, detection in zip(attentions, detections):
            for det in detection:
                attention[int(det[0]): int(det[2]), int(det[1]): int(det[2])] += det[5]
        adv_attention = attentions[adv_mask]
        image_attention = attentions[~adv_mask]
        attention_loss = image_attention.sum() - adv_attention.sum()
        if batch_idx % self.log_interval == 0:
            # origin_boxes = {
            #     "predictions": {
            #         "box_data": [{
            #             "position": {
            #                 "minX": box[0].item(),
            #                 "maxX": box[2].item(),
            #                 "minY": box[1].item(),
            #                 "maxY": box[3].item(),
            #             },
            #             "class_id": int(label),
            #             "scores": {
            #                 "prob": classprob.item()
            #             },
            #             "domain": "pixel",
            #             "box_caption": "%s (%.3f)" % (NAMES[int(label)], classprob.item())
            #         }
            #             for label, box, classprob in zip([0 for _ in range(len(batch['boxes'][0][0]))], batch['boxes'][0][0], batch['classprobs'][0][0])
            #         ],
            #         "class_labels": {i: j for i, j in enumerate(NAMES)},
            #     }
            # }
            # patched_boxes = {
            #     "predictions": {
            #         "box_data": [{
            #             "position": {
            #                 "minX": box[0].item(),
            #                 "maxX": box[2].item(),
            #                 "minY": box[1].item(),
            #                 "maxY": box[3].item(),
            #             },
            #             "class_id": int(label.item()),
            #             "scores": {
            #                 "prob": classprob.item()
            #             },
            #             "domain": "pixel",
            #             "box_caption": "%s (%.3f)" % (NAMES[int(label.item())], classprob.item())
            #         }
            #             for label, box, classprob in zip(pred['labels'][0], pred['boxes'][0], pred['classprobs'][0])
            #         ],
            #         "class_labels": {i: j for i, j in enumerate(NAMES)},
            #     }
            # }
            attention_img = torchvision.utils.make_grid(attentions).permute(1, 2, 0)
            plt.axis('off')
            attention_map = plt.imshow(attention_img.cpu())
            plt.colorbar()

            self.logger.experiment.log({
                'patch': wandb.Image(self.patch.clone().detach()),
                #'adv_patch': wandb.Image(adv_batch.clone().detach()),   # boxes=origin_boxes),
                'orig_image': wandb.Image(image_batch.clone().detach()),   # boxes=origin_boxes),
                'patched_img': wandb.Image(patched_batch.clone().detach()),  # boxes=patched_boxes)
                'attention_map': attention_map
            },
                commit=False)
        loss = det_loss + tv_loss + attention_loss
        losses = {'loss': loss, "det_loss": det_loss, "tv_loss": tv_loss, "attention_loss": attention_loss}
        return losses

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
        # opt = hydra.utils.instantiate(
        #     self.hparams.optim.optimizer, params=[self.patch]
        # )
        opt = self.hparams.optim.optimizer(params=[self.patch])
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        # scheduler = hydra.utils.instantiate(
        #     self.hparams.optim.lr_scheduler, optimizer=opt
        # )
        scheduler = self.hparams.optim.lr_scheduler(optimizer=opt)
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
