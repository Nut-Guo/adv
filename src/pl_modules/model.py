from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from omegaconf import DictConfig
from torch.optim import Optimizer
from src.common.utils import PROJECT_ROOT, mask2bbox
from src.pl_modules.yolo import *
from src.pl_modules.patch import *
import wandb
import matplotlib.pyplot as plt
# from src.pl_modules.median_pool import *


class PatchNet(pl.LightningModule):
    def __init__(self, yolo_version, patch_size=100, init_patch='random', alpha=0.1, log_interval=100, patch_transformer=None,
                 pred_extractor=None, thresh_hold=0.5, log_att=False,  *args, **kwargs) -> None:
        super().__init__()
        self.yolo, self.yolo_config = get_yolo(yolo_version)
        self.patch_applier = PatchApplier()
        self.alpha = alpha
        self.log_att = log_att
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

    def get_boxes(self, detections):
        # for boxes, classprobs in zip(detections['boxes'], detections['classprobs']):
        #     print(boxes, classprobs)
        boxes = [{
            "predictions": {
                "box_data": [{
                    "position": {
                        "minX": box[0].item(),
                        "maxX": box[2].item(),
                        "minY": box[1].item(),
                        "maxY": box[3].item(),
                    },
                    "class_id": int(label),
                    "scores": {
                        "prob": classprob.item()
                    },
                    "domain": "pixel",
                    "box_caption": "%s (%.3f)" % (NAMES[int(label)], classprob.item())
                }
                    for label, box, classprob in
                    zip([0 for _ in range(len(boxes))], boxes, classprobs)
                ],
                "class_labels": {i: j for i, j in enumerate(NAMES)},
            }
        } for boxes, classprobs in zip(detections['boxes'], detections['classprobs'])]
        return boxes

    def step(self, batch: Any, batch_idx: int):
        image_batch = batch['image'][0]
        bboxes = batch['bboxes'][0].clamp(0, image_batch.shape[-1])
        with torch.no_grad():
            self.patch.data = self.patch.data.clamp(0.001, 0.999)
            gt = self.yolo(image_batch)
            gt_output = self.pred_extractor(gt)
        adv_batch = self.patch_transformer(self.patch, bboxes)  # gt_output['boxes'])  # batch['boxes'])
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

        atts = torch.sum((detections[:, :, 3] - detections[:, :, 1]) *
                         (detections[:, :, 2] - detections[:, :, 0]) *
                         detections[:, :, 4], dim=1) / (image_batch.shape[-1] * image_batch.shape[-2])
        # adv_mask = adv_batch != 0
        # adv_box = mask2bbox(adv_mask).expand_as(detections[:, :, :4])
        # inter_x0 = torch.max(adv_box[:, :, 0], detections[:, :, 0])
        # inter_x1 = torch.min(adv_box[:, :, 2], detections[:, :, 2])
        # inter_y0 = torch.max(adv_box[:, :, 1], detections[:, :, 1])
        # inter_y1 = torch.min(adv_box[:, :, 3], detections[:, :, 3])
        # adv_atts = torch.sum((inter_y1 - inter_y0) *
        #                  (inter_x1 - inter_x0) *
        #                  detections[:, :, 4], dim=1) / (image_batch.shape[-1] * image_batch.shape[-2])
        att_loss = atts.sum()  # - adv_atts.sum()
        if batch_idx % self.log_interval == 0:
            # if self.log_att:
            #     # orig_attentions = torch.zeros_like(image_batch[:, 0, :, :], requires_grad=False)
            #     # attentions = torch.zeros_like(image_batch[:, 0, :, :], requires_grad=False)
            #     # for attention, detection in zip(attentions, detections.clone().detach()):
            #     #     for det in detection:
            #     #         attention[int(det[1]): int(det[3]), int(det[0]): int(det[2])] += det[4]
            #     # with torch.no_grad():
            #     #     for attention, detection in zip(orig_attentions, gt.clone().detach()):
            #     #         for det in detection:
            #     #             attention[int(det[1]): int(det[3]), int(det[0]): int(det[2])] += det[4]
            #     attention_img = torchvision.utils.make_grid(attentions).permute(1, 2, 0)
            #     self.logger.experiment.log({
            #         # 'orig_attention': wandb.Image(orig_attentions.clone().detach().unsqueeze(dim=1)),
            #         'attention_map': wandb.Image(attentions.clone().detach().unsqueeze(dim=1))
            #     })
            # plt.axis('off')
            # attention_map = plt.imshow(attention_img.cpu(), cmap='jet',aspect='auto')
            # plt.colorbar()
            self.logger.experiment.log({
                'patch': wandb.Image(self.patch.clone().detach()),
                #'adv_patch': wandb.Image(adv_batch.clone().detach()),   # boxes=origin_boxes),
                'orig_image': [wandb.Image(image, boxes=boxes) for image, boxes in zip(image_batch.clone().detach(),
                                                                                       self.get_boxes(gt_output))],
                # 'orig_attention': wandb.Image(orig_attentions.clone().detach().unsqueeze(dim=1)),
                'patched_img': [wandb.Image(image, boxes=boxes) for image, boxes in zip(patched_batch.clone().detach(),
                                                                                        self.get_boxes(pred))],  # boxes=patched_boxes)
                # 'attention_map': wandb.Image(attentions.clone().detach().unsqueeze(dim=1))
            },
                commit=False)
        loss = det_loss + tv_loss + att_loss
        losses = {'loss': loss, "det_loss": det_loss, "tv_loss": tv_loss, "att_loss": att_loss}
        return losses

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": losses['loss'],
                "train_det_loss": losses['det_loss'],
                "train_tv_loss": losses['tv_loss'],
                "train_att_loss": losses['att_loss'],
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
