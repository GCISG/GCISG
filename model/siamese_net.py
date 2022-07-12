"""Adapted from MoCo implementation of PytorchLightning/lightning-bolts"""
from typing import Union, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from utils.metrics import accuracy_new, mean_sum, sum_on_key, mean
from .resnet import BasicBlock, Bottleneck


class SiameseNet(pl.LightningModule):
    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "resnet101",
        num_classes: int = 12,
        emb_dim: int = 128,
        emb_depth: int = 1,
        fc_dim: int = 512,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        stages: Union[List, Tuple] = (3, 4),
        siamese: bool = True,
        guidance_loss=None,
        ci_loss=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["guidance_loss", "ci_loss"])

        self.extra_args = kwargs
        self.guidance_loss = guidance_loss
        self.ci_loss = ci_loss

        self.criterion = lambda t, l: F.cross_entropy(t, l, ignore_index=255)
        self.register_buffer("val_acc_best", torch.FloatTensor([0]))

        self.encoder_q, self.encoder_k, self.encoder_q_e = self._init_encoders(
            base_encoder
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._override_classifier()
        self._attach_projector(self.encoder_q)
        self._attach_projector(self.encoder_k)
        self._attach_projector(self.encoder_q_e)

        self._copy_weights(self.encoder_q, self.encoder_k, grad=False)
        self._copy_weights(self.encoder_q, self.encoder_q_e, grad=False)

    def _init_encoders(self, base_encoder):
        encoder_q = base_encoder(pretrained=True)
        encoder_k = base_encoder(pretrained=True) if self.guidance_loss else None
        encoder_q_e = base_encoder(pretrained=True) if self.ci_loss else None

        return encoder_q, encoder_k, encoder_q_e

    def _override_classifier(self):
        classifier_layer = self.encoder_q.fc
        dim_fc = classifier_layer.weight.shape[1]
        dim_head = self.hparams.fc_dim

        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_fc, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, self.hparams.num_classes),
        )
        # The output of fc is not used, but these exist for consistency
        if self.encoder_k:
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_fc, dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, self.hparams.num_classes),
            )
        if self.encoder_q_e:
            self.encoder_q_e.fc = nn.Sequential(
                nn.Linear(dim_fc, dim_head),
                nn.ReLU(),
                nn.Linear(dim_head, self.hparams.num_classes),
            )

    def _attach_projector(self, encoder):
        if not encoder:
            return
        # add mlp layer for contrastive loss
        mlp = {}
        for stage in self.hparams.stages:
            if stage == 0:
                dim_mlp = encoder.conv1.weight.shape[0]
            else:
                block = getattr(encoder, f"layer{stage}")[-1]

            if isinstance(block, Bottleneck):
                dim_mlp = block.conv3.weight.shape[0]
            elif isinstance(block, BasicBlock):
                dim_mlp = block.conv2.weight.shape[0]
            else:
                raise NotImplementedError(f"{type(block)} not supported.")

            emb = []
            for _ in range(self.hparams.emb_depth):
                emb.append(nn.Linear(dim_mlp, dim_mlp))
                emb.append(nn.ReLU())

            emb.append(nn.Linear(dim_mlp, self.hparams.emb_dim))

            mlp[f"mlp_{stage}"] = nn.Sequential(
                *emb,
            )

        encoder.mlp = nn.ModuleDict(mlp)

    def _copy_weights(self, encoder_q, encoder_k, grad=False):
        if not encoder_k:
            return

        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = grad

    def forward(self, img_1, img_2):
        def _forward(encoder, img, stages):
            if not encoder:
                return None, {}

            y_pred, features = encoder(img)
            outputs = {}
            for idx, stage in enumerate(stages):
                q = self.avgpool(features[stage])
                q = torch.flatten(q, 1)
                q = encoder.mlp[f"mlp_{stage}"](q)
                q = nn.functional.normalize(q, dim=1)

                outputs[stage] = {
                    "output": q,
                    "activation": features[stage],
                }

            return y_pred, outputs

        features = {}
        y_pred, features_q = _forward(self.encoder_q, img_1, self.hparams.stages)
        with torch.no_grad():
            _, features_k = _forward(self.encoder_k, img_1, self.hparams.stages)
            _, features_q_e = _forward(self.encoder_q_e, img_2, self.hparams.stages)

        for stage in self.hparams.stages:
            features[stage] = {
                "q": features_q[stage]["output"],
                "q_activation": features_q[stage]["activation"],
            }

            if self.encoder_k:
                features[stage]["k"] = features_k[stage]["output"]
                features[stage]["k_activation"] = features_k[stage]["activation"]

            if self.encoder_q_e:
                features[stage]["q_e"] = features_q_e[stage]["output"]
                features[stage]["q_e_activation"] = features_q_e[stage]["activation"]

        return y_pred, features

    def _calc_loss_auxiliary(
        self, encoder_q, encoder_k, features, loss_func, stages, q, k
    ):
        if not encoder_k:
            return 0.0

        loss_func.on_forward(encoder_q=encoder_q, encoder_k=encoder_k)

        return sum(
            [
                loss_func(
                    q=f[q],
                    k=f[k],
                    stage=stage,
                    q_activation=f[f"{q}_activation"],
                    k_activation=f[f"{k}_activation"],
                )
                for stage, f in features.items()
                if stage in stages
            ]
        )

    def _calc_loss(self, img_1, img_2, labels, symmetric=False):
        y_pred, features = self(img_1=img_1, img_2=img_2)

        if symmetric:
            class_loss = 0
            guidance_loss = 0.0
        else:
            class_loss = self.criterion(y_pred, labels.long())
            guidance_loss = self._calc_loss_auxiliary(
                self.encoder_q,
                self.encoder_k,
                features,
                self.guidance_loss,
                self.hparams.stages,
                q="q",
                k="k",
            )

        ci_loss = self._calc_loss_auxiliary(
            self.encoder_q,
            self.encoder_q_e,
            features,
            self.ci_loss,
            self.hparams.stages,
            q="q",
            k="q_e",
        )

        loss = class_loss + guidance_loss + ci_loss

        if symmetric:
            _, loss_symmetric = self._calc_loss(img_2, img_1, labels, symmetric=False)
            loss = (loss + loss_symmetric["loss"])
            class_loss = (class_loss + loss_symmetric["class_loss"])
            guidance_loss = (guidance_loss + loss_symmetric["guidance_loss"])
            ci_loss = (ci_loss + loss_symmetric["ci_loss"])

        return y_pred, {
            "loss": loss,
            "class_loss": class_loss,
            "guidance_loss": guidance_loss,
            "ci_loss": ci_loss,
        }

    def training_step(self, batch, batch_idx):
        # Use pre-calculated batch norm statistics
        if self.encoder_k:
            self.encoder_k.apply(set_bn_eval)

        (img_1, img_2), labels = batch
        y_pred, loss = self._calc_loss(
            img_1, img_2, labels, symmetric=self.hparams.symmetric
        )

        self._post_training_step()

        log = {
            "train_loss": loss["loss"],
            "train_loss_class": loss["class_loss"],
            "train_loss_guidance": loss["guidance_loss"],
            "train_loss_ci": loss["ci_loss"],
        }

        self.log_dict(log, sync_dist=True)
        return loss["loss"]

    def _post_training_step(self):
        if self.encoder_q_e:
            self._ema(
                base=self.encoder_q,
                target=self.encoder_q_e,
                # TODO: paramerterize
                layers=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
                momentum=0.996,
            )

    def _ema(self, base, target, layers, momentum):
        for layer_name in layers:
            layer_base = getattr(base, layer_name)
            layer_target = getattr(target, layer_name)
            for param_base, param_target in zip(
                layer_base.parameters(), layer_target.parameters()
            ):
                param_target.data = param_target.data * momentum + param_base.data * (
                    1.0 - momentum
                )

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        y_pred, _ = self.encoder_q(img)

        loss = self.criterion(y_pred, labels.long())
        # per class accuracy (following VisDA accuracy protocol)
        acc1, count = accuracy_new(y_pred, labels, self.hparams.num_classes)
        return {"val_loss": loss, "val_acc1": acc1, "count": count}

    def validation_epoch_end(self, results):
        val_loss = torch.mean(self.all_gather(mean(results, "val_loss")))
        val_acc1_sum = torch.sum(
            self.all_gather(mean_sum(results, "val_acc1", "count")), dim=0
        )
        val_acc1_count = torch.sum(self.all_gather(sum_on_key(results, "count")), dim=0)
        val_acc1 = torch.mean(val_acc1_sum / val_acc1_count)

        self.val_acc_best = torch.max(self.val_acc_best, val_acc1)

        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc1_best": self.val_acc_best,
        }

        if not self.trainer.sanity_checking:
            self.log_dict(log)
            log_str = " / ".join([f"{k}: {float(v):.3f}" for k, v in log.items()])
            epoch_result = f"\t[Epoch {self.current_epoch}]: [{log_str}]"
            if self.trainer.is_global_zero:
                print(epoch_result)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


def set_bn_eval(module):
    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
        module.eval()
