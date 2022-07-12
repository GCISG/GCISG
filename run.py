#!/usr/bin/env python3

import argparse
import datetime
import random
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from data.dataloader import VISDA17DataModule
from model import resnet, siamese_net
import loss as _loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        default=os.environ.get("LOGS", "logs"),
        help="Output directory where checkpoint and log will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--root",
        default=os.path.join(os.environ.get("DATASETS", "datasets")),
        help="Root directory of the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification"],
        help="Task to run (default: %(default)s)",
    )
    parser.add_argument(
        "--encoder",
        default="resnet101",
        choices=["resnet101"],
        help="Backbone network to use (default: %(default)s)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Optimizer momentum (default: %(default)s)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=12,
        help="Number of classes (default: %(default)s)",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Do not train, evaluate model",
    )
    parser.add_argument(
        "--gpus",
        default=-1,
        help="GPUs to use (default: All GPUs in the machine)",
    )
    parser.add_argument(
        "--resume", default=None, help="Resume from the given checkpoint"
    )
    parser.add_argument(
        "--dev-run",
        action="store_true",
        default=False,
        help="Run small steps to test whether model is valid",
    )
    parser.add_argument(
        "--exp-name",
        default="GCISG",
        help="Experiment name used for a log directory name",
    )
    parser.add_argument(
        "--augmentation",
        default="rand_augment",
        help="Augmentations to use (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", default="random", help="Random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--fc-dim",
        type=int,
        default=512,
        help="Dimension of FC hidden layer (default: %(default)s)",
    )
    parser.add_argument(
        "--single-network",
        dest="siamese",
        action="store_false",
        default=True,
        help="Train with single network, for comparison",
    )
    parser.add_argument(
        "--stages",
        default=(3, 4),
        nargs="+",
        help="Feature stages to use in contrastive loss calculation",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="Feature dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--emb-depth",
        type=int,
        default=1,
        help="Depth of project layers (default: %(default)s)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Calculate loss symmetrically",
    )

    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=1,
        help="Weight of Guidance loss",
    )

    parser.add_argument(
        "--ci-weight",
        type=float,
        default=1,
        help="Weight of Causal Invariance loss",
    )

    parser.add_argument(
        "--ci-temperature-q",
        type=float,
        default=0.12,
        help="Temperature Q of Causal Invariance loss",
    )

    parser.add_argument(
        "--ci-temperature-k",
        type=float,
        default=0.04,
        help="Temperature K of Causal Invariance loss",
    )

    parser.add_argument(
        "--ci-queue-size",
        type=int,
        default=2**16,
        help="Queue size of Causal Invariance loss",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision to boost the training speed",
    )

    return parser.parse_args()


def build_loss(hparams, loss):
    loss = loss.lower()
    if loss == "ci":
        return _loss.CILoss(
            scale=hparams.ci_weight,
            stages=hparams.stages,
            temperature_q=hparams.ci_temperature_q,
            temperature_k=hparams.ci_temperature_k,
            queue_size=hparams.ci_queue_size,
            embedding_dim=hparams.emb_dim,
        )
    elif loss == "guidance":
        return _loss.GuidanceLoss(
            scale=hparams.guidance_weight,
        )
    else:
        raise NotImplementedError()


def main():
    args = parse_args()
    print("Args: ", args)

    seed = random.randint(0, 1e7) if args.seed == "random" else int(args.seed)
    pl.seed_everything(seed)

    exp_name = args.exp_name
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output, "log", f"{exp_name}_{start_time}")

    print(f"Run command: {' '.join(sys.argv)}")
    print(f"Commandline args: {args}")
    print(f"Seed: {seed}")

    encoder = {
        "resnet101": resnet.resnet101,
    }[args.encoder]
    task = args.task.lower()

    if task == "classification":
        _model = siamese_net.SiameseNet
        _datamodule = VISDA17DataModule
        monitor = "val_acc1"
    else:
        raise NotImplementedError()

    model = _model(
        base_encoder=encoder,
        **vars(args),
        guidance_loss=build_loss(args, "guidance"),
        ci_loss=build_loss(args, "ci"),
    )

    datamodule = _datamodule(
        root_dir=args.root,
        batch_size=args.batch_size,
        transforms=args.augmentation,
    )

    loggers = []
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.output,
        name="log",
        version=f"{exp_name}{start_time}",
        default_hp_metric=False,
    )
    loggers.append(tensorboard_logger)

    checkpoint_monitor = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        dirpath=log_dir,
        filename=f"best_{exp_name}" + "_{epoch:02d}_{val_loss:.2f}_{val_acc1:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        accelerator="ddp",
        logger=loggers,
        callbacks=[
            lr_monitor,
            checkpoint_monitor,
        ],
        resume_from_checkpoint=args.resume,
        fast_dev_run=args.dev_run,
        plugins=[
            DDPPlugin(find_unused_parameters=True),
        ],
        precision=16 if args.amp else 32,
        num_sanity_val_steps=0,
    )

    if args.eval_only:
        assert args.resume is not None, "resume must be set along with --eval-only"

        model = _model.load_from_checkpoint(
            checkpoint_path=args.resume,
            strict=False,
            guidance_loss=None,
            ci_loss=None,
        )

        datamodule.setup(stage="val")
        # Use test to support older pytorch-lightning versions that doesn't have validate method.
        trainer.test(model, test_dataloaders=datamodule.val_dataloader())
    else:  # train
        if args.resume:
            model = _model.load_from_checkpoint(
                checkpoint_path=args.resume,
                **vars(args),
                guidance_loss=build_loss(args, "guidance"),
                ci_loss=build_loss(args, "ci"),
            )

        trainer.fit(
            model,
            datamodule,
        )

        trainer.save_checkpoint(log_dir + "/final.ckpt")


if __name__ == "__main__":
    if os.name == "nt":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    main()
