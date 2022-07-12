import pathlib

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import pytorch_lightning as pl

from .transforms import (
    ClassificationTransforms,
    TwoCropsTransform,
)


class VISDA17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size,
        transforms="default",
        drop_last=True,
        siamese=True,
    ):
        """
        Args:
            root_dir: dataset root directory (visda17 directory should be inside root directory)
            batch_size: batch size
            transforms: augmentations to use, if not given, use default augmentation setting
            drop_last: drop last incomplete training batch
            siamese: make train dataloader yield two images for training siamese network
        """
        super().__init__()
        self.data_dir = pathlib.Path(root_dir) / "visda17"
        self.batch_size = batch_size
        self.siamese = siamese
        self.drop_last = drop_last

        if isinstance(transforms, str):
            self.train_transforms = ClassificationTransforms.get_transform(transforms)
        else:
            self.train_transforms = transforms

        print(f"Using transforms: {transforms}")

        self.val_transforms = ClassificationTransforms.default

        if self.siamese:
            self.train_transforms = TwoCropsTransform(self.train_transforms)

    def setup(self, stage):
        self.trainset = datasets.ImageFolder(
            (self.data_dir / "train").as_posix(),
            self.train_transforms,
        )

        self.valset = datasets.ImageFolder(
            (self.data_dir / "validation").as_posix(),
            self.val_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
