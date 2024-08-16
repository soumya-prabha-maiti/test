# Define the data module
from glaseg.module.dataset import GlaS_SegmentationDataset


import os


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        # No need to download the dataset, as it's a custom dataset
        pass

    def setup(self, stage=None):
        self.train_dataset = GlaS_SegmentationDataset(
            os.path.join(self.data_dir, "train"), transform=self.transform
        )
        self.val_dataset = GlaS_SegmentationDataset(
            os.path.join(self.data_dir, "val"), transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)