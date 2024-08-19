import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GlaS_SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in self.get_all_files() if not f.endswith("_anno.bmp")]
        )
        self.mask_files = sorted(
            [f for f in self.get_all_files() if f.endswith("_anno.bmp")]
        )
        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of images and masks should be the same"

    def get_all_files(self):
        return glob.glob(f"{self.data_dir}/**")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == "__main__":
    dataset = GlaS_SegmentationDataset(data_dir=os.path.normpath("data/train"))
    print(len(dataset))
    image, mask = dataset[0]
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
