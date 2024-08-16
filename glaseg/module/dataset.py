from PIL import Image
from torch.utils.data import Dataset
import glob


class GlaS_SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in self.get_all_files() if f.endswith(".bmp")]
        )
        self.mask_files = sorted(
            [f for f in self.get_all_files() if f.endswith("_anno.bmp")]
        )

    def get_all_files(self):
        return glob.glob(f"{self.data_dir}/**")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
