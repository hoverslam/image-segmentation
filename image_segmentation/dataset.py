import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2


class PetsDataset(Dataset):

    def __init__(self, root: str, split: str) -> None:
        super().__init__()
        transforms = v2.Compose([v2.Resize((256, 256)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        target_transforms = v2.Compose([v2.Resize((256, 256)), v2.ToImage(), v2.ToDtype(torch.float32)])
        self.data = OxfordIIITPet(
            root,
            split,
            target_types="segmentation",
            download=True,
            transform=transforms,
            target_transform=target_transforms,
        )
        type(self.data[0][0])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.data[idx]
        target = mask.type(torch.long) - 1  # 1: pet, 2: background, 3: border => 0: pet, 1: background, 2: border
        target = F.one_hot(target, num_classes=3).squeeze(0).permute(2, 0, 1)

        return img, target
