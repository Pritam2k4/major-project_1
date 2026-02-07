from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from base.torchvision_dataset import TorchvisionDataset


class _FusedImageDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # return (image, target, index). Target will be 0 (normal) by default
        return img, 0, index


class OutputFused_Dataset(TorchvisionDataset):
    """Simple dataset for images in a single folder (no subfolders).

    Treats all images as the normal class. Both train and test sets point to
    the same images by default so DeepSVDD can be trained and evaluated on them.
    """

    def __init__(self, root: str):
        super().__init__(root)

        self.n_classes = 2
        self.normal_classes = (0,)
        self.outlier_classes = tuple()

        # Basic transforms: resize to 32x32 and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # Gather image file paths
        import os
        files = [os.path.join(self.root, f) for f in sorted(os.listdir(self.root))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        dataset = _FusedImageDataset(files, transform=transform)

        # Use the same set as both train and test (no labels available)
        self.train_set = dataset
        self.test_set = dataset
