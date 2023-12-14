import os
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from torchvision.datasets import VisionDataset


class EmotionDetectionDataset(VisionDataset):
    labels_and_classes = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprised"
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        self.data, self.targets = self._load_data()

    def _load_data(self) -> Tuple[Any, Any]:
        base = 'train' if self.train else 'test'
        h5_data_file = f'{base}_data.h5'
        h5_targets_file = f'{base}_targets.h5'
        data_path = os.path.join(self.root, h5_data_file)
        targets_path = os.path.join(self.root, h5_targets_file)

        # Save to h5 if not already for easier future data importing
        if not os.path.isfile(data_path) or not os.path.isfile(targets_path):
            images, targets = self.load_and_save_h5_data_and_targets(base, data_path, targets_path)
        # Load from h5 files
        else:
            with h5py.File(data_path, 'r') as hf:
                images = hf[f'{base}_data'][:]
            with h5py.File(targets_path, 'r') as hf:
                targets = hf[f'{base}_targets'][:]

        return torch.tensor(images, dtype=torch.uint8), torch.tensor(targets, dtype=torch.uint8)

    def load_and_save_h5_data_and_targets(self, base, data_path, targets_path):
        base_path = os.path.join(self.root, base)
        images = []
        targets = []
        for label, cls in self.labels_and_classes.items():
            class_path = os.path.join(base_path, cls)
            for img_name in os.listdir(class_path):
                img = imread(os.path.join(class_path, img_name))
                images.append(img)
                targets.append(label)
        images = np.array(images).astype(np.uint8)
        targets = np.array(targets)

        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset(f'{base}_data', data=images)
        with h5py.File(targets_path, 'w') as hf:
            hf.create_dataset(f'{base}_targets', data=targets)

        return images, targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
