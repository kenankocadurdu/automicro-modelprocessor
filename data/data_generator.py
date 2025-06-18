import os
import json
from io import BytesIO
from PIL import Image
import torch.utils.data as data
from data import augmentations

class DatasetClassificationMinIO(data.Dataset):
    """
    Custom dataset for image classification tasks.
    Reads images and labels from MinIO or local paths and applies transformations.
    """

    def __init__(
        self,
        entries,
        path_dataset: str,
        mean,
        std,
        size=224,
        augmentation=True,
        train_mode=True,
        minio_client=None,
        augmentation_config=None
    ):
        self.entries = entries

        # Build class-to-index map
        class_names = sorted(set(e["label"] for e in self.entries))
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}

        self.path_dataset = path_dataset
        self.mean = mean
        self.std = std
        self.size = size
        self.augmentation = augmentation
        self.augmentation_config = augmentation_config or {}
        self.client = minio_client
        self.train_mode = train_mode

        # Set up transformations
        self.transform = (
            augmentations.get_train_transforms(size, mean, std, config=self.augmentation_config)
            if augmentation
            else augmentations.get_val_transforms(size, mean, std)
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        try:
            # Load image from MinIO or local file
            if self.client is not None:
                response = self.client.get_object(self.client.bucket, entry["image_key"])
                img_data = response.read()
                image = Image.open(BytesIO(img_data)).convert("RGB")
            else:
                local_path = os.path.join(self.path_dataset, entry["image_key"])
                image = Image.open(local_path).convert("RGB")
        except Exception as e:
            source = 'MinIO' if self.client else 'local filesystem'
            raise RuntimeError(f"Failed to load image {entry['image_key']} from {source}: {e}")

        image = self.transform(image)
        label = self.class_to_index[entry["label"]]

        if self.train_mode:
            return image, label
        else:
            filename = entry["image_key"].split("/")[-1]
            return image, label, filename
