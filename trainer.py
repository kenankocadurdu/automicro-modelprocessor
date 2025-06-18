import os
import json
import logging
from minio import Minio
from torch.utils.data import DataLoader
from data.data_generator import DatasetClassificationMinIO
from model import model_arch

class Executor:
    """
    Encapsulates the training execution logic, including dataset loading,
    model training, evaluation, and optional MinIO result storage.
    """

    def __init__(self, path_dataset, batch_size: int, augmentation: bool, num_threads: int, device_id: int,
                 accumulation_steps: int, chunk_size: int, num_epochs: int, lr: float, patience: int, opt_func: str,
                 criterion: str, save_path: str, use_minio: bool,
                 minio_endpoint: str = None, minio_access_key: str = None, minio_secret_key: str = None,
                 minio_bucket: str = None, augmentation_config: dict = None) -> None:

        self.path_dataset = path_dataset
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.augmentation_config = augmentation_config or {}
        self.num_threads = num_threads
        self.device_id = device_id
        self.num_epochs = int(num_epochs)
        self.lr = lr
        self.patience = patience
        self.opt_func = opt_func
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.chunk_size = chunk_size
        self.save_path = save_path
        self.use_minio = use_minio

        if self.use_minio:
            self.minio_client = Minio(
                minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False
            )
            self.minio_client.bucket = minio_bucket
        else:
            self.minio_client = None

        self._log_initial_parameters()

    def _log_initial_parameters(self):
        """Logs training configuration for debugging and traceability."""
        logging.info("Executor Parameters:")
        for key, value in vars(self).items():
            logging.info(f"{key}: {value}")

    def execute(self, model_generator):
        """
        Run the training process using the provided model generator.

        Args:
            model_generator: Instantiated model object with architecture and metadata.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Load training dataset from MinIO
        train_key = f"training/{model_generator.data_project_id}/{model_generator.dataset_id}/{model_generator.training_project_id}/datasets/train_dataset.json"
        train_obj = self.minio_client.get_object(self.minio_client.bucket, train_key)
        train_data = json.loads(train_obj.read().decode("utf-8"))
        train_obj.close()

        ds_train = DatasetClassificationMinIO(
            entries=train_data,
            path_dataset=self.path_dataset,
            mean=mean,
            std=std,
            size=model_generator.image_size,
            augmentation=self.augmentation,
            train_mode=True,
            minio_client=self.minio_client,
            augmentation_config=self.augmentation_config
        )

        # Load validation dataset from MinIO
        val_key = f"training/{model_generator.data_project_id}/{model_generator.dataset_id}/{model_generator.training_project_id}/datasets/val_dataset.json"
        val_obj = self.minio_client.get_object(self.minio_client.bucket, val_key)
        val_data = json.loads(val_obj.read().decode("utf-8"))
        val_obj.close()

        ds_val = DatasetClassificationMinIO(
            entries=val_data,
            path_dataset=self.path_dataset,
            mean=mean,
            std=std,
            size=model_generator.image_size,
            augmentation=False,
            train_mode=True,
            minio_client=self.minio_client
        )

        # Prepare data loaders
        dl_train = DataLoader(ds_train, batch_size=int(self.batch_size), shuffle=True,
                              num_workers=0, pin_memory=True, prefetch_factor=None)

        dl_val = DataLoader(ds_val, batch_size=int(self.batch_size), shuffle=False,
                            num_workers=0, pin_memory=True, prefetch_factor=None)

        # Launch training loop
        model_arch.fit(
            epochs=self.num_epochs,
            lr=self.lr,
            model_generator=model_generator,
            train_loader=dl_train,
            val_loader=dl_val,
            opt_func=self.opt_func,
            criterion=self.criterion,
            batch_size=int(self.batch_size),
            accumulation_steps=self.accumulation_steps,
            chunk_size=self.chunk_size,
            patience=self.patience,
            minio_client=self.minio_client,
            minio_bucket=self.minio_client.bucket
        )
