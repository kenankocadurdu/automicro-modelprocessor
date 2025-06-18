import json
import os
import logging
import requests
import io
import random
import traceback

from io import BytesIO
from collections import defaultdict
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from minio import Minio
from botocore.exceptions import ClientError

import torch
import torchvision.transforms as transforms

from model.model_generator import generator
from trainer import Executor

# Initialize FastAPI app
app = FastAPI(
    title="AutoMicro MLOps",
    version="1.0.0",
    description="Training and Prediction API"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Request schema for training
class TrainingRequest(BaseModel):
    training_project_id: int
    data_project_id: int
    dataset_id: int
    version_tag: str
    model_name: str
    num_class: int
    hyperparameters: Dict[str, float]
    callback_url: Optional[str] = None
    use_minio: bool = True
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str

def send_status(callback_url, status, training_project_id, dataset_id, version_tag, error=None):
    """Send training status updates to callback URL."""
    if not callback_url:
        return
    try:
        data = {
            "status": status,
            "training_project_id": training_project_id,
            "dataset_id": dataset_id,
            "version_tag": version_tag,
            "error": error
        }
        requests.post(callback_url, json=data, timeout=5)
    except Exception as e:
        logging.warning(f"Status callback failed: {e}")

@app.post("/train")
def trigger_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """Endpoint to start training in background."""
    background_tasks.add_task(run_training_pipeline, req)
    return {"message": "Training scheduled"}

def run_training_pipeline(req: TrainingRequest):
    """Main training pipeline."""
    try:
        logging.info(f"Received TrainingRequest: {req.json()}")

        # Initialize MinIO client
        minio_client = Minio(
            req.minio_endpoint,
            access_key=req.minio_access_key,
            secret_key=req.minio_secret_key,
            secure=False
        )

        # Read version JSON file
        json_key = f"versions/{req.data_project_id}/{req.dataset_id}/{req.version_tag}.json"
        response = minio_client.get_object(req.minio_bucket, json_key)
        version_json = json.loads(response.read())
        files = version_json["files"]

        # Split into training and validation sets
        random.seed(45)
        class_files = defaultdict(list)
        for item in files:
            class_files[item["class"]].append(item)

        train_files, val_files = [], []
        for cls, items in class_files.items():
            random.shuffle(items)
            split_idx = int(len(items) * 0.8)
            for item in items[:split_idx]:
                train_files.append({
                    "image_key": f"datasets/{req.data_project_id}/{req.dataset_id}/{item['filename']}",
                    "label": item["class"]
                })
            for item in items[split_idx:]:
                val_files.append({
                    "image_key": f"datasets/{req.data_project_id}/{req.dataset_id}/{item['filename']}",
                    "label": item["class"]
                })
    except Exception as e:
        logging.exception("MinIO JSON read failed")
        raise HTTPException(status_code=500, detail=f"MinIO JSON read failed: {e}")

    try:
        # Upload training/validation split files
        for split, file_data in zip(["train", "val"], [train_files, val_files]):
            stream = BytesIO(json.dumps(file_data).encode("utf-8"))
            minio_client.put_object(
                req.minio_bucket,
                f"training/{req.data_project_id}/{req.dataset_id}/{req.training_project_id}/datasets/{split}_dataset.json",
                stream,
                length=stream.getbuffer().nbytes
            )
            stream.close()
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"MinIO upload error: {str(e)}")

    try:
        # Initialize model
        model_generator = generator(
            name=req.model_name,
            num_class=req.num_class,
            image_size=int(req.hyperparameters.get("image_size", 224)),
            run_name=f"{req.model_name}_{req.version_tag}"
        )
        model_generator.training_project_id = req.training_project_id
        model_generator.save_path = f"training/{req.data_project_id}/{req.dataset_id}/{req.training_project_id}/results"
        model_generator.data_project_id = req.data_project_id
        model_generator.dataset_id = req.dataset_id

        # Set up executor
        executor = Executor(
            path_dataset=f"datasets/{req.training_project_id}/{req.dataset_id}",
            batch_size=int(req.hyperparameters.get("batch_size", 4)),
            augmentation=True,
            num_threads=1,
            device_id=0,
            accumulation_steps=int(req.hyperparameters.get("accumulation_steps", 1)),
            chunk_size=int(req.hyperparameters.get("chunk_size", 500)),
            num_epochs=int(req.hyperparameters.get("epochs", 5)),
            lr=req.hyperparameters.get("lr", 0.0001),
            patience=int(req.hyperparameters.get("patience", 2)),
            opt_func=req.hyperparameters.get("opt_func", "Adam"),
            criterion=req.hyperparameters.get("criterion", "CrossEntropyLoss"),
            save_path=model_generator.save_path,
            use_minio=req.use_minio,
            minio_endpoint=req.minio_endpoint,
            minio_access_key=req.minio_access_key,
            minio_secret_key=req.minio_secret_key,
            minio_bucket=req.minio_bucket
        )

        # Train the model
        send_status(req.callback_url, "running", req.training_project_id, req.dataset_id, req.version_tag)
        os.makedirs(model_generator.save_path, exist_ok=True)
        executor.execute(model_generator)
        send_status(req.callback_url, "completed", req.training_project_id, req.dataset_id, req.version_tag)

    except Exception as e:
        logging.exception("Training failed")
        send_status(req.callback_url, "failed", req.training_project_id, req.dataset_id, req.version_tag, error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    return {"message": "Training started and executed successfully"}

@app.get("/available_models")
def list_models():
    """Return list of supported models."""
    return ["ResNet18", "MobileNetV3Small", "ResNet18_CBAM"]

# Request schema for prediction
class PredictRequest(BaseModel):
    model_path: str
    image_path: str
    model_type: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str

def load_model_from_minio_with_generator(minio_client, bucket, model_path, model_type):
    """Download and load model from MinIO."""
    response = minio_client.get_object(bucket, model_path)
    buffer = io.BytesIO(response.read())
    state_dict = torch.load(buffer, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model_generator = generator(name=model_type, num_class=2, image_size=224, run_name="predict_run")
    model = model_generator.model
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image_from_minio(minio_client, bucket, image_path):
    response = minio_client.get_object(bucket, image_path)
    return Image.open(io.BytesIO(response.read())).convert("RGB")

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(req: PredictRequest):
    """Perform image classification prediction."""
    try:
        minio_client = Minio(
            req.minio_endpoint,
            access_key=req.minio_access_key,
            secret_key=req.minio_secret_key,
            secure=False
        )

        model = load_model_from_minio_with_generator(minio_client, req.minio_bucket, req.model_path, req.model_type)
        image = load_image_from_minio(minio_client, req.minio_bucket, req.image_path)
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        return {"predicted_class": predicted_class}

    except Exception as e:
        logging.error("Prediction failed with exception:")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
