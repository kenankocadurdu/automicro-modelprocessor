import json, os, logging, requests, io, random
from io import BytesIO
from collections import defaultdict
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from minio import Minio
from botocore.exceptions import ClientError
from model.model_generator import generator
from trainer import Executor
from fastapi import BackgroundTasks
import torch
import torchvision.transforms as transforms
import traceback

app = FastAPI(
    title="AutoMicro MLOps",
    version="1.0.0",
    description="Training and Prediction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

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
    background_tasks.add_task(run_training_pipeline, req)
    return {"message": "Training scheduled"}


def run_training_pipeline(req: TrainingRequest):
    try:
        logging.info(f"Received TrainingRequest: {req.json()}")
        minio_client = Minio(
            req.minio_endpoint,
            access_key=req.minio_access_key,
            secret_key=req.minio_secret_key,
            secure=False
        )
        json_key = f"versions/{req.data_project_id}/{req.dataset_id}/{req.version_tag}.json"
        response = minio_client.get_object(req.minio_bucket, json_key)
        version_json = json.loads(response.read())
        files = version_json["files"]

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
        train_io = BytesIO()
        train_io.write(json.dumps(train_files).encode("utf-8"))
        train_io.seek(0)
        minio_client.put_object(
            req.minio_bucket,
            f"training/{req.data_project_id}/{req.dataset_id}/{req.training_project_id}/datasets/train_dataset.json",
            train_io,
            length=train_io.getbuffer().nbytes
        )
        train_io.close()

        val_io = BytesIO()
        val_io.write(json.dumps(val_files).encode("utf-8"))
        val_io.seek(0)
        minio_client.put_object(
            req.minio_bucket,
            f"training/{req.data_project_id}/{req.dataset_id}/{req.training_project_id}/datasets/val_dataset.json",
            val_io,
            length=val_io.getbuffer().nbytes
        )
        val_io.close()
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"MinIO upload error: {str(e)}")

    try:
        logging.info(f"Starting training with model: {req.model_name}, num_class: {req.num_class}, image_size: {req.hyperparameters.get('image_size', 224)}")
        model_generator = generator(
                        name=req.model_name, num_class=req.num_class, 
                        image_size=int(req.hyperparameters.get("image_size", 224)),
                        run_name=f"{req.model_name}_{req.version_tag}"
        )
        model_generator.training_project_id = req.training_project_id
        model_generator.save_path = f"training/{req.data_project_id}/{req.dataset_id}/{req.training_project_id}/results"
        model_generator.data_project_id = req.data_project_id
        model_generator.dataset_id = req.dataset_id



        executor = Executor(
            path_dataset=f"datasets/{req.training_project_id}/{req.dataset_id}",
            batch_size = int(req.hyperparameters.get("batch_size", 4)),
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
    return ["ResNet18", "MobileNetV3Small", "ResNet18_CBAM"]


class PredictRequest(BaseModel):
    model_path: str
    image_path: str
    model_type: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str

def load_model_from_minio_with_generator(minio_client, bucket, model_path, model_type):
    response = minio_client.get_object(bucket, model_path)
    model_bytes = response.read()
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    model_generator = generator(name=model_type, num_class=2, image_size=224, run_name="predict_run")
    model = model_generator.model
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image_from_minio(minio_client, bucket, image_path):
    response = minio_client.get_object(bucket, image_path)
    image = Image.open(io.BytesIO(response.read())).convert("RGB")
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


@app.post("/predict")
async def predict(req: PredictRequest):
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
