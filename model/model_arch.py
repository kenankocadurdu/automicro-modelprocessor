import os
import numpy as np
import logging
import json
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Module(nn.Module):
    """
    Base model class with shared training and validation logic.
    """

    def training_step(self, batch, criterion):
        images, labels = batch
        out = self(images)
        loss = self.calculate_loss(out, labels, criterion)
        return loss, 0

    def validation_step(self, batch, criterion):
        images, labels = batch
        out = self(images)
        loss = self.calculate_loss(out, labels, criterion)
        acc = calculate_metrics(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': torch.tensor(acc)}

    def calculate_loss(self, out, labels, criterion):
        if criterion == "CrossEntropyLoss":
            if out.shape[1] == 1:
                raise ValueError("CrossEntropyLoss expects num_classes > 1")
            return nn.CrossEntropyLoss()(out, labels.long())
        elif criterion == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()(out, labels.float())
        elif criterion == "FocalLoss":
            alpha, gamma = 0.25, 2
            ce_loss = nn.CrossEntropyLoss()(out, labels.long())
            pt = torch.exp(-ce_loss)
            return (alpha * (1 - pt) ** gamma * ce_loss).mean()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        acc_scores = [x['val_acc'] for x in outputs]
        return {
            'val_loss': torch.stack(batch_losses).mean().item(),
            'val_acc': torch.stack(acc_scores).mean().item()
        }

    def epoch_end(self, epoch, result):
        logging.info(f"Epoch [{epoch}] | Train Loss: {result['train_loss']:.4f} | Val Loss: {result['val_loss']:.4f} | Accuracy: {result['val_acc']:.4f}")

def calculate_metrics(pred, target):
    _, preds = torch.max(pred, dim=1)
    correct = (preds == target).float()
    return correct.sum() / len(correct)

def get_optimizer(opt_func, parameters, lr):
    if opt_func == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr, alpha=0.9)
    elif opt_func == 'SGD':
        return torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    elif opt_func == 'Adam':
        return torch.optim.Adam(parameters, lr)
    elif opt_func == 'AdamW':
        return torch.optim.AdamW(parameters, lr)
    elif opt_func == 'Adagrad':
        return torch.optim.Adagrad(parameters, lr)
    else:
        raise ValueError(f"Unknown optimizer function: {opt_func}")

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    return [to_device(x, device) for x in data] if isinstance(data, (list, tuple)) else data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

class EarlyStopping:
    """
    Stops training when validation loss does not improve after specified patience.
    Saves best model and optionally exports ONNX.
    """

    def __init__(self, patience=10, delta=1e-7, pth_name="best_model", save_path="results", model_generator=None, device=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(os.getcwd(), save_path, pth_name)
        self.model_generator = model_generator
        self.device = device

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None or score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        logging.info(f"Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}). Saving model...")
        torch.save(model.state_dict(), self.path + ".pth")
        self.val_loss_min = val_loss

@torch.no_grad()
def evaluate(model, val_loader, criterion):
    model.eval()
    outputs = [model.validation_step(batch, criterion) for batch in val_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def evaluate_best_model(model_path, val_loader, device, minio_client, minio_bucket, result_dir, model_generator):
    model = model_generator.model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(all_labels, all_preds, output_dict=True)

    evaluation_results = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }

    eval_path = os.path.join(result_dir, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    upload_to_minio(minio_client, minio_bucket, eval_path, f"{result_dir}/evaluation.json")

def plot_training_progress(history, save_path):
    epochs = range(1, len(history) + 1)
    val_loss = [entry['val_loss'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.plot(epochs, train_loss, 'g-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def fit(epochs: int, lr: float, model_generator, train_loader, val_loader, opt_func: str, criterion: str,
        batch_size: int, accumulation_steps: int, chunk_size: int, patience: int,
        minio_client=None, minio_bucket=None):

    device = get_default_device()
    model = to_device(model_generator.model, device)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    early_stopping = EarlyStopping(patience=patience, save_path=model_generator.save_path,
                                   model_generator=model_generator, device=device)

    optimizer = get_optimizer(opt_func, model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, min_lr=1e-7)

    try:
        history = []
        for epoch in range(int(epochs)):
            model.train()
            logging.info(f"Epoch [{epoch}/{epochs}] started with lr = {optimizer.param_groups[0]['lr']:.10f}")

            train_losses = []
            for chunk_start in range(0, len(train_loader), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(train_loader))
                chunk_loader = [b for i, b in enumerate(train_loader) if chunk_start <= i < chunk_end]

                pbar = tqdm(chunk_loader, desc=f'Chunk {chunk_start}-{chunk_end}', leave=True)
                for i, batch in enumerate(pbar):
                    optimizer.zero_grad()
                    loss, _ = model.training_step(batch, criterion)
                    loss = loss / accumulation_steps
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    train_losses.append(loss)
                    torch.cuda.empty_cache()

            mean_train_loss = torch.stack(train_losses).mean()
            result = evaluate(model, val_loader, criterion)
            result['train_loss'] = mean_train_loss.item()
            model.epoch_end(epoch, result)
            history.append(result)

            early_stopping(result['val_loss'], model)
            scheduler.step(result['val_loss'])

            if early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                evaluate_best_model(
                    model_path=os.path.join(model_generator.save_path, 'best_model.pth'),
                    val_loader=val_loader,
                    device=device,
                    minio_client=minio_client,
                    minio_bucket=minio_bucket,
                    result_dir=model_generator.save_path,
                    model_generator=model_generator
                )
                break

    except Exception as e:
        logging.exception("Training failed due to an unexpected error.")
        raise

    plot_training_progress(history, model_generator.save_path)

    evaluate_best_model(
        model_path=os.path.join(model_generator.save_path, 'best_model.pth'),
        val_loader=val_loader,
        device=device,
        minio_client=minio_client,
        minio_bucket=minio_bucket,
        result_dir=model_generator.save_path,
        model_generator=model_generator
    )

    history_path = os.path.join(model_generator.save_path, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    if minio_client and minio_bucket:
        for filename in ["loss_plot.png", "history.json", "best_model.pth", "evaluation.json"]:
            local_file = os.path.join(model_generator.save_path, filename)
            remote_file = f"{model_generator.save_path}/{filename}"
            upload_to_minio(minio_client, minio_bucket, local_file, remote_file)
            if os.path.exists(local_file):
                os.remove(local_file)
        if os.path.isdir(model_generator.save_path) and not os.listdir(model_generator.save_path):
            os.rmdir(model_generator.save_path)

def upload_to_minio(minio_client, bucket, local_path, remote_path):
    with open(local_path, "rb") as file_data:
        minio_client.put_object(
            bucket,
            remote_path,
            file_data,
            length=os.path.getsize(local_path)
        )
