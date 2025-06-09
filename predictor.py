import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms
from xai.gradcam import GradCAM


class RealtimePredictor:
    def __init__(self, model_generator, device=None):
        self.model_generator = model_generator
        self.model = model_generator.model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = f"{model_generator.save_path}/best_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((model_generator.image_size, model_generator.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, use_gradcam: bool = True):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred_label = output.argmax(dim=1).item()
            confidence = (
                torch.softmax(output, dim=1).max().item()
                if output.shape[1] > 1
                else torch.sigmoid(output).item()
            )

        gradcam_img = None
        if use_gradcam:
            target_layer = get_target_layer(self.model, self.model_generator.name)
            gradcam = GradCAM(self.model, target_layer)
            cam = gradcam.generate(img_tensor)
            cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            image_np = np.array(image.resize((self.model_generator.image_size, self.model_generator.image_size)))
            if image_np.max() <= 1:
                image_np = (image_np * 255).astype(np.uint8)
            gradcam_img = cv2.addWeighted(image_np, 0.5, cam_heatmap, 0.5, 0)

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "gradcam": gradcam_img
        }

    def load_from_minio(self, minio_client, bucket: str, image_key: str) -> Image.Image:
        obj = minio_client.get_object(bucket, image_key)
        img_data = obj.read()
        return Image.open(BytesIO(img_data)).convert("RGB")


def get_target_layer(model, model_name):
    if model_name.startswith("ResNet18"):
        return model.backbone.layer4[-1]
    elif model_name.startswith("MobileNetV3Small"):
        return model.backbone.features[-1]
    else:
        raise ValueError(f"No target layer mapping defined for model: {model_name}")