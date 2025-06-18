import torch
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Generates heatmaps for visual explanation of CNN decisions.
    """

    def __init__(self, model, target_layer=None):
        if not isinstance(target_layer, torch.nn.Module):
            raise ValueError("target_layer must be a torch.nn.Module (e.g., model.layer4[1])")

        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate the Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): Input image tensor with shape [1, C, H, W].
            target_class (int, optional): Class index to generate CAM for.

        Returns:
            np.ndarray: Normalized CAM heatmap (H x W).
        """
        input_tensor.requires_grad_(True)

        with torch.set_grad_enabled(True):
            self.model.zero_grad()
            output = self.model(input_tensor)
            if isinstance(output, dict):
                output = output["out"]

            score = output[:, target_class].sum() if target_class is not None else output.sum()
            score.backward()

            # Compute Grad-CAM weights and map
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalize CAM to [0, 1]
            cam_min = cam.flatten(1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            cam_max = cam.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            # Resize to original input size
            cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            cam = cam.detach().cpu()

            if cam.ndim == 4:
                cam = cam.squeeze(0)

            return cam.squeeze().numpy()
