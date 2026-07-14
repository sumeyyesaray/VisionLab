import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for a single conv layer.

    Hooks `target_layer`'s forward activations and backward gradients, then
    weights each activation channel by the global-average-pooled gradient of
    the target class score — the standard Grad-CAM recipe. Works for any
    `nn.Module` with a convolutional target layer, not just ResNet.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, target_class: int | None = None):
        """`image_tensor` is a single (un-batched) normalized image, shape (C, H, W).

        Returns `(cam, target_class, confidence)` — `cam` is an (H, W) numpy
        array in [0, 1], resized to the input image's resolution.
        """
        self.model.eval()
        batched = image_tensor.unsqueeze(0)
        batched.requires_grad_(True)

        output = self.model(batched)
        probs = output.softmax(dim=1)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=batched.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, float(probs[0, target_class].item())
