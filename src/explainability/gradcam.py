import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMPP_CNN:
    # hooks the last EfficientNet conv block — gives [B, C, H, W] activations
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = self.grads = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):
        self.acts = out.detach()

    def _save_grads(self, m, gin, gout):
        self.grads = gout[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def compute(self, tensor, device):
        self.model.eval()
        self.acts = self.grads = None

        x = tensor.to(device).requires_grad_(True)
        self.model(x).backward()

        acts, grads = self.acts, self.grads
        g2 = grads ** 2
        alpha = g2 / (2 * g2 + (acts * grads**3).sum(dim=(2, 3), keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        cam = _norm(cam)
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class GradCAMPP_ViT:
    # GradCAM++ for CLIP ViT-B/16
    # timm gives [B, N, C] with N=197 (CLS + 196 patches), hook on last block's norm1
    # strip CLS, reshape 196 -> 14x14, upsample to 224

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = self.grads = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):
        self.acts = out.detach()

    def _save_grads(self, m, gin, gout):
        self.grads = gout[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def compute(self, tensor, device):
        self.model.eval()
        self.acts = self.grads = None

        x = tensor.to(device).requires_grad_(True)
        self.model(x).backward()

        # drop CLS token, keep the 196 patch tokens
        acts  = self.acts[:, 1:, :]
        grads = self.grads[:, 1:, :]

        g2 = grads ** 2
        alpha = g2 / (2 * g2 + (acts * grads**3).sum(dim=1, keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=1)  # [1, C]
        cam = F.relu((acts * weights.unsqueeze(1)).sum(dim=-1)).squeeze()  # [196]

        cam = _norm(cam.cpu().numpy()).reshape(14, 14)
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class GradCAMPP_Swin:
    # GradCAM++ for Swin-T
    # hook on model.model.norm (final LayerNorm) — fires on full [B, 49, C] sequence
    # 224px -> 4px patch -> 56x56 -> 3 PatchMerge ops -> 7x7 final grid, always

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = self.grads = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):
        self.acts = out.detach()

    def _save_grads(self, m, gin, gout):
        self.grads = gout[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def compute(self, tensor, device):
        self.model.eval()
        self.acts = self.grads = None

        x = tensor.to(device).requires_grad_(True)
        self.model(x).backward()

        acts, grads = self.acts, self.grads
        g2 = grads ** 2
        alpha = g2 / (2 * g2 + (acts * grads**3).sum(dim=1, keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=1)  # [1, C]
        cam = F.relu((acts * weights.unsqueeze(1)).sum(dim=-1)).squeeze()  # [49]

        cam = _norm(cam.cpu().numpy()).reshape(7, 7)  # always 7×7 for swin_tiny at 224px
        return cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _norm(cam):
    lo, hi = cam.min(), cam.max()
    return (cam - lo) / (hi - lo + 1e-8)


def get_gradcam(model, model_name, input_tensor, device):
    if model_name == "efb4":
        gcam = GradCAMPP_CNN(model, model.backbone.blocks[-1])
    elif model_name == "swin":
        gcam = GradCAMPP_Swin(model, model.model.norm)
    elif model_name == "clip":
        gcam = GradCAMPP_ViT(model, model.model.blocks[-1].norm1)
    else:
        raise ValueError(f"unknown model: {model_name}")

    try:
        cam = gcam.compute(input_tensor, device)
    finally:
        gcam.remove()

    return cam


def overlay_heatmap(img_rgb, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    h, w = img_rgb.shape[:2]
    heat_bgr = cv2.applyColorMap((cv2.resize(cam, (w, h)) * 255).astype(np.uint8), colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat_rgb, alpha, 0)
