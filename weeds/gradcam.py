"""Grad-CAM: visual explanation of which image regions drove the CNN's prediction.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization", ICCV 2017.  heatmap = ReLU( sum_k  mean(dY_c/dA_k) * A_k )
"""
import torch
import torch.nn.functional as F


def target_layer(model, arch):
    """Last convolutional feature block of each architecture."""
    if arch == "shufflenet_v2":
        return model.conv5
    return model.features  # weednet_lite, mobilenet_v3_*, efficientnet_b0, squeezenet


def grad_cam(model, arch, x, class_idx=None):
    """x: (1,3,H,W) normalised tensor. Returns (H,W) heatmap in [0,1] and class index."""
    acts, grads = {}, {}
    layer = target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = model(x)
            if class_idx is None:
                class_idx = int(out.argmax(1))
            out[0, class_idx].backward()
    finally:
        h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), class_idx
