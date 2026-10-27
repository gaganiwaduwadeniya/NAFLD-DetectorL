"""Optional Grad-CAM for one explicitly identified ensemble member."""

import numpy as np
import torch
from torch.nn import functional as F
from nafld.utils.config import TASK_LABELS


def grad_cam(model, pixels, *, target_index, layer=None):
    if pixels.ndim != 4 or pixels.shape[0] != 1:
        raise ValueError('Grad-CAM accepts exactly one image')
    if layer is None:
        convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        if not convolutions:
            raise ValueError('No convolutional target layer found; specify a compatible layer')
        layer = convolutions[-1]
    captured = []
    def capture(module, inputs, output):
        captured.append(output)
    hook = layer.register_forward_hook(capture)
    prior = model.training
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(pixels.detach().clone().requires_grad_(True))
            if len(captured) != 1 or captured[0].ndim != 4 or not 0 <= target_index < logits.shape[1]:
                raise ValueError('Grad-CAM requires one spatial activation and a valid output target')
            activation = captured[0]
            gradient, = torch.autograd.grad(logits[0, target_index], activation)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights*activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=pixels.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam-cam.min()
            if cam.max().item() > 0:
                cam = cam/cam.max()
            return cam.detach().float().cpu().numpy()
    finally:
        hook.remove()
        model.train(prior)


def save_explanation(predictor, image_path, output, *, member=0, target_index=1):
    from pathlib import Path
    from PIL import Image, ImageOps
    from nafld.training.plots import _pyplot, _save

    model, preprocessor = predictor.models[member], predictor.preprocessors[member]
    prepared = preprocessor.load(image_path)
    pixels = torch.from_numpy(prepared.pixels).unsqueeze(0).to(predictor.device)
    heatmap = grad_cam(model, pixels, target_index=target_index)
    left, top, right, bottom = prepared.geometry['padding_ltrb']
    height, width = heatmap.shape
    unpadded = heatmap[top:height-bottom if bottom else height, left:width-right if right else width]
    with Image.open(image_path) as original:
        original = ImageOps.exif_transpose(original).convert('RGB')
        spatial = np.asarray(Image.fromarray(unpadded).resize(original.size, Image.Resampling.BILINEAR))
        plt = _pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].imshow(original)
        axes[0].set_title('Supplied image')
        axes[1].imshow(original)
        axes[1].imshow(spatial, cmap='jet', alpha=.4, vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM: one model member')
        for ax in axes:
            ax.axis('off')
        _save(fig, output, {'model_version': predictor.version, 'experiment_id': predictor.member_ids[member],
              'target_index': target_index, 'label': TASK_LABELS[predictor.task][target_index],
              'geometry': prepared.geometry, 'note': 'Qualitative gradient explanation, not localization ground truth or clinical evidence.'})
    np.save(Path(output).with_suffix('.npy'), spatial)


