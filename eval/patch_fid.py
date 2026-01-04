from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def image_to_255_scale(images: Tensor, *, dtype: torch.dtype = torch.uint8) -> Tensor:
    """Convert images in [0, 1] float range to 0..255 integer scale.

    This mirrors the behavior expected by torchmetrics' FID/KID implementations,
    which accept uint8 tensors with values in [0, 255].

    Args:
        images: Tensor shaped (N, 3, H, W) in [0, 1].
        dtype: Output dtype, typically torch.uint8.

    Returns:
        Tensor on the same device, converted to dtype, scaled to [0, 255].
    """
    if not torch.is_floating_point(images):
        images = images.float()

    images = images.clamp(0.0, 1.0)
    images_255 = torch.round(images * 255.0)
    return images_255.to(dtype=dtype)


def update_patch_fid(
    input_images: Tensor,
    pred: Tensor,
    *,
    fid_metric: Optional[object] = None,
    fid_swav_metric: Optional[object] = None,
    kid_metric: Optional[object] = None,
    patch_size: int = 256,
) -> int:
    """Update FID/KID metrics using the FID/256 patch protocol.

    This reimplements `neuralcompression.metrics.update_patch_fid` so this repo
    no longer depends on the (often hard-to-install) `neuralcompression` package.

    The method is described in:
      High-Fidelity Generative Image Compression (Mentzer et al.)

    Behavior:
      1) Divide each image into a grid of non-overlapping patches of size
         `patch_size` with stride `patch_size`, treat patches as images, update.
      2) If the image is large enough, repeat with a half-patch shift.

    Args:
        input_images: Ground truth images in [0, 1], shape (N, 3, H, W).
        pred: Reconstructed images in [0, 1], shape (N, 3, H, W).
        fid_metric: torchmetrics-style metric with `.update(images, real=bool)`.
        fid_swav_metric: Same update API as `fid_metric` (optional).
        kid_metric: torchmetrics-style metric with `.update(images, real=bool)`.
        patch_size: Patch size (default 256).

    Returns:
        Total number of patches used to update metrics.
    """
    if fid_metric is None and kid_metric is None and fid_swav_metric is None:
        raise ValueError("At least one metric must not be None.")

    # Non-overlapping grid of patches.
    real = image_to_255_scale(
        F.unfold(input_images, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    fake = image_to_255_scale(
        F.unfold(pred, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )

    patch_count = int(real.shape[0])
    if fid_metric is not None:
        fid_metric.update(real, real=True)
        fid_metric.update(fake, real=False)
    if fid_swav_metric is not None:
        fid_swav_metric.update(real, real=True)
        fid_swav_metric.update(fake, real=False)
    if kid_metric is not None:
        kid_metric.update(real, real=True)
        kid_metric.update(fake, real=False)

    # Half-patch shift pass (only if there's room).
    num_y, num_x = int(input_images.shape[2]), int(input_images.shape[3])
    if num_y >= 1.5 * patch_size and num_x >= 1.5 * patch_size:
        real = image_to_255_scale(
            F.unfold(
                input_images[:, :, patch_size // 2 :, patch_size // 2 :],
                kernel_size=patch_size,
                stride=patch_size,
            )
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size),
            dtype=torch.uint8,
        )
        fake = image_to_255_scale(
            F.unfold(
                pred[:, :, patch_size // 2 :, patch_size // 2 :],
                kernel_size=patch_size,
                stride=patch_size,
            )
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size),
            dtype=torch.uint8,
        )
        patch_count += int(real.shape[0])
        if fid_metric is not None:
            fid_metric.update(real, real=True)
            fid_metric.update(fake, real=False)
        if fid_swav_metric is not None:
            fid_swav_metric.update(real, real=True)
            fid_swav_metric.update(fake, real=False)
        if kid_metric is not None:
            kid_metric.update(real, real=True)
            kid_metric.update(fake, real=False)

    return patch_count
