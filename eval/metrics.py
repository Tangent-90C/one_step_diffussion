import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_path(value: str | os.PathLike | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def find_default_dists_weights() -> Path | None:
    """Locate a DISTS weight file compatible with the original pyiqa metric.

    Priority:
    1) env var `DISTS_WEIGHTS`
    2) legacy pyiqa torch.hub cache path

    Returns None if not found.
    """
    env_path = _as_path(os.environ.get("DISTS_WEIGHTS"))
    if env_path is not None and env_path.is_file():
        return env_path

    # pyiqa used this exact cache path/name in many setups.
    legacy = Path.home() / ".cache" / "torch" / "hub" / "pyiqa" / "DISTS_weights-f5e65c96.pth"
    if legacy.is_file():
        return legacy

    return None


class PSNR(nn.Module):
    def __init__(self, *, data_range: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.data_range = float(data_range)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"PSNR expects same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim != 4:
            raise ValueError(f"PSNR expects NCHW tensors, got {tuple(pred.shape)}")

        mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        # match common psnr definition; avoid inf by eps
        psnr = 10.0 * torch.log10((self.data_range ** 2) / (mse + self.eps))
        return psnr


class MSSSIM(nn.Module):
    """MS-SSIM matching the legacy pyiqa implementation.

    Notes:
    - Uses Y (luminance) channel by default (YIQ luma coefficients).
    - Uses data_range=255 and float64 internally.
    - Forces CS map nonnegative via ReLU (pyiqa behavior).
    """

    def __init__(self, *, data_range: float = 1.0) -> None:
        super().__init__()
        # `data_range` here refers to the incoming tensor range (we expect [0,1]).
        self.input_range = float(data_range)
        self._weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)
        self._win_size = 11
        self._win_sigma = 1.5
        self._data_range_internal = 255.0

    @staticmethod
    def _to_y_channel_yiq(x: torch.Tensor) -> torch.Tensor:
        # Match pyiqa: rgb2yiq + take Y channel.
        yiq_weights = (
            torch.tensor(
                [
                    [0.299, 0.587, 0.114],
                    [0.5959, -0.2746, -0.3213],
                    [0.2115, -0.5227, 0.3112],
                ]
            )
            .t()
            .to(x)
        )
        x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
        return x_yiq[:, [0], :, :]

    def _gaussian_window(self, *, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        win_size = self._win_size
        sigma = self._win_sigma

        coords = torch.arange(win_size, device=device, dtype=dtype) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        return w.repeat(channels, 1, 1, 1)

    @staticmethod
    def _filter2_valid(x: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
        # win is (C,1,kh,kw); do per-channel conv.
        return F.conv2d(x, win, padding=0, groups=x.shape[1])

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, *, win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Implements pyiqa's SSIM core with nonnegative CS map.
        dr = self._data_range_internal
        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2

        mu1 = self._filter2_valid(x, win)
        mu2 = self._filter2_valid(y, win)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._filter2_valid(x * x, win) - mu1_sq
        sigma2_sq = self._filter2_valid(y * y, win) - mu2_sq
        sigma12 = self._filter2_valid(x * y, win) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        cs_map = F.relu(cs_map)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

        ssim_val = ssim_map.mean(dim=(1, 2, 3))
        cs_val = cs_map.mean(dim=(1, 2, 3))
        return ssim_val, cs_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"MS-SSIM expects same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"MS-SSIM expects NCHW with 3 channels, got {tuple(pred.shape)}")

        # Convert to Y channel (YIQ) and scale to [0,255] like legacy pyiqa.
        x = pred.float().clamp(0, self.input_range) / self.input_range
        y = target.float().clamp(0, self.input_range) / self.input_range

        x = self._to_y_channel_yiq(x) * self._data_range_internal
        y = self._to_y_channel_yiq(y) * self._data_range_internal

        # Match pyiqa: use rounded uint8 values (differentiable round).
        x = x - x.detach() + x.round()
        y = y - y.detach() + y.round()

        x = x.to(torch.float64)
        y = y.to(torch.float64)

        win = self._gaussian_window(channels=1, device=x.device, dtype=x.dtype)
        weights = self._weights.to(device=x.device)

        mcs: list[torch.Tensor] = []
        ssim_val = None
        levels = int(weights.numel())

        for _ in range(levels):
            ssim_val, cs = self._ssim(x, y, win=win)
            mcs.append(cs)
            padding = (x.shape[2] % 2, x.shape[3] % 2)
            x = F.avg_pool2d(x, kernel_size=2, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, padding=padding)

        if ssim_val is None:
            raise RuntimeError("MS-SSIM internal error")

        mcs_t = torch.stack(mcs, dim=0)  # (levels, N)
        # pyiqa uses product form by default.
        out = torch.prod((mcs_t[:-1] ** weights[:-1].unsqueeze(1)), dim=0) * (ssim_val ** weights[-1])
        return out


class _L2Pooling(nn.Module):
    def __init__(self, channels: int, filter_size: int = 5, stride: int = 2) -> None:
        super().__init__()
        # For filter_size=5, padding should be 1 (matches common DISTS impl).
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = int(channels)

        # Match the legacy implementation: numpy.hanning(filter_size)[1:-1]
        import numpy as np

        a = np.hanning(filter_size)[1:-1]
        g = torch.tensor(a[:, None] * a[None, :], dtype=torch.float32)
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.pow(2)
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + 1e-12).sqrt()


@dataclass(frozen=True)
class DISTSConfig:
    c1: float = 1e-6
    c2: float = 1e-6


class DISTS(nn.Module):
    """Deep Image Structure and Texture Similarity (DISTS).

    This is a local reimplementation intended to be compatible with the original
    pyiqa `dists` metric when using the same alpha/beta weights.

    Inputs are expected to be float tensors in [0,1] with shape (N,3,H,W).
    """

    def __init__(self, *, weights_path: str | os.PathLike | None = None, config: DISTSConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DISTSConfig()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        try:
            from torchvision.models import vgg16, VGG16_Weights
        except Exception as e:  # pragma: no cover
            raise ImportError("torchvision is required for DISTS") from e

        vgg_feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Match pyiqa's stage partitioning exactly.
        self.stage1 = nn.Sequential(*[vgg_feats[x] for x in range(0, 4)])
        self.stage2 = nn.Sequential(_L2Pooling(channels=64), *[vgg_feats[x] for x in range(5, 9)])
        self.stage3 = nn.Sequential(_L2Pooling(channels=128), *[vgg_feats[x] for x in range(10, 16)])
        self.stage4 = nn.Sequential(_L2Pooling(channels=256), *[vgg_feats[x] for x in range(17, 23)])
        self.stage5 = nn.Sequential(_L2Pooling(channels=512), *[vgg_feats[x] for x in range(24, 30)])

        for p in self.parameters():
            p.requires_grad_(False)

        wp = _as_path(weights_path) or find_default_dists_weights()
        if wp is None or not wp.is_file():
            raise FileNotFoundError(
                "DISTS weights not found. Set env var `DISTS_WEIGHTS` to the alpha/beta .pth file "
                "(e.g. DISTS_weights-f5e65c96.pth)."
            )

        state = torch.load(str(wp), map_location="cpu")
        if not isinstance(state, dict) or "alpha" not in state or "beta" not in state:
            raise ValueError(f"Unexpected DISTS weight format at: {wp}")

        alpha = state["alpha"].detach().float()
        beta = state["beta"].detach().float()
        if alpha.shape != beta.shape:
            raise ValueError(f"alpha/beta shape mismatch: {tuple(alpha.shape)} vs {tuple(beta.shape)}")
        if alpha.shape[1] != 1475:
            raise ValueError(f"Expected 1475-channel alpha/beta, got {tuple(alpha.shape)}")

        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

        self.chns = [3, 64, 128, 256, 512, 512]

    def forward_once(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Match pyiqa: keep raw input `x` as the first feature (not normalized).
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"DISTS expects same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"DISTS expects NCHW with 3 channels, got {tuple(pred.shape)}")

        x = pred.float().clamp(0, 1)
        y = target.float().clamp(0, 1)

        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        dist1 = 0.0
        dist2 = 0.0
        c1 = self.cfg.c1
        c2 = self.cfg.c2

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha_parts = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta_parts = torch.split(self.beta / w_sum, self.chns, dim=1)

        for k in range(len(self.chns)):
            fx = feats_x[k]
            fy = feats_y[k]

            x_mean = fx.mean(dim=(2, 3), keepdim=True)
            y_mean = fy.mean(dim=(2, 3), keepdim=True)
            s1 = (2 * x_mean * y_mean + c1) / (x_mean.pow(2) + y_mean.pow(2) + c1)
            dist1 = dist1 + (alpha_parts[k] * s1).sum(dim=1, keepdim=True)

            x_var = (fx - x_mean).pow(2).mean(dim=(2, 3), keepdim=True)
            y_var = (fy - y_mean).pow(2).mean(dim=(2, 3), keepdim=True)
            xy_cov = (fx * fy).mean(dim=(2, 3), keepdim=True) - x_mean * y_mean
            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta_parts[k] * s2).sum(dim=1, keepdim=True)

        score = 1.0 - (dist1 + dist2)
        return score.squeeze(-1).squeeze(-1)
