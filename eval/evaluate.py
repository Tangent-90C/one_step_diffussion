import sys
import os
import argparse
import csv
import tqdm
import pyiqa
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Local reimplementation (removes neuralcompression dependency)
_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
from patch_fid import update_patch_fid

os.environ['HF_HOME'] = '/mnt/HDD-data/jianuo/cache'


class _SkimageSSIM:
    def __init__(self) -> None:
        try:
            from skimage.metrics import structural_similarity as ssim
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "scikit-image is required for SSIM. Install it with: pip install scikit-image"
            ) from e
        self._ssim = ssim

    def __call__(self, recon_batch: torch.Tensor, gt_batch: torch.Tensor) -> torch.Tensor:
        if recon_batch.shape != gt_batch.shape:
            raise ValueError(f"SSIM expects same shape, got {tuple(recon_batch.shape)} vs {tuple(gt_batch.shape)}")
        if recon_batch.ndim != 4 or recon_batch.shape[1] != 3:
            raise ValueError(f"SSIM expects NCHW with 3 channels, got {tuple(recon_batch.shape)}")

        scores: list[float] = []
        n = int(recon_batch.shape[0])
        for i in range(n):
            gt = (
                gt_batch[i]
                .detach()
                .float()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            recon = (
                recon_batch[i]
                .detach()
                .float()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            # Newer skimage uses channel_axis; older versions use multichannel.
            try:
                score = float(self._ssim(gt, recon, channel_axis=-1, data_range=1.0))
            except TypeError:
                score = float(self._ssim(gt, recon, multichannel=True, data_range=1.0))
            scores.append(score)

        return torch.tensor(scores)


def _load_pairs_csv(pairs_csv: str | Path) -> list[tuple[Path, Path]]:
    """Load paired image paths from a CSV.

    Expected headers: at least two columns named 'clean' and 'rainy'.
    Returns list of (clean_path, rainy_path).
    """
    pairs_csv = Path(pairs_csv)
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"pairs_csv not found: {pairs_csv}")

    pairs: list[tuple[Path, Path]] = []
    with pairs_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"pairs_csv has no header row: {pairs_csv}")
        fields = {name.strip(): name for name in reader.fieldnames if name is not None}

        if "clean" not in fields or "rainy" not in fields:
            raise ValueError(
                f"pairs_csv must contain columns 'clean' and 'rainy', got: {reader.fieldnames}"
            )
        clean_key = fields["clean"]
        rainy_key = fields["rainy"]

        for row in reader:
            clean = (row.get(clean_key) or "").strip()
            rainy = (row.get(rainy_key) or "").strip()
            if not clean or not rainy:
                continue
            pairs.append((Path(clean), Path(rainy)))

    if not pairs:
        raise ValueError(f"No valid pairs found in pairs_csv: {pairs_csv}")
    return pairs


def _sum_metric_output(value) -> float:
    """Convert a metric output (tensor/float) into a sum over the batch."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return float(value.item())
        return float(value.sum().item())
    return float(value)


def _load_image_tensor_rgb(path: str | Path, *, totensor: ToTensor, device: torch.device) -> torch.Tensor:
    with open(str(path), "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
    return totensor(image).unsqueeze(0).to(device)


def _pad_to_min_size(x: torch.Tensor, *, min_size: int) -> torch.Tensor:
    """Pad NCHW tensor to at least min_size using replicate padding."""
    _, _, h, w = x.shape
    if h >= min_size and w >= min_size:
        return x
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="replicate")


def eval_batch(
    *,
    recon_batch: torch.Tensor,
    gt_batch: torch.Tensor | None,
    metric_dict: dict,
    metric_paired_dict: dict,
    fid_metric=None,
    kid_metric=None,
    use_amp: bool = True,
    patch_size: int = 256,
    min_size: int = 256,
) -> tuple[dict[str, float], int]:
    """Evaluate a batch of tensors and return summed metric values.

    This is the reusable batch interface for other code.

    Args:
        recon_batch: (N,3,H,W) float tensor in [0,1].
        gt_batch: (N,3,H,W) float tensor in [0,1], or None.
        metric_dict: metrics that take recon_batch only.
        metric_paired_dict: metrics that take (recon_batch, gt_batch).
        fid_metric/kid_metric: torchmetrics objects to be updated in-place.
        use_amp: whether to use autocast for unpaired metrics.
        patch_size: patch size used for patch-FID/KID updates.
        min_size: minimum spatial size required for patch-FID/KID.

    Returns:
        (result_sums, batch_size)
    """
    if recon_batch.ndim != 4:
        raise ValueError(f"recon_batch must be NCHW, got shape {tuple(recon_batch.shape)}")
    if gt_batch is not None and gt_batch.ndim != 4:
        raise ValueError(f"gt_batch must be NCHW, got shape {tuple(gt_batch.shape)}")
    if gt_batch is not None and recon_batch.shape != gt_batch.shape:
        raise ValueError(
            f"recon_batch and gt_batch must have same shape, got {tuple(recon_batch.shape)} vs {tuple(gt_batch.shape)}"
        )

    result: dict[str, float] = {}
    batch_n = int(recon_batch.shape[0])

    if metric_dict:
        ctx = torch.cuda.amp.autocast() if use_amp else torch.autocast(device_type="cuda", enabled=False)
        with ctx:
            for key, metric in metric_dict.items():
                value = metric(recon_batch)
                result[key] = result.get(key, 0.0) + _sum_metric_output(value)

    if gt_batch is not None:
        # Patch-FID/KID update (requires >= min_size)
        if fid_metric is not None or kid_metric is not None:
            gt_p = _pad_to_min_size(gt_batch, min_size=min_size)
            recon_p = _pad_to_min_size(recon_batch, min_size=min_size)
            update_patch_fid(gt_p, recon_p, fid_metric=fid_metric, kid_metric=kid_metric, patch_size=patch_size)

        for key, metric in metric_paired_dict.items():
            recon_in = recon_batch
            gt_in = gt_batch

            # Some metrics (notably MS-SSIM) require a minimum spatial size.
            # pyiqa's ms_ssim uses an 11x11 window across multiple scales; pad to a safe size.
            if key == "ms_ssim":
                ms_ssim_min_size = 176  # 11 * 2**4 for 5-scale MS-SSIM
                gt_in = _pad_to_min_size(gt_in, min_size=ms_ssim_min_size)
                recon_in = _pad_to_min_size(recon_in, min_size=ms_ssim_min_size)

            try:
                value = metric(recon_in, gt_in)
            except RuntimeError as e:
                # Fallback for unexpected small-size cases: pad and retry once.
                msg = str(e)
                if key == "ms_ssim" and ("Kernel size can't be greater than actual input size" in msg or "Kernel size" in msg):
                    gt_in = _pad_to_min_size(gt_batch, min_size=176)
                    recon_in = _pad_to_min_size(recon_batch, min_size=176)
                    value = metric(recon_in, gt_in)
                else:
                    raise
            result[key] = result.get(key, 0.0) + _sum_metric_output(value)

    return result, batch_n



def evaluate(recon_dir, gt_dir, ntest, pairs_csv: str | Path | None = None):

    device = torch.device("cuda")
    totensor = ToTensor()

    metric_dict = {}
    # metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    # metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    # metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    # metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}
    recon_dir = Path(recon_dir) if not isinstance(recon_dir, Path) else recon_dir
    assert recon_dir.is_dir()

    # Index recon images by file name for fast lookup (supports nested folders).
    recon_path_list_all = sorted([x for x in recon_dir.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    recon_index: dict[str, Path] = {}
    for p in recon_path_list_all:
        name = p.name
        if name in recon_index:
            raise AssertionError(f"Duplicated recon image name found: {name}")
        recon_index[name] = p
    
    gt_index = None
    if gt_dir is not None:
        gt_dir = Path(gt_dir) if not isinstance(gt_dir, Path) else gt_dir
        gt_path_list = sorted([x for x in gt_dir.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if ntest is not None:
            gt_path_list = gt_path_list[:ntest]
        # index by file name for fast pairing
        gt_index = {}
        for p in gt_path_list:
            name = p.name
            if name in gt_index:
                raise AssertionError(f"Duplicated GT image name found: {name}")
            gt_index[name] = p

    # Build evaluation pairs.
    eval_pairs: list[tuple[Path, Path]] = []  # (recon_path, gt_path)

    if pairs_csv is not None:
        pairs = _load_pairs_csv(pairs_csv)
        missing_recon: list[str] = []
        missing_gt: list[str] = []

        # CSV defines mapping: rainy (input/recon name) -> clean (gt).
        for clean_path, rainy_path in pairs:
            recon_name = rainy_path.name
            if recon_name not in recon_index:
                missing_recon.append(recon_name)
                continue
            recon_path = recon_index[recon_name]

            gt_path: Path
            if gt_index is not None and clean_path.name in gt_index:
                gt_path = gt_index[clean_path.name]
            else:
                gt_path = clean_path
                if not gt_path.is_file():
                    missing_gt.append(str(gt_path))
                    continue

            eval_pairs.append((recon_path, gt_path))

        if not eval_pairs:
            raise AssertionError(
                f"No valid eval pairs found. missing recon: {len(missing_recon)}, missing gt: {len(missing_gt)}"
            )
        if missing_recon:
            print(f"Warning: {len(missing_recon)} rainy images in pairs_csv not found in recon_dir; they will be skipped.")
        if missing_gt:
            print(f"Warning: {len(missing_gt)} clean images in pairs_csv not found on disk; they will be skipped.")
    else:
        # Original behavior: pair by same file name between recon_dir and gt_dir.
        recon_path_list = recon_path_list_all
        if ntest is not None:
            recon_path_list = recon_path_list[:ntest]
        if gt_index is not None:
            for p in recon_path_list:
                base_name = p.name
                if base_name not in gt_index:
                    raise AssertionError(f"GT image for {base_name} not found!")
                eval_pairs.append((p, gt_index[base_name]))
        else:
            # No GT provided; only unpaired metrics could be computed.
            for p in recon_path_list:
                eval_pairs.append((p, None))

    # Initialize paired metrics if GT exists (either via gt_dir or pairs_csv).
    has_gt = any(gt is not None for _, gt in eval_pairs)
    if has_gt:
        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device)
        metric_paired_dict["ssim"] = _SkimageSSIM()
        metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)  # lpips-alexnet
        fid_metric = FrechetInceptionDistance().to(device)
        kid_metric = KernelInceptionDistance().to(device)
    else:
        fid_metric = None
        kid_metric = None

    print(f"Find {len(recon_path_list_all)} images in {recon_dir}")
    if pairs_csv is not None:
        print(f"Evaluate {len(eval_pairs)} pairs from pairs_csv: {pairs_csv}")
    elif not has_gt:
        print("GT dir not provided, skipping paired metrics.")

    # Keep original behavior by default: process one image at a time.
    batch_size = 1
    result: dict[str, float] = {}
    total_images = 0

    for start in tqdm.tqdm(range(0, len(eval_pairs), batch_size)):
        batch_pairs = eval_pairs[start : start + batch_size]
        recon_paths = [p for p, _ in batch_pairs]
        recon_tensors = [_load_image_tensor_rgb(p, totensor=totensor, device=device) for p in recon_paths]
        recon_batch = torch.cat(recon_tensors, dim=0)

        gt_batch = None
        if has_gt:
            gt_tensors = []
            for _, gt_path in batch_pairs:
                if gt_path is None:
                    raise AssertionError("Internal error: expected GT path but got None")
                gt_tensors.append(_load_image_tensor_rgb(gt_path, totensor=totensor, device=device))
            gt_batch = torch.cat(gt_tensors, dim=0)

        batch_result, batch_n = eval_batch(
            recon_batch=recon_batch,
            gt_batch=gt_batch,
            metric_dict=metric_dict,
            metric_paired_dict=metric_paired_dict,
            fid_metric=fid_metric,
            kid_metric=kid_metric,
            use_amp=True,
            patch_size=256,
            min_size=256,
        )
        total_images += batch_n
        for k, v in batch_result.items():
            result[k] = result.get(k, 0.0) + float(v)

    
    if has_gt and fid_metric is not None and kid_metric is not None and total_images > 50:
        result['fid'] = float(fid_metric.compute())
        kid_tuple = kid_metric.compute()
        result['kid_mean'], result['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        elif key == 'kid_mean' or key == 'kid_std':
            print(f"{key}: {res:.7f}")
            print_results.append(f"{key}: {res:.7f}")
        else:
            denom = max(1, total_images)
            print(f"{key}: {res/denom:.5f}")
            print_results.append(f"{key}: {res/denom:.5f}")
    return print_results


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example evaluation script.")
    parser.add_argument("--recon_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument(
        "--pairs_csv",
        type=str,
        default=None,
        help="Optional CSV defining clean/rainy pairing (e.g. gtrain_pairs_val.csv). If set, pairs are taken from CSV (rainy->clean).",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print_results = evaluate(args.recon_dir, args.gt_dir, None, pairs_csv=args.pairs_csv)

if __name__ == "__main__":
    main(sys.argv[1:])