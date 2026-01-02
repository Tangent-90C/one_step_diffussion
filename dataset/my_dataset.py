import sys
import glob
import os
import csv
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.realesrgan import RealESRGAN_degradation


class CSVPairsDataset(torch.utils.data.Dataset):
    """Two-column paired dataset from CSV.

    The CSV should contain either:
    - a header row with columns: clean,rainy
    - or no header (then column0=clean, column1=rainy)

    For de-raining training we return (rainy, clean):
      lq_img = rainy (model input)
      hq_img = clean (target)
    """

    def __init__(self, csv_path: str, resolution: int, base_dir: str | None = None):
        super().__init__()
        self.resolution = int(resolution)
        self.csv_path = csv_path

        csv_p = os.path.abspath(csv_path)
        if base_dir is None:
            base_dir = os.path.dirname(csv_p)
        self.base_dir = os.path.abspath(base_dir)

        self.pairs: list[tuple[str, str]] = []  # (clean_path, rainy_path)
        self._load_pairs(csv_p)

    def _resolve_path(self, p: str) -> str:
        p = p.strip()
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return os.path.abspath(os.path.join(self.base_dir, p))

    def _load_pairs(self, csv_p: str) -> None:
        if not os.path.isfile(csv_p):
            raise FileNotFoundError(f"CSV not found: {csv_p}")

        with open(csv_p, "r", encoding="utf-8") as f:
            # Peek first row to detect header.
            first = f.readline()
            if not first:
                raise ValueError(f"Empty CSV: {csv_p}")
            f.seek(0)

            sniff = [s.strip().lower() for s in first.split(",")]
            has_header = len(sniff) >= 2 and (sniff[0] == "clean" and sniff[1] == "rainy")

            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    clean = self._resolve_path(row.get("clean", ""))
                    rainy = self._resolve_path(row.get("rainy", ""))
                    if clean and rainy:
                        self.pairs.append((clean, rainy))
            else:
                reader2 = csv.reader(f)
                for row in reader2:
                    if len(row) < 2:
                        continue
                    clean = self._resolve_path(row[0])
                    rainy = self._resolve_path(row[1])
                    if clean and rainy:
                        self.pairs.append((clean, rainy))

        if not self.pairs:
            raise ValueError(f"No valid (clean,rainy) pairs found in CSV: {csv_p}")

    def __len__(self):
        return len(self.pairs)

    def _paired_preproc(self, img_a: Image.Image, img_b: Image.Image):
        # Pad if needed (try reflect to match existing code; fallback to edge if reflect fails).
        res = self.resolution
        w, h = img_a.size
        pad_w = max(0, res - w)
        pad_h = max(0, res - h)
        if pad_w > 0 or pad_h > 0:
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            padding = [left, top, right, bottom]
            try:
                img_a = F.pad(img_a, padding, padding_mode="reflect")
                img_b = F.pad(img_b, padding, padding_mode="reflect")
            except Exception:
                img_a = F.pad(img_a, padding, padding_mode="edge")
                img_b = F.pad(img_b, padding, padding_mode="edge")

        # Same random crop params for both.
        i, j, th, tw = transforms.RandomCrop.get_params(img_a, output_size=(res, res))
        img_a = F.crop(img_a, i, j, th, tw)
        img_b = F.crop(img_b, i, j, th, tw)

        # Keep resize (no-op after crop, but matches original pipeline behavior).
        img_a = F.resize(img_a, [res, res], interpolation=Image.Resampling.BICUBIC)
        img_b = F.resize(img_b, [res, res], interpolation=Image.Resampling.BICUBIC)

        # Same horizontal flip for both.
        if torch.rand(()) < 0.5:
            img_a = F.hflip(img_a)
            img_b = F.hflip(img_b)

        return img_a, img_b

    def __getitem__(self, idx):
        clean_path, rainy_path = self.pairs[idx]

        clean = Image.open(clean_path).convert("RGB")
        rainy = Image.open(rainy_path).convert("RGB")

        # Apply paired augmentations.
        clean, rainy = self._paired_preproc(clean, rainy)

        to_tensor = transforms.ToTensor()
        clean_t = to_tensor(clean)
        rainy_t = to_tensor(rainy)

        # Scale to [-1, 1]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        clean_t = F.normalize(clean_t, mean=mean, std=std)
        rainy_t = F.normalize(rainy_t, mean=mean, std=std)

        # Return (input, target) = (rainy, clean)
        return rainy_t, clean_t

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_txt_or_dir_paths, resolution):
        super().__init__()
        self.resolution = resolution
        self.degradation = RealESRGAN_degradation(device='cpu', resolution=resolution)
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop(
                (resolution, resolution), 
                pad_if_needed=True,
                padding_mode='reflect'
            ),
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
        ])
        self.gt_list = []
        for p in dataset_txt_or_dir_paths:
            if os.path.isdir(p):
                self.gt_list.extend(glob.glob(f"{p}/*.png") + glob.glob(f"{p}/*.jpg") + glob.glob(f"{p}/*.jpeg"))
            elif os.path.splitext(p)[1] == ".txt":
                with open(p, 'r') as f:
                    self.gt_list.extend([line.strip() for line in f.readlines()])
            else:
                raise ValueError(f"Unsupported path type: {p}. Expected either a directory or a file named 'txt'")
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_path = self.gt_list[idx]
        gt_img = Image.open(gt_path).convert('RGB')
        if 'ffhq' in gt_path and self.resolution == 512:   
            gt_img = gt_img.resize((512, 512), Image.Resampling.LANCZOS)  
        gt_img = self.crop_preproc(gt_img)

        img_gt, img_lq = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
        img_gt, img_lq = img_gt.squeeze(0), img_lq.squeeze(0)

        # input images scaled to -1,1
        img_gt = F.normalize(img_gt, mean=[0.5], std=[0.5])
        # output images scaled to -1,1
        img_lq = F.normalize(img_lq, mean=[0.5], std=[0.5])

        return img_lq, img_gt
            
   