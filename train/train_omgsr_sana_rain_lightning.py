#!/usr/bin/env python
# coding=utf-8

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset.my_dataset import PairedDataset, CSVPairsDataset
from dinov3_gan.dinov3_convnext_disc import Dinov3ConvNeXtDiscriminator
from dinov3_gan.dinov3_convnext_dists import DINOv3ConvNeXtDISTS
from diffusers.optimization import get_scheduler
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import lightning.pytorch as pl
from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
import omegaconf
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from typing import Callable, Optional
from pathlib import Path
import logging
import argparse
import math
import copy
 


# DINOv3 losses/disc


# Monkeypatch to fix:
# TypeError: SanaCombinedTimestepGuidanceEmbeddings.forward() got an unexpected keyword argument 'batch_size'
try:
    from diffusers.models.transformers.sana_transformer import (
        SanaCombinedTimestepGuidanceEmbeddings,
    )

    _original_forward = SanaCombinedTimestepGuidanceEmbeddings.forward

    def _new_forward(self, timestep, guidance=None, hidden_dtype=None, **kwargs):
        return _original_forward(self, timestep, guidance, hidden_dtype)

    SanaCombinedTimestepGuidanceEmbeddings.forward = _new_forward
except Exception:
    pass


logger = logging.getLogger(__name__)


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    encoded = vae.encode(pixels.to(vae.dtype))
    if hasattr(encoded, "latent_dist"):
        pixel_latents = encoded.latent_dist.sample()
    elif hasattr(encoded, "latent"):
        pixel_latents = encoded.latent
    else:
        raise AttributeError(
            "EncoderOutput has neither latent_dist nor latent")

    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    pixel_latents = (pixel_latents - shift_factor) * scaling_factor
    return pixel_latents.to(weight_dtype)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def get_sana_setting_timesteps(n=999, resolution=1024):
    latent_size = resolution // 32
    return get_schedule(n, latent_size * latent_size, shift=True)


def set_vae_encoder_lora(vae_encoder, rank: int):
    target_modules = r"(^conv_in$|^conv_out$|.*\\.conv1$|.*\\.conv2$|.*\\.conv_shortcut$|.*\\.conv$|.*\\.to_k$|.*\\.to_q$|.*\\.to_v$|.*\\.to_out$|.*\\.conv_inverted$|.*\\.conv_point$)"
    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vae_encoder = PeftModel(
        vae_encoder, vae_encoder_lora_config, adapter_name="vae_encoder_adapter"
    )
    vae_encoder.print_trainable_parameters()
    return vae_encoder


def set_sana_transformer_lora(sana_transformer, rank: int):
    target_modules = ["to_k", "to_q", "to_v"]
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    sana_transformer = PeftModel(
        sana_transformer, transformer_lora_config, adapter_name="sana_adapter"
    )
    sana_transformer.print_trainable_parameters()
    return sana_transformer


class SanaRainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_txt_or_dir_paths,
        resolution: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.dataset_txt_or_dir_paths = dataset_txt_or_dir_paths
        self.resolution = int(resolution)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_dataset = None

    def setup(self, stage: Optional[str] = None):
        if (
            isinstance(
                self.dataset_txt_or_dir_paths,
                (list, tuple, omegaconf.listconfig.ListConfig),
            )
            and len(self.dataset_txt_or_dir_paths) == 1
            and str(self.dataset_txt_or_dir_paths[0]).lower().endswith(".csv")
        ):
            self.train_dataset = CSVPairsDataset(
                self.dataset_txt_or_dir_paths[0], self.resolution
            )
        else:
            self.train_dataset = PairedDataset(
                self.dataset_txt_or_dir_paths, self.resolution)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


class _SyntheticPairsDataset(Dataset):
    def __init__(self, resolution: int, length: int = 1):
        super().__init__()
        self.resolution = int(resolution)
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Match training code expectations: tensors in [-1, 1]
        h = self.resolution
        w = self.resolution
        lq = torch.rand(3, h, w) * 2 - 1
        hq = torch.rand(3, h, w) * 2 - 1
        return lq, hq


class OMGSR_SanaRain_Lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(args, resolve=True))
        self.args = args

        self.automatic_optimization = False

        self.grad_accum_steps = int(
            getattr(self.args, "gradient_accumulation_steps", 1))
        if self.grad_accum_steps < 1:
            self.grad_accum_steps = 1

        # Counts optimizer steps (matches accelerate's global_step semantics under gradient accumulation).
        self.opt_step = 0

        # Will be created in setup() so Lightning can move parameters correctly.
        self.fixed_vae = None
        self.lora_vae = None
        self.sana_transformer = None
        self.net_dv3d = None
        self.net_disc = None

        self.sigma_t = None
        self.shift_factor = 0.0
        self.scaling_factor = 1.0
        self.weight_dtype = torch.float32

    def _infer_weight_dtype(self) -> torch.dtype:
        mp = str(getattr(self.args, "mixed_precision", "no")).lower()
        if mp in {"bf16", "bf16-mixed"}:
            return torch.bfloat16
        if mp in {"fp16", "16", "16-mixed"}:
            return torch.float16
        return torch.float32

    def setup(self, stage: Optional[str] = None):
        if self.fixed_vae is not None:
            return

        self.weight_dtype = self._infer_weight_dtype()

        # Prompt embeds (kept on device)
        if not self.args.fixed_prompt_path:
            text_encoding_pipeline = SanaPipeline.from_pretrained(
                self.args.model_path, transformer=None, vae=None, torch_dtype=self.weight_dtype
            )
            text_encoding_pipeline = text_encoding_pipeline.to(self.device)
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                    self.args.fixed_prompt
                )
            del text_encoding_pipeline
        else:
            prompts = torch.load(
                self.args.fixed_prompt_path,
                weights_only=True,
                map_location=self.device,
            )
            prompt_embeds = prompts["prompt_embeds"]
            prompt_attention_mask = prompts.get("prompt_attention_mask", None)

        self.register_buffer("_prompt_embeds", prompt_embeds, persistent=False)
        if prompt_attention_mask is None:
            self._prompt_attention_mask = None
        else:
            self.register_buffer(
                "_prompt_attention_mask", prompt_attention_mask, persistent=False
            )

        # mid-timestep
        sana_timesteps = get_sana_setting_timesteps(
            resolution=self.args.resolution)
        self.sigma_t = float(
            sana_timesteps[-(int(self.args.mid_timestep) + 1)])
        logger.info(
            f"Current {self.args.model} mid-timestep = {self.args.mid_timestep}")

        # fixed vae
        fixed_vae = AutoencoderDC.from_pretrained(
            self.args.model_path, subfolder="vae", torch_dtype=self.weight_dtype
        )
        fixed_vae.requires_grad_(False)
        fixed_vae.eval()
        self.fixed_vae = fixed_vae

        self.shift_factor = float(
            getattr(self.fixed_vae.config, "shift_factor", 0.0))
        self.scaling_factor = float(
            getattr(self.fixed_vae.config, "scaling_factor", 1.0))

        # lora vae (encoder only)
        lora_vae = copy.deepcopy(self.fixed_vae)
        lora_vae.requires_grad_(False)
        if hasattr(lora_vae, "decoder"):
            del lora_vae.decoder
        lora_vae.encoder = set_vae_encoder_lora(
            lora_vae.encoder, int(self.args.vae_lora_rank))
        lora_vae.train()
        self.lora_vae = lora_vae

        # transformer
        sana_transformer = SanaTransformer2DModel.from_pretrained(
            self.args.model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )
        sana_transformer.requires_grad_(False)
        sana_transformer = set_sana_transformer_lora(
            sana_transformer, int(self.args.transformer_lora_rank)
        )
        sana_transformer.train()
        self.sana_transformer = sana_transformer

        if bool(self.args.enable_xformers_memory_efficient_attention):
            if is_xformers_available():
                self.sana_transformer.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available, please install it by running `pip install xformers`"
                )

        if bool(self.args.gradient_checkpointing):
            self.sana_transformer.enable_gradient_checkpointing()

        self.net_dv3d = DINOv3ConvNeXtDISTS(
            dinov3_convnext_size=self.args.dinov3_convnext_size
        )
        self.net_disc = Dinov3ConvNeXtDiscriminator(
            dinov3_convnext_size=self.args.dinov3_convnext_size,
            resolution=self.args.resolution,
        )

        if bool(self.args.allow_tf32) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Memory profiling flags
        self._profile_memory = bool(getattr(self.args, "profile_memory", False))
        self._mem_profile_armed = False

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        if not self._profile_memory:
            return
        if torch.cuda.is_available() and self.trainer.is_global_zero and not self._mem_profile_armed:
            torch.cuda.reset_peak_memory_stats()
            self._mem_profile_armed = True

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self._profile_memory:
            return
        if torch.cuda.is_available() and self.trainer.is_global_zero and self._mem_profile_armed:
            alloc = torch.cuda.max_memory_allocated()
            reserv = torch.cuda.max_memory_reserved()
            logger.info(
                f"[VRAM] peak allocated={alloc/1024**3:.2f} GB, peak reserved={reserv/1024**3:.2f} GB"
            )
            self._mem_profile_armed = False

    def one_mid_timestep_pred(self, lq_latent: torch.Tensor) -> torch.Tensor:
        bsz, _, _, _ = lq_latent.shape
        guidance_vec = torch.full(
            (bsz,), 5.0, device=lq_latent.device, dtype=lq_latent.dtype
        )

        model_pred = self.sana_transformer(
            hidden_states=lq_latent,
            encoder_hidden_states=self._prompt_embeds,
            timestep=torch.tensor([self.sigma_t], device=lq_latent.device),
            encoder_attention_mask=getattr(
                self, "_prompt_attention_mask", None),
            guidance=guidance_vec,
            return_dict=False,
        )[0]

        lq_latent = lq_latent - self.sigma_t * model_pred
        lq_latent = (lq_latent / self.scaling_factor) + self.shift_factor
        pred_img = self.fixed_vae.decode(
            lq_latent.to(self.fixed_vae.dtype), return_dict=False
        )[0]
        return pred_img

    def configure_optimizers(self):
        use_8bit_adam = bool(self.args.use_8bit_adam)
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`."
                ) from e
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        sr_params = [p for p in self.parameters(
        ) if p.requires_grad and p is not None]

        # Explicit param groups to avoid accidentally including fixed_vae.
        sr_params = []
        sr_params.extend(
            [p for p in self.lora_vae.parameters() if p.requires_grad])
        sr_params.extend(
            [p for p in self.sana_transformer.parameters() if p.requires_grad]
        )
        disc_params = [p for p in self.net_disc.parameters()
                       if p.requires_grad]

        optimizer_sr = optimizer_class(
            sr_params,
            lr=float(self.args.learning_rate),
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            weight_decay=float(self.args.adam_weight_decay),
            eps=float(self.args.adam_epsilon),
        )
        optimizer_disc = optimizer_class(
            disc_params,
            lr=float(self.args.learning_rate),
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            weight_decay=float(self.args.adam_weight_decay),
            eps=float(self.args.adam_epsilon),
        )

        # Lightning calls schedulers automatically if returned in the dict format.
        sched_sr = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer_sr,
            num_warmup_steps=int(self.args.lr_warmup_steps) *
            max(1, self.trainer.num_devices),
            num_training_steps=int(self.args.max_train_steps) *
            max(1, self.trainer.num_devices),
            num_cycles=int(self.args.lr_num_cycles),
            power=float(self.args.lr_power),
        )
        sched_disc = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=int(self.args.lr_warmup_steps) *
            max(1, self.trainer.num_devices),
            num_training_steps=int(self.args.max_train_steps) *
            max(1, self.trainer.num_devices),
            num_cycles=int(self.args.lr_num_cycles),
            power=float(self.args.lr_power),
        )

        return (
            [optimizer_sr, optimizer_disc],
            [
                {"scheduler": sched_sr, "interval": "step", "name": "lr_sr"},
                {"scheduler": sched_disc, "interval": "step", "name": "lr_disc"},
            ],
        )

    @rank_zero_only
    def _save_weight(self, global_step: int):
        weight_path = os.path.join(
            self.args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        self.sana_transformer.save_pretrained(weight_path)
        self.lora_vae.encoder.save_pretrained(weight_path)
        logger.info(f"Saved weight to {weight_path}")

    @rank_zero_only
    def _save_debug_image(self, global_step: int, lq_img, pred_img, hq_img):
        img_path = os.path.join(self.args.output_dir, f"img-{global_step}.png")
        save_imgs = (torch.stack(
            [lq_img[0], pred_img[0], hq_img[0]], dim=0) + 1) / 2
        save_image(save_imgs.detach().cpu(), img_path)
        logger.info(f"img-{global_step}.png saved!")

    def training_step(self, batch, batch_idx):
        opt_sr, opt_disc = self.optimizers()
        sch_sr, sch_disc = self.lr_schedulers()

        lq_img, hq_img = batch
        lq_img = lq_img.to(self.device)
        hq_img = hq_img.to(self.device)

        # Generator / SR
        hq_latent = encode_images(hq_img, self.fixed_vae, self.weight_dtype)
        noise = torch.randn_like(hq_latent)
        pretrained_noisy_latent = (1 - self.sigma_t) * \
            hq_latent + self.sigma_t * noise

        lq_latent = encode_images(lq_img, self.lora_vae, self.weight_dtype)

        loss_LRR = (
            F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean")
            * float(self.args.lambda_LRR)
        )

        pred_img = self.one_mid_timestep_pred(lq_latent)

        loss_Dv3D = self.net_dv3d(pred_img, hq_img) * \
            float(self.args.lambda_Dv3D)
        loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * \
            float(self.args.lambda_L1)
        loss_G = self.net_disc(pred_img, for_G=True) * \
            float(self.args.lambda_GAN)

        total_G_loss = loss_LRR + loss_Dv3D + loss_L1 + loss_G

        # Manual grad accumulation (Lightning's accumulate_grad_batches does not apply in manual optimization).
        self.manual_backward(total_G_loss / float(self.grad_accum_steps))

        is_accum_end = (
            ((batch_idx + 1) % self.grad_accum_steps == 0)
            or (batch_idx + 1 == self.trainer.num_training_batches)
        )

        total_D_loss = None
        loss_D_fake = None
        loss_D_real = None

        if is_accum_end:
            if float(self.args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.lora_vae.parameters() if p.requires_grad]
                    + [p for p in self.sana_transformer.parameters()
                       if p.requires_grad],
                    float(self.args.max_grad_norm),
                )
            opt_sr.step()
            sch_sr.step()
            opt_sr.zero_grad(set_to_none=True)

            # Discriminator update only on optimizer-step boundaries (matches accelerate.sync_gradients behavior).
            fake_img = pred_img.detach()
            loss_D_fake = self.net_disc(
                fake_img, for_real=False) * float(self.args.lambda_GAN)
            loss_D_real = self.net_disc(hq_img.to(fake_img.dtype), for_real=True) * float(
                self.args.lambda_GAN
            )
            total_D_loss = loss_D_real + loss_D_fake

            self.manual_backward(total_D_loss)
            if float(self.args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.net_disc.parameters() if p.requires_grad],
                    float(self.args.max_grad_norm),
                )
            opt_disc.step()
            sch_disc.step()
            opt_disc.zero_grad(set_to_none=True)

            self.opt_step += 1

            if self.trainer.is_global_zero:
                if self.opt_step > 0 and self.opt_step % int(self.args.save_img_steps) == 0:
                    self._save_debug_image(
                        self.opt_step, lq_img, pred_img, hq_img)
                if (
                    self.opt_step > 0
                    and self.opt_step % int(self.args.checkpointing_steps) == 0
                ):
                    self._save_weight(self.opt_step)

        self.log_dict(
            {
                "loss_LRR": loss_LRR.detach(),
                "loss_D_fake": loss_D_fake.detach() if loss_D_fake is not None else torch.tensor(0.0, device=self.device),
                "loss_D_real": loss_D_real.detach() if loss_D_real is not None else torch.tensor(0.0, device=self.device),
                "loss_Dv3D": loss_Dv3D.detach(),
                "loss_L1": loss_L1.detach(),
                "loss_G": loss_G.detach(),
                "lr": torch.tensor(opt_sr.param_groups[0]["lr"], device=self.device),
                "opt_step": torch.tensor(float(self.opt_step), device=self.device),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return total_G_loss.detach()

    def on_train_end(self):
        if self.trainer.is_global_zero:
            self._save_weight(int(self.opt_step))


def parse_args():
    parser = argparse.ArgumentParser(description="OMGSR Sana-Rain (Lightning)")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/omgsr_sana_1024_rain.yml",
        help="path to config",
    )
    parser.add_argument(
        "--dry_run_steps",
        type=int,
        default=None,
        help="If set, run only this many optimizer steps (approx).",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Print CUDA peak memory after a full forward+backward.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging (useful for profiling).",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        choices=["cpu", "gpu", "auto"],
        help="Override Trainer accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override Trainer devices.",
    )
    parser.add_argument(
        "--synthetic_batch",
        action="store_true",
        help="Use a synthetic random batch (no dataset I/O).",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    args = OmegaConf.load(cli.config)

    # Merge CLI overrides into config (keep config as the single source of truth).
    if cli.dry_run_steps is not None:
        args.max_train_steps = int(cli.dry_run_steps)
    if bool(cli.profile_memory):
        args.profile_memory = True
    if bool(cli.no_wandb):
        args.no_wandb = True
    if cli.accelerator is not None:
        args.trainer_accelerator = str(cli.accelerator)
    if cli.devices is not None:
        args.trainer_devices = int(cli.devices)
    if bool(cli.synthetic_batch):
        args.synthetic_batch = True

    os.makedirs(args.output_dir, exist_ok=True)
    OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.seed is not None:
        pl.seed_everything(int(args.seed), workers=True)

    if bool(getattr(args, "synthetic_batch", False)):
        synthetic_ds = _SyntheticPairsDataset(int(args.resolution), length=1)

        class _SyntheticDM(pl.LightningDataModule):
            def train_dataloader(self_nonlocal):
                return DataLoader(
                    synthetic_ds,
                    batch_size=int(args.train_batch_size),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )

        datamodule = _SyntheticDM()
    else:
        datamodule = SanaRainDataModule(
            dataset_txt_or_dir_paths=args.dataset_txt_or_dir_paths,
            resolution=args.resolution,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    model = OMGSR_SanaRain_Lightning(args)

    if bool(getattr(args, "no_wandb", False)):
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(
            project=str(getattr(args, "wandb_project", "omgsr")),
            name=str(getattr(args, "wandb_name", Path(args.output_dir).name)),
            save_dir=str(Path(args.output_dir)),
        )
        # Record the full resolved config for reproducibility.
        wandb_logger.experiment.config.update(
            OmegaConf.to_container(args, resolve=True), allow_val_change=True
        )

    precision = "32-true"
    mp = str(getattr(args, "mixed_precision", "no")).lower()
    if mp in {"bf16", "bf16-mixed"}:
        precision = "bf16-mixed"
    elif mp in {"fp16", "16", "16-mixed"}:
        precision = "16-mixed"

    # Trainer device selection
    accelerator = str(getattr(args, "trainer_accelerator", "auto")).lower()
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = int(getattr(args, "trainer_devices", 1))

    trainer = pl.Trainer(
        default_root_dir=str(args.output_dir),
        logger=wandb_logger,
        max_steps=int(args.max_train_steps),
        # Manual optimization: handle gradient accumulation inside training_step.
        accumulate_grad_batches=1,
        precision=precision,
        log_every_n_steps=1,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        plugins=[LightningEnvironment()],
        limit_train_batches=1 if int(args.max_train_steps) <= 1 else 1.0,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
