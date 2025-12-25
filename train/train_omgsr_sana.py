#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys
import argparse
import logging
import math
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


from pathlib import Path
from typing import Callable
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from diffusers import (
    AutoencoderDC, SanaPipeline, SanaTransformer2DModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
import warnings
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.my_dataset import PairedDataset
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
import copy


class SRBundle(torch.nn.Module):
    def __init__(self, lora_vae: torch.nn.Module, sana_transformer: torch.nn.Module):
        super().__init__()
        self.lora_vae = lora_vae
        self.sana_transformer = sana_transformer

    def forward(self, *args, **kwargs):
        return self.sana_transformer(*args, **kwargs)

# Monkeypatch to fix TypeError: SanaCombinedTimestepGuidanceEmbeddings.forward() got an unexpected keyword argument 'batch_size'
try:
    from diffusers.models.transformers.sana_transformer import SanaCombinedTimestepGuidanceEmbeddings
    original_forward = SanaCombinedTimestepGuidanceEmbeddings.forward
    def new_forward(self, timestep, guidance=None, hidden_dtype=None, **kwargs):
        return original_forward(self, timestep, guidance, hidden_dtype)
    SanaCombinedTimestepGuidanceEmbeddings.forward = new_forward
except ImportError:
    pass

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    encoded = vae.encode(pixels.to(vae.dtype))
    if hasattr(encoded, "latent_dist"):
        pixel_latents = encoded.latent_dist.sample()
    elif hasattr(encoded, "latent"):
        pixel_latents = encoded.latent
    else:
        raise AttributeError("EncoderOutput has neither latent_dist nor latent")
    
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
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def get_sana_setting_timesteps(n=999, resolution=1024):
    # Sana latent size is resolution / 32
    latent_size = resolution // 32
    return get_schedule(
        n,
        latent_size * latent_size,
        shift=True,
    )

def set_vae_encoder_lora(vae_encoder, rank):
    # Adjust target modules for AutoencoderDC if needed
    # For now assuming similar structure or generic
    # target_modules = [
    #     "conv1",
    #     "conv2",
    #     "conv_in",
    #     "conv_shortcut",
    #     "conv",
    #     "conv_out",
    #     "to_k",
    #     "to_q",
    #     "to_v",
    #     "to_out.0",
    #     # "GLUMBConv"
    # ]
    
    target_modules = r"(^conv_in$|^conv_out$|.*\.conv1$|.*\.conv2$|.*\.conv_shortcut$|.*\.conv$|.*\.to_k$|.*\.to_q$|.*\.to_v$|.*\.to_out$|.*\.conv_inverted$|.*\.conv_point$)"

    # Filter target modules that actually exist in the model
    # This is safer than hardcoding
    # But Peft might complain if modules don't exist.
    # I'll keep it as is for now, assuming standard VAE structure or user can adjust.
    # AutoencoderDC might have different names.
    # rank = 1536
    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    vae_encoder = PeftModel(vae_encoder, vae_encoder_lora_config, adapter_name="vae_encoder_adapter")
    vae_encoder.print_trainable_parameters()
    return vae_encoder





def set_sana_transformer_lora(sana_transformer, rank):
    # target_modules = [
    #     "attn.to_k",
    #     "attn.to_q",
    #     "attn.to_v",
    #     "attn.to_out.0",
    #     "ff.net.0.proj",
    #     "ff.net.2",
    # ]
 
    target_modules = ["to_k", "to_q", "to_v"]
    
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    # print(get_peft_model_state_dict(sana_transformer))
    sana_transformer = PeftModel(sana_transformer, transformer_lora_config, adapter_name="sana_adapter")
    sana_transformer.print_trainable_parameters()
    return sana_transformer

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/omgsr_sana_1024.yml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # prompt embeds
    if not args.fixed_prompt_path:
        text_encoding_pipeline = SanaPipeline.from_pretrained(
            args.model_path, transformer=None, vae=None, torch_dtype=weight_dtype
        )
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            # Sana encode_prompt returns prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                args.fixed_prompt
            )
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")  # Move to CPU first
        del text_encoding_pipeline
        free_memory()
    else:
        prompts = torch.load(args.fixed_prompt_path, weights_only=True, map_location=accelerator.device)
        prompt_embeds = prompts["prompt_embeds"]
        prompt_attention_mask = prompts.get("prompt_attention_mask", None)

    # mid-timestep
    sana_timesteps = get_sana_setting_timesteps(resolution=args.resolution)
    sigma_t = sana_timesteps[-(args.mid_timestep + 1)]
    logger.info(f"Current {args.model} mid-timestep = {args.mid_timestep}")

    # fixed vae
    # Sana uses AutoencoderDC
    fixed_vae = AutoencoderDC.from_pretrained(args.model_path, subfolder="vae", torch_dtype=weight_dtype)
    fixed_vae.requires_grad_(False)
    fixed_vae.eval()

    # lora_vae
    lora_vae = copy.deepcopy(fixed_vae)
    lora_vae.requires_grad_(False)
    # AutoencoderDC might not have 'decoder' attribute in the same way or we might want to keep it?
    # We only need encoder for LRR loss (encoding lq_img).
    # If we delete decoder, we save memory.
    if hasattr(lora_vae, "decoder"):
        del lora_vae.decoder
    free_memory()
    lora_vae.encoder = set_vae_encoder_lora(lora_vae.encoder, args.vae_lora_rank)
    lora_vae.train()

    # sana_transformer
    sana_transformer = SanaTransformer2DModel.from_pretrained(args.model_path, subfolder="transformer", torch_dtype=weight_dtype)    
    sana_transformer.requires_grad_(False)
    sana_transformer = set_sana_transformer_lora(sana_transformer, args.transformer_lora_rank)
    sana_transformer.train()

    # Bundle SR trainable parts into a single module for DeepSpeed.
    sr_bundle = SRBundle(lora_vae=lora_vae, sana_transformer=sana_transformer)

    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            sana_transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )
    
    # DINOv3-ConvNeXt DISTS Loss
    from dinov3_gan.dinov3_convnext_dists import DINOv3ConvNeXtDISTS
    net_dv3d = DINOv3ConvNeXtDISTS(dinov3_convnext_size=args.dinov3_convnext_size)

    # DINOv3-ConvNeXt Discrminator
    from dinov3_gan.dinov3_convnext_disc import Dinov3ConvNeXtDiscriminator
    net_disc = Dinov3ConvNeXtDiscriminator(dinov3_convnext_size=args.dinov3_convnext_size, resolution=args.resolution)

    fixed_vae.to(device=accelerator.device)  
    sr_bundle.to(device=accelerator.device)
    net_dv3d.to(device=accelerator.device)
    net_disc.to(device=accelerator.device)

    if args.gradient_checkpointing:
        # lora_vae.enable_gradient_checkpointing()
        sana_transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        f"Total vae_encoder training parameters: {sum([p.numel() for p in sr_bundle.lora_vae.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total sana_transformer training parameters: {sum([p.numel() for p in sr_bundle.sana_transformer.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total disc training parameters: {sum([p.numel() for p in net_disc.parameters() if p.requires_grad]) / 1000000} M"
    )
    sr_opt = list(filter(lambda p: p.requires_grad, sr_bundle.parameters()))
    disc_opt = list(filter(lambda p: p.requires_grad, net_disc.parameters()))

    # bitsandbytes 8-bit optimizers frequently conflict with DeepSpeed ZeRO optimizers.
    # Prefer regular AdamW when using DeepSpeed.
    use_8bit_adam = bool(args.use_8bit_adam)
    # if accelerator.distributed_type == DistributedType.DEEPSPEED and use_8bit_adam:
    #     # 跑不通就取消注释这个
    #     logger.warning("DeepSpeed is enabled; disabling bitsandbytes 8-bit Adam for compatibility.")
    #     use_8bit_adam = False

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_sr = optimizer_class(
        sr_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_disc = optimizer_class(
        disc_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PairedDataset(args.dataset_txt_or_dir_paths, args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed supports a single engine per Accelerator instance.
        # Wrap only SR bundle + its optimizer/dataloader/scheduler.
        sr_bundle, optimizer_sr, train_dataloader, lr_scheduler_sr = accelerator.prepare(
            sr_bundle, optimizer_sr, train_dataloader, lr_scheduler_sr
        )
        # Keep discriminator outside DeepSpeed to avoid a second engine.
        # (This code path is intended for num_processes=1. For multi-GPU GAN training,
        # prefer DDP over DeepSpeed or refactor to a single-engine design.)
    else:
        sr_bundle, optimizer_sr, train_dataloader, lr_scheduler_sr = accelerator.prepare(
            sr_bundle, optimizer_sr, train_dataloader, lr_scheduler_sr
        )
        net_disc, optimizer_disc, lr_scheduler_disc = accelerator.prepare(
            net_disc, optimizer_disc, lr_scheduler_disc
        )

    sr_module = unwrap_model(sr_bundle)
    sr_opt = [p for p in sr_module.parameters() if p.requires_grad]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info(f"***** Start training {args.model} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            # global_step = int(path.split("-")[1])

            # initial_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch
            # TODO
            pass
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    shift_factor = getattr(fixed_vae.config, "shift_factor", 0.0)
    scaling_factor = getattr(fixed_vae.config, "scaling_factor", 1.0)

    def one_mid_timestep_pred(lq_latent):
        bsz, c, h, w = lq_latent.shape
        
        guidance_vec = torch.full(
            (bsz,), 5.0, device=lq_latent.device, dtype=lq_latent.dtype
        )

        # Sana forward arguments:
        # hidden_states, encoder_hidden_states, timestep, encoder_attention_mask
        
        model_pred = sr_module.sana_transformer(
            hidden_states=lq_latent,
            encoder_hidden_states=prompt_embeds,
            timestep=torch.tensor([sigma_t], device=lq_latent.device),
            encoder_attention_mask=prompt_attention_mask,
            guidance=guidance_vec,
            return_dict=False,
        )[0]

        lq_latent = lq_latent - sigma_t * model_pred  
        
        lq_latent = (lq_latent / scaling_factor) + shift_factor
        pred_img = fixed_vae.decode(lq_latent.to(fixed_vae.dtype), return_dict=False)[0]
        return pred_img
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sr_bundle):
                # Prepare data
                lq_img, hq_img = batch
                lq_img = lq_img.to(accelerator.device)
                hq_img = hq_img.to(accelerator.device)

                hq_latent = encode_images(hq_img, fixed_vae, weight_dtype)
                noise = torch.randn_like(hq_latent)
                pretrained_noisy_latent = (1 - sigma_t) * hq_latent + sigma_t * noise  

                lq_latent = encode_images(lq_img, sr_module.lora_vae, weight_dtype)

                # LRR Loss: Latent Representation Refinement Loss
                loss_LRR = F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean") * args.lambda_LRR
                
                # Onestep prediction at mid-timestep
                pred_img = one_mid_timestep_pred(lq_latent)

                # DINOv3-ConvNext DISTS Loss 
                loss_Dv3D = net_dv3d(pred_img, hq_img) * args.lambda_Dv3D

                # L1 Loss
                loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * args.lambda_L1

                # Generator Loss (FLUX/SANA)
                loss_G = net_disc(pred_img, for_G=True) * args.lambda_GAN
                
                total_G_loss = loss_LRR + loss_Dv3D + loss_L1 + loss_G

                accelerator.backward(total_G_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_opt, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad()
                
                fake_img = pred_img.detach()
                # Fake images
                loss_D_fake = net_disc(fake_img, for_real=False) * args.lambda_GAN 
                # Real images
                hq_img = hq_img.to(fake_img.dtype)
                loss_D_real = net_disc(hq_img, for_real=True) * args.lambda_GAN 
          
                total_D_loss = loss_D_real + loss_D_fake 

                if accelerator.distributed_type == DistributedType.DEEPSPEED:
                    # Do NOT use accelerator.backward() here: it is tied to the SR DeepSpeed engine.
                    total_D_loss.backward()
                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(disc_opt, args.max_grad_norm)
                        optimizer_disc.step()
                        lr_scheduler_disc.step()
                        optimizer_disc.zero_grad()
                else:
                    accelerator.backward(total_D_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(disc_opt, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (
                    accelerator.is_main_process
                    and global_step % args.save_img_steps == 0
                ):
                    img_path = os.path.join(args.output_dir, f"img-{global_step}.png")
                    save_imgs = (torch.stack([lq_img[0], pred_img[0], hq_img[0]], dim=0) + 1) / 2
                    save_image(save_imgs.detach(), img_path)
                    logger.info(f"img-{global_step}.png saved!")

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    if global_step % args.checkpointing_steps == 0:
                        weight_path = os.path.join(
                            args.output_dir, f"weight-{global_step}"
                        )
                        os.makedirs(weight_path, exist_ok=True)
                        sr_module.sana_transformer.save_pretrained(weight_path)
                        sr_module.lora_vae.encoder.save_pretrained(weight_path)
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss_LRR": loss_LRR.detach().item(),
                "loss_D_fake": loss_D_fake.detach().item(),
                "loss_D_real": loss_D_real.detach().item(),
                "loss_Dv3D": loss_Dv3D.detach().item(),
                "loss_L1": loss_L1.detach().item(),
                "lr": lr_scheduler_sr.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        sr_module.sana_transformer.save_pretrained(weight_path)
        sr_module.lora_vae.encoder.save_pretrained(weight_path)
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
