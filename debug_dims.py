import torch
from diffusers import SanaTransformer2DModel, AutoencoderDC

model_path = '/mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers'

print("Checking SanaTransformer2DModel...")
try:
    transformer = SanaTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16)
    found = False
    for name, module in transformer.named_modules():
        dims = []
        if hasattr(module, 'weight') and module.weight is not None:
            dims.extend(module.weight.shape)
        if hasattr(module, 'bias') and module.bias is not None:
            dims.extend(module.bias.shape)
        if hasattr(module, 'in_features'): dims.append(module.in_features)
        if hasattr(module, 'out_features'): dims.append(module.out_features)
        if hasattr(module, 'in_channels'): dims.append(module.in_channels)
        if hasattr(module, 'out_channels'): dims.append(module.out_channels)
        
        if 1536 in dims:
            print(f"Found 1536 in {name} ({type(module).__name__}): {dims}")
            found = True
    if not found:
        print("No 1536 dimension found in Transformer.")
except Exception as e:
    print(f"Error loading transformer: {e}")

print("\nChecking AutoencoderDC...")
try:
    vae = AutoencoderDC.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16)
    found = False
    for name, module in vae.named_modules():
        dims = []
        if hasattr(module, 'weight') and module.weight is not None:
            dims.extend(module.weight.shape)
        if hasattr(module, 'bias') and module.bias is not None:
            dims.extend(module.bias.shape)
        if hasattr(module, 'in_features'): dims.append(module.in_features)
        if hasattr(module, 'out_features'): dims.append(module.out_features)
        if hasattr(module, 'in_channels'): dims.append(module.in_channels)
        if hasattr(module, 'out_channels'): dims.append(module.out_channels)
        
        if 1536 in dims:
            print(f"Found 1536 in {name} ({type(module).__name__}): {dims}")
            found = True
    if not found:
        print("No 1536 dimension found in VAE.")
except Exception as e:
    print(f"Error loading VAE: {e}")
