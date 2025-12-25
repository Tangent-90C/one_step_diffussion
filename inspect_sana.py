
import inspect
from diffusers import SanaTransformer2DModel
import torch

try:
    path = "/mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers"
    print(f"Loading config from {path}...")
    config = SanaTransformer2DModel.load_config(path, subfolder="transformer")
    model = SanaTransformer2DModel.from_config(config)
    
    print(f"Type of time_embed: {type(model.time_embed)}")
    print(f"Signature of forward: {inspect.signature(model.time_embed.forward)}")
except Exception as e:
    print(f"Error: {e}")
