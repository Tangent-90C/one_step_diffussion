
import torch
from diffusers import AutoencoderDC

def inspect_vae_output():
    try:
        # We don't have the model path easily available without downloading or using the one in the script.
        # But the user provided the path in the logs: /mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers
        model_path = "/mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers"
        
        print(f"Loading AutoencoderDC from {model_path}...")
        vae = AutoencoderDC.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        
        dummy_input = torch.randn(1, 3, 1024, 1024)
        print("Running encode...")
        encoded = vae.encode(dummy_input)
        
        print(f"Type of encoded: {type(encoded)}")
        print(f"Attributes of encoded: {dir(encoded)}")
        
        if hasattr(encoded, 'latent_dist'):
            print("Has latent_dist")
        else:
            print("No latent_dist")
            
        if hasattr(encoded, 'latents'):
            print("Has latents")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_vae_output()
