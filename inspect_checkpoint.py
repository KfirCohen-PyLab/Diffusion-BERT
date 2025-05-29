import torch
import os

def print_checkpoint_structure(checkpoint_path):
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=== Checkpoint Structure ===")
    
    # Print time_embed structure
    print("\nTime Embedding Layers:")
    time_embed_keys = [k for k in checkpoint.keys() if k.startswith('time_embed')]
    for k in sorted(time_embed_keys):
        print(f"{k}: {checkpoint[k].shape}")
    
    # Print denoise_net structure
    print("\nDenoise Network Layers:")
    denoise_keys = [k for k in checkpoint.keys() if k.startswith('denoise_net')]
    for k in sorted(denoise_keys):
        print(f"{k}: {checkpoint[k].shape}")
    
    # Print refinement structure
    print("\nRefinement Layers:")
    refinement_keys = [k for k in checkpoint.keys() if k.startswith('refinement')]
    for k in sorted(refinement_keys):
        print(f"{k}: {checkpoint[k].shape}")

if __name__ == "__main__":
    # Try both possible checkpoint locations
    checkpoint_paths = [
        "checkpoints/diffusion_bert_lm1b_final.pt",
        "G:/ML_Project_Sem6/diffusion_bert_lm1b_checkpoint_e0_s26999.pt"
    ]
    
    for path in checkpoint_paths:
        print(f"\nTrying path: {path}")
        print_checkpoint_structure(path) 