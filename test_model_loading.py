import torch
from transformers import BertConfig
from models.modeling_diffusion_bert_checkpoint import DiffusionBertForMaskedLM

def test_model_loading(checkpoint_path):
    print(f"Testing model loading from: {checkpoint_path}")
    
    # Initialize model
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = DiffusionBertForMaskedLM(config)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print model structure before loading
    print("\nModel structure before loading:")
    for name, param in model.named_parameters():
        if name.startswith(('time_embed', 'denoise_net')):
            print(f"{name}: {param.shape}")
    
    # Print checkpoint structure
    print("\nCheckpoint structure:")
    for key in checkpoint.keys():
        if key.startswith(('time_embed', 'denoise_net')):
            print(f"{key}: {checkpoint[key].shape}")
    
    # Try loading the checkpoint
    print("\nAttempting to load checkpoint...")
    try:
        model.load_state_dict(checkpoint, strict=False)
        print("Successfully loaded checkpoint!")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")

if __name__ == "__main__":
    checkpoint_path = "G:/ML_Project_Sem6/diffusion_bert_lm1b_checkpoint_e0_s26999.pt"
    test_model_loading(checkpoint_path) 