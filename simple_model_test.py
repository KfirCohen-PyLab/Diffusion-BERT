import torch
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM

def test_basic_model():
    """Test if the model can do basic inference without diffusion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load model
    cfg = BertConfig.from_pretrained('bert-base-uncased')
    cfg.overall_timestep = 512
    model = BertForMaskedLM(cfg)
    
    # Load checkpoint
    model_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts\epoch_4.pt'
    try:
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Test sentences
    test_sentences = [
        "The [MASK] brown fox jumps over the lazy dog.",
        "Artificial [MASK] is transforming the world."
    ]
    
    print("\n" + "="*60)
    print("BASIC MODEL TEST (No Diffusion)")
    print("="*60)
    
    with torch.no_grad():
        for i, sentence in enumerate(test_sentences):
            print(f"\nTest {i+1}: {sentence}")
            
            # Tokenize
            inputs = tokenizer(sentence, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
            
            try:
                # Basic model inference
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = outputs.logits
                
                print(f"✅ Model inference successful")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                
                # Find mask token position
                mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
                
                if len(mask_positions[0]) > 0:
                    batch_idx, seq_idx = mask_positions[0][0], mask_positions[1][0]
                    mask_logits = logits[batch_idx, seq_idx]
                    
                    # Get top predictions
                    top_predictions = torch.topk(mask_logits, 5)
                    predicted_tokens = [tokenizer.decode([token_id]) for token_id in top_predictions.indices]
                    predicted_probs = torch.softmax(mask_logits, dim=-1)[top_predictions.indices]
                    
                    print(f"   Top 5 predictions for [MASK]:")
                    for j, (token, prob) in enumerate(zip(predicted_tokens, predicted_probs)):
                        print(f"     {j+1}. '{token}' (prob: {prob:.3f})")
                    
                    # Create completed sentence
                    completed_ids = input_ids.clone()
                    completed_ids[batch_idx, seq_idx] = top_predictions.indices[0]
                    completed_sentence = tokenizer.decode(completed_ids[0], skip_special_tokens=True)
                    print(f"   Completed: {completed_sentence}")
                
            except Exception as e:
                print(f"❌ Model inference failed: {e}")
                import traceback
                traceback.print_exc()

def test_tokenizer_vocab():
    """Test tokenizer vocabulary bounds"""
    print("\n" + "="*60)
    print("TOKENIZER VOCABULARY TEST")
    print("="*60)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"CLS token ID: {tokenizer.cls_token_id}")
    print(f"SEP token ID: {tokenizer.sep_token_id}")
    print(f"MASK token ID: {tokenizer.mask_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    
    # Test some edge case token IDs
    edge_cases = [0, 1, vocab_size-2, vocab_size-1]
    print(f"\nEdge case tokens:")
    for token_id in edge_cases:
        try:
            token_str = tokenizer.decode([token_id])
            print(f"  Token ID {token_id}: '{token_str}'")
        except Exception as e:
            print(f"  Token ID {token_id}: ERROR - {e}")
    
    # Test if we can safely generate random token IDs
    print(f"\nRandom token test:")
    try:
        random_ids = torch.randint(0, vocab_size, (5,))
        random_tokens = [tokenizer.decode([token_id]) for token_id in random_ids]
        print(f"  Random token IDs {random_ids.tolist()}: {random_tokens}")
    except Exception as e:
        print(f"  Random token test failed: {e}")

if __name__ == "__main__":
    test_tokenizer_vocab()
    test_basic_model() 