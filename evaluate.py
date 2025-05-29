import torch
import logging
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoConfig, AutoModel, AutoTokenizer
from models.modeling_diffusion_bert import DiffusionBertForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(config):
    try:
        # Register custom model
        AutoConfig.register("diffusion-bert", DiffusionBertForMaskedLM)
        AutoModel.register(DiffusionBertForMaskedLM, "diffusion-bert")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Load config and update with custom parameters
        model_config = AutoConfig.from_pretrained(config['model_name'])
        model_config.max_position_embeddings = config.get('max_position_embeddings', 512)
        
        # Initialize model
        model = DiffusionBertForMaskedLM(model_config)
        
        # Load checkpoint
        checkpoint = torch.load(config['model_checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def load_word_frequencies(config):
    try:
        word_freq_path = Path(config['word_freq_path'])
        if word_freq_path.suffix == '.pt':
            word_freq = torch.load(word_freq_path)
        else:
            with open(word_freq_path) as f:
                word_freq = json.load(f)
            word_freq = torch.tensor(word_freq)
        
        return word_freq
        
    except Exception as e:
        logger.error(f"Error loading word frequencies: {str(e)}")
        raise

def evaluate_model(model, tokenizer, word_freq, config):
    try:
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'config': config,
            'metrics': {},
            'samples': []
        }
        
        # Evaluation logic here
        # TODO: Implement evaluation metrics and sample generation
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def save_results(results, config):
    try:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"eval_results_{results['timestamp']}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    try:
        # Configuration
        config = {
            'model_name': 'bert-base-uncased',
            'model_checkpoint_path': 'path/to/your/checkpoint.th',
            'word_freq_path': 'word_freq/bert-base-uncased_lm1b.pt',
            'output_dir': 'evaluation_results',
            'max_position_embeddings': 512
        }
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Load word frequencies
        word_freq = load_word_frequencies(config)
        
        # Evaluate model
        results = evaluate_model(model, tokenizer, word_freq, config)
        
        # Save results
        save_results(results, config)
        
        # Print summary
        print("\nEvaluation Results:")
        if 'avg_perplexity' in results.get('metrics', {}):
            print(f"Average Perplexity: {results['metrics']['avg_perplexity']:.2f} ± {results['metrics']['std_perplexity']:.2f}")
        if 'avg_word_freq_score' in results.get('metrics', {}):
            print(f"Average Word Frequency Score: {results['metrics']['avg_word_freq_score']:.4f} ± {results['metrics']['std_word_freq_score']:.4f}")
        
        if results.get('samples'):
            print("\nSample Generations:")
            for i, sample in enumerate(results['samples'][:3]):
                print(f"\nSample {i+1}:")
                print(f"Input: {sample['input']}")
                print(f"Generated: {sample['generated']}")
                
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 