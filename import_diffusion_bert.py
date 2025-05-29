import sys
import os
from pathlib import Path
import subprocess
import shutil
import torch

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        # First ensure we have build tools
        subprocess.run(
            ["pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"],
            check=True,
            capture_output=True,
            text=True
        )
        
        if sys.platform.startswith('win'):
            print("Windows detected, installing pre-built packages...")
            # On Windows, try to install pre-built wheels
            subprocess.run(
                ["pip", "install", 
                 "--find-links", "https://download.pytorch.org/whl/torch_stable.html",
                 "torch==1.13.1+cpu",
                 "tokenizers==0.12.1",
                 "transformers==4.18.0",
                 "--quiet"],
                check=True,
                capture_output=True,
                text=True
            )
        else:
            # For non-Windows platforms
            subprocess.run(
                ["pip", "install", "tokenizers==0.12.1", "transformers==4.18.0", "torch", "--quiet"],
                check=True,
                capture_output=True,
                text=True
            )
        
        print("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e.stderr}")
        # Try alternative installation method
        try:
            print("Trying alternative installation method...")
            # Install packages one by one
            packages = [
                "tokenizers==0.12.1",
                "transformers==4.18.0",
                "torch"
            ]
            for package in packages:
                print(f"Installing {package}...")
                subprocess.run(
                    ["pip", "install", package, "--quiet"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            print("Alternative installation successful")
        except subprocess.CalledProcessError as e2:
            print(f"Alternative installation failed: {e2.stderr}")
            raise ImportError("Failed to install required packages")

def create_compatibility_layer():
    """Create a compatibility layer for transformers utils"""
    compat_code = """
# Compatibility layer for older transformers versions
def is_torch_tpu_available():
    return False

def is_tf_available():
    return False

def is_torch_available():
    return True

def is_tokenizers_available():
    return True
"""
    return compat_code

def import_diffusion_bert():
    """
    Import DiffusionBertForMaskedLM from various possible locations.
    Returns the DiffusionBertForMaskedLM class if found.
    """
    # Install requirements first
    install_requirements()

    # Define possible paths
    current_dir = Path.cwd()
    local_models_path = (current_dir / "models").resolve()
    drive_models_path = Path("/content/drive/MyDrive/DiffusionBERT/models")
    repo_name = "Diffusion-BERT"
    repo_url = "https://github.com/Hzfinfdu/Diffusion-BERT.git"
    
    def try_import(path):
        """Helper function to try importing from a path"""
        if path.exists():
            print(f"Trying to import from {path}")
            # Add both the models directory and its parent to sys.path
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            parent_dir = str(path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            try:
                if (path / "modeling_bert.py").exists():
                    print(f"Found modeling_bert.py in {path}")
                    # First ensure transformers is imported
                    import transformers
                    print(f"Successfully imported transformers {transformers.__version__}")
                    
                    # Create utils.py with compatibility layer if needed
                    utils_path = path / "utils.py"
                    if not utils_path.exists():
                        print("Creating compatibility utils.py")
                        with open(utils_path, "w") as f:
                            f.write(create_compatibility_layer())
                    
                    # Now try to import our model
                    from modeling_bert import DiffusionBertForMaskedLM
                    print(f"Successfully imported DiffusionBertForMaskedLM from {path}")
                    return DiffusionBertForMaskedLM
            except ImportError as e:
                print(f"Import error from {path}: {str(e)}")
                if str(path) in sys.path:
                    sys.path.remove(str(path))
                if parent_dir in sys.path:
                    sys.path.remove(parent_dir)
            except Exception as e:
                print(f"Error importing from {path}: {str(e)}")
        else:
            print(f"Path does not exist: {path}")
        return None

    # Try local models directory first
    print(f"Checking local models directory: {local_models_path}")
    if not local_models_path.exists():
        print("Creating local models directory...")
        os.makedirs(local_models_path, exist_ok=True)
    
    result = try_import(local_models_path)
    if result:
        return result

    # Try Google Drive path if running in Colab
    try:
        from google.colab import drive
        print("Running in Colab, checking Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        result = try_import(drive_models_path)
        if result:
            return result
    except (ImportError, ModuleNotFoundError):
        print("Not running in Google Colab, skipping Drive import")

    # Clone repo if needed
    repo_path = Path(repo_name).resolve()
    if repo_path.exists():
        print(f"Removing existing {repo_name} directory...")
        shutil.rmtree(repo_path)
    
    print(f"Cloning {repo_name} repository...")
    try:
        subprocess.run(
            ["git", "clone", repo_url],
            check=True,
            capture_output=True,
            text=True
        )
        print("Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")
        raise ImportError("Failed to clone repository")

    # Look for the model file in the cloned repo
    model_file = None
    repo_models_path = None

    # Try multiple possible locations for the model file
    possible_locations = [
        repo_path / "models",
        repo_path / "Diffusion-BERT" / "models",
        repo_path / "src" / "models",
        repo_path,
        repo_path / "Diffusion-BERT-main" / "models"
    ]

    for loc in possible_locations:
        if (loc / "modeling_bert.py").exists():
            model_file = loc / "modeling_bert.py"
            repo_models_path = loc
            print(f"Found model file at: {model_file}")
            break

    if not model_file:
        # Try recursive search as last resort
        print("Searching recursively for model file...")
        model_files = list(repo_path.rglob("modeling_bert.py"))
        if model_files:
            model_file = model_files[0]
            repo_models_path = model_file.parent
            print(f"Found model file at: {model_file}")
        else:
            # List all Python files found to help debug
            print("\nAll Python files found in repository:")
            for py_file in repo_path.rglob("*.py"):
                print(f"- {py_file}")
            raise ImportError("modeling_bert.py not found in cloned repository")

    # Copy all Python files from the models directory to ensure dependencies are available
    print(f"Copying model files to {local_models_path}")
    os.makedirs(local_models_path, exist_ok=True)
    
    # Copy all .py files from the source directory
    for py_file in repo_models_path.glob("*.py"):
        print(f"Copying {py_file.name}")
        shutil.copy2(py_file, local_models_path / py_file.name)

    # Create an empty __init__.py if it doesn't exist
    init_file = local_models_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    # Create utils.py with compatibility layer
    utils_file = local_models_path / "utils.py"
    print("Creating utils.py with compatibility layer")
    with open(utils_file, "w") as f:
        f.write(create_compatibility_layer())

    # Try importing from the original location first
    result = try_import(repo_models_path)
    if result:
        return result

    # Try importing from local models directory
    result = try_import(local_models_path)
    if result:
        return result

    # If we get here, we've failed to import
    print("\nDebug information:")
    print(f"sys.path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Model file exists: {(local_models_path / 'modeling_bert.py').exists()}")
    print("\nDirectory contents:")
    for item in local_models_path.iterdir():
        print(f"- {item.name}")
    
    raise ImportError(
        "Could not import DiffusionBertForMaskedLM from any location. "
        "Please ensure the model file exists and is accessible."
    )

def load_model_and_tokenizer(config):
    from transformers import BertTokenizerFast
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model config
    model_config = DiffusionBertConfig(
        vocab_size=config['vocab_size'],
        max_position_embeddings=config['max_position_embeddings'],
        hidden_size=config['hidden_size'],
        intermediate_size=config['intermediate_size'],
        num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers']
    )

    # Initialize model
    print("Initializing model...")
    model = DiffusionBertForMaskedLM(model_config)

    # Load checkpoint
    print(f"Loading checkpoint from {config['model_checkpoint_path']}")
    checkpoint = torch.load(config['model_checkpoint_path'], map_location=device)

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    # Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    return model, tokenizer

if __name__ == "__main__":
    try:
        DiffusionBertForMaskedLM = import_diffusion_bert()
        print("Successfully imported DiffusionBertForMaskedLM")
    except Exception as e:
        print(f"Error importing DiffusionBertForMaskedLM: {str(e)}") 