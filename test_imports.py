import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import diffusion
    print("Successfully imported diffusion module")
except ImportError as e:
    print(f"Failed to import diffusion module: {str(e)}")

try:
    from models.modeling_diffusion_bert import DiffusionBertForMaskedLM
    print("Successfully imported DiffusionBertForMaskedLM")
except ImportError as e:
    print(f"Failed to import DiffusionBertForMaskedLM: {str(e)}")

try:
    from sample import Categorical, WholeWordMasking
    print("Successfully imported sampling classes")
except ImportError as e:
    print(f"Failed to import sampling classes: {str(e)}") 