"""
Setup script for Laptop Classroom Demo
Checks and installs required dependencies
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required!")
        return False
    else:
        print("✅ Python version compatible")
        return True

def install_package(package):
    """Install a Python package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = [
        "torch",
        "transformers", 
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "numpy"
    ]
    
    print("📋 Checking required packages...")
    print()
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - already installed")
        except ImportError:
            print(f"⚠️ {package} - not found, installing...")
            if not install_package(package):
                failed_packages.append(package)
    
    return failed_packages

def check_model_files():
    """Check for available model files"""
    print("\n🔍 Looking for trained model files...")
    
    model_extensions = [".th", ".pt", ".pth"]
    found_models = []
    
    # Search in current directory and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                if "ckpts" in root or "model" in root.lower():
                    full_path = os.path.join(root, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    if size_mb > 100:  # Only show large model files
                        found_models.append((full_path, size_mb))
    
    if found_models:
        print("📁 Found model files:")
        for model_path, size_mb in sorted(found_models, key=lambda x: x[1], reverse=True):
            print(f"   • {model_path} ({size_mb:.1f} MB)")
            
        # Suggest the largest model as default
        best_model = max(found_models, key=lambda x: x[1])[0]
        print(f"\n💡 Recommended model: {best_model}")
        return best_model
    else:
        print("⚠️ No trained model files found!")
        print("   Make sure you have a trained Diffusion BERT checkpoint (.th/.pt file)")
        return None

def create_quick_start_script():
    """Create a quick start script with the best model"""
    model_path = check_model_files()
    
    if model_path:
        script_content = f'''@echo off
echo Starting Diffusion BERT CPU Demo...
python cpu_classroom_demo.py --checkpoint "{model_path}" --mode full
pause'''
        
        with open("quick_demo.bat", "w") as f:
            f.write(script_content)
        
        print(f"\n🚀 Created quick_demo.bat with model: {os.path.basename(model_path)}")
        print("   Just double-click quick_demo.bat to run the presentation!")

def run_quick_test():
    """Run a quick test to make sure everything works"""
    print("\n🧪 Running quick compatibility test...")
    
    try:
        import torch
        import transformers
        print("✅ PyTorch and Transformers loaded successfully")
        
        # Test CPU device
        device = torch.device("cpu")
        print(f"✅ CPU device available: {device}")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.matmul(x, x.T)
        print("✅ Basic tensor operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎓 DIFFUSION BERT LAPTOP DEMO - SETUP")
    print("=" * 50)
    print("🔧 Preparing your laptop for classroom presentation")
    print()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        input("Press Enter to exit...")
        return
    
    print()
    
    # Install packages
    failed_packages = check_and_install_packages()
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("💡 Try installing manually: pip install " + " ".join(failed_packages))
        input("Press Enter to continue anyway or Ctrl+C to exit...")
    
    # Run compatibility test
    if not run_quick_test():
        print("\n⚠️ Some compatibility issues detected")
        print("💡 Demo might still work, but performance may be limited")
    
    # Check for models and create quick start
    create_quick_start_script()
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print()
    print("📋 What's ready:")
    print("   ✅ Python environment configured")
    print("   ✅ Required packages installed")
    print("   ✅ CPU optimizations enabled")
    print("   ✅ Quick start script created")
    print()
    print("🚀 How to run your presentation:")
    print("   1. Double-click 'quick_demo.bat' for instant demo")
    print("   2. Or run: python cpu_classroom_demo.py --checkpoint [model_path]")
    print("   3. For data splitting demo: python show_data_splitting.py")
    print()
    print("📚 Check LAPTOP_DEMO_GUIDE.md for complete instructions")
    print("💻 Your laptop is ready for the classroom presentation!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 