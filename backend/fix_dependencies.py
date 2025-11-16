"""
Script to fix torch/torchvision compatibility issues.
Run this if you're experiencing pipeline import errors.
"""
import subprocess
import sys

def fix_dependencies():
    print("Fixing torch/torchvision compatibility...")
    print("=" * 50)
    
    # Uninstall problematic packages
    print("\n1. Uninstalling torch and torchvision...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], 
                  check=False)
    
    # Reinstall compatible versions
    print("\n2. Installing compatible torch and torchvision...")
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0"], 
                  check=True)
    
    # Verify installation
    print("\n3. Verifying installation...")
    try:
        import torch
        import torchvision
        print(f"✓ torch version: {torch.__version__}")
        print(f"✓ torchvision version: {torchvision.__version__}")
        
        # Test pipeline import
        print("\n4. Testing transformers pipeline import...")
        from transformers import pipeline
        print("✓ Pipeline import successful!")
        
        print("\n" + "=" * 50)
        print("Dependencies fixed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nYou may need to manually install:")
        print("  pip install torch==2.1.0 torchvision==0.16.0")
        return False

if __name__ == "__main__":
    fix_dependencies()

