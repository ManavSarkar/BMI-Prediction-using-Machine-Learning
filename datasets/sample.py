import subprocess

def install_detectron2():
    try:
        # Install PyTorch and its dependencies
        subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
        
        # Install additional packages
        subprocess.run(['pip', 'install', 'cython'])
        subprocess.run(['pip', 'install', 'pyyaml'])
        
        # Clone the Detectron2 repository
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/detectron2.git'])
        
        # Install Detectron2 in editable mode
        subprocess.run(['pip', 'install', '-e', 'detectron2'])
        
        print("Detectron2 installation completed successfully!")

    except Exception as e:
        print(f"Error during installation: {e}")

# Run the function
install_detectron2()