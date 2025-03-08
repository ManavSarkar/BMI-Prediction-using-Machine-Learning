import subprocess
import sys
import os
import distutils.core

def install_detectron2():
    try:
        # Install PyTorch and dependencies
        subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
        subprocess.run(['pip', 'install', 'cython'])
        subprocess.run(['pip', 'install', 'pyyaml'])

        # Clone detectron2 repository
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/detectron2'])

        # Run setup.py for detectron2 to get install_requires
        dist = distutils.core.run_setup("detectron2/setup.py")

        # Install dependencies listed in install_requires
        install_requires_cmd = ['pip', 'install'] + [f"'{x}'" for x in dist.install_requires]
        subprocess.run(install_requires_cmd)

        # Add detectron2 to sys.path
        sys.path.insert(0, os.path.abspath('./detectron2'))

        print("Detectron2 installation completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")

# install_detectron2()
import torch, detectron2
# !nvcc --version
import detectron2
print("detectron2:", detectron2.__version__)

     