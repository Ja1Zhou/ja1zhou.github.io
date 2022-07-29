---
title: 'Building PyTorch From Source with CUDA Support on Windows'
date: 2022-01-23
permalink: /posts/2022/01/Building-PyTorch-from-source-on-Windows/
tags:
  - PyTorch
  - torch
  - CUDA
---
It is common to build PyTorch from source to support specific versions of CUDA or for customization. Building PyTorch from source on Windows platform is somewhat messier than building on Linux platforms. I hope to better organize the building procedures in this blog. 

The contents of this blog is largely indentical to the contents in the official [README](https://github.com/pytorch/pytorch#from-source) and a recent blog [here](https://datagraphi.com/blog/post/2021/9/13/building-pytorch-from-source-on-windows-to-work-with-an-old-gpu).

## Crucial Components List
For the build to be successful, here is a checklist for things to install or store on your computer. They are presented in recommended installation order.
- [Anaconda](https://www.anaconda.com/distribution/#download-section)
    - optional but highly recommended
    - I personally prefer [Miniconda](https://docs.conda.io/en/latest/miniconda.html). It is more compact and command-line based.
- [The PyTorch repo](https://github.com/pytorch/pytorch)
    - along with third-party dependencies
- [Microsoft build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    - installing through vs2019 also works fine
    - appart from default build tools, Windows SDK is also needed
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
    - make sure that NVTX(Nsight Compute) is also installed
- [NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
    - there is a installation guide from NVIDIA [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)


## My Version Info
- CUDA 11.5
- cuDNN 8.3.2
- Windows 11
- PyTorch master branch

## Preparing Conda Environment

```bash
# Create environment according to your preferences
conda create -n torch python=3.9 -y
conda activate torch
# Install dependencies
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

## Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# If you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --depth 1
# In my case, git submodule update constently hangs 
# Therefore, I added --depth 1 
# The original command from PyTorch repo is 
# git submodule update --init --recursive --jobs 0
```
## Installing Additional Libraries
### Magma
Download magma through the following link:
[https://s3.amazonaws.com/ossci-windows/magma_2.5.4_cuda115_release.7z](https://s3.amazonaws.com/ossci-windows/magma_2.5.4_cuda115_release.7z)

Note that you should replace 115 with your cuda version.

Up till now, the newest version of magma is 2.6.1 and the newest CUDA version is 11.6. The newest supported magma version on this website is 2.5.4 and the newest supported CUDA version is 11.5.
### mkl
Download mkl through the following website:
[https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z](https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z)

### Extract the zip files
Extract the two downloaded zip files to a folder such as `C:\Users\username\Documents\pytorch_dependencies`. The folder structure is as follows:
```
pytorch_dependencies/
    magma_2.5.4_cuda115_release/
        lib/
        include/
    mkl_2020.2.254/
        lib/
        include/
```
## Setting Environment Variables
Note that MAGMA_HOME is for magma library. CMAKE_INCLUDE_PATH and LIB is for mkl library.
You might need to modify the path according to your situation. 
```powershell
$Env:CMAKE_INCLUDE_PATH="C:\Users\username\Documents\pytorch_dependencies\mkl_2020.2.254\include"
$Env:LIB="C:\Users\username\Documents\pytorch_dependencies\mkl_2020.2.254\lib"
$Env:MAGMA_HOME="C:\Users\username\Documents\pytorch_dependencies\magma_2.5.4_cuda115_release"
$Env:CMAKE_GENERATOR="Ninja"
# uses NINJA to speed up compilation
$Env:BUILD_TEST=0
# disables test build
$Env:USE_ROCM=0
# disables ROCm build
$Env:TORCH_CUDA_ARCH_LIST="7.5"
# optinal, specifies architecture can be found here:
# https://developer.nvidia.com/cuda-gpus
```

## Install PyTorch
```bash
python setup.py install
# for some situations, --cmake is required to force regenerate cmake configurations
```

