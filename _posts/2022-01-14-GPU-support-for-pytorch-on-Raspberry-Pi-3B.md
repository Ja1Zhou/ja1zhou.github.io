---
title: 'GPU Support for PyTorch on Raspberry Pi 3B'
date: 2022-01-14
permalink: /posts/2022/01/GPU-support-for-pytorch-on-Raspberry-Pi-3B/
tags:
  - PyTorch
  - torch
  - Raspberry Pi
  - Raspberry Pi 3B
  - GPU
  - QPU
---
In this project, we transplanted QPU support for Raspberry Pi 3B to PyTorch scenario. Further, we programmed custom operators and managed to register them to PyTorch.
This project is for academic purposes only

## Overall Idea

- Uses a raspberry pi docker image for cross-compile environment
- Uses open source [QPULib](https://github.com/mn416/QPULib) on github for GPU support
- Uses cross-compilation to speed up compiling process for
  - pytorch
  - custom c++ extensions(for GPU support)

- Registers c++ extension through [PyTorch interface](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- Mounts docker filesystem through nfs to directly run PyTorch

## Detailed Implementation

### 1. Installing cross-compilation environment

#### This process is inspired by this [blog](https://choonkiatlee.github.io/Pytorch-Raspberry-Pi/). My process for finding this blog is as follows:

- I googled for prebuilt PyTorch wheels for Raspberry Pi 3B, and found this [discussion on PyTorch forum](https://discuss.pytorch.org/t/installing-pytorch-on-raspberry-pi-3/25215/19)
- In this dicussion, **choonkiatlee** mentioned this [prebuilt wheel](https://github.com/choonkiatlee/pi-torch) and the [docker image](https://github.com/choonkiatlee/qemu-raspbian) used for cross-compilation
- I found the blog on **choonkiatlee**'s github pages.

#### Install **docker** on host machine (Windows operating system in my case)

#### Pull Raspberry Pi docker image and set up share volume

- ```bash
  docker pull choonkiatlee/raspbian:build
  docker volume create --driver local -o o=bind -o type=none -o device="C:\Users\MyUsername\Downloads\rasp_docker" rasp_docker
  #the purpose of this command is to directly share files between host and docker
  #download torch-1.4.0a0+7f73f1d-cp37-cp37m-linux_armv7l.whl into the volume created
  docker run -it --name rasp -v rasp_docker:/root/rasp_docker choonkiatlee/raspbian:build
  ```

- After entering the docker image environment, the following commands are needed. This is a complement of the previous blog.

  ```bash
  #the docker image comes with python 3.7.3
  apt-get update && apt-get install -y python3-numpy
  #torch requires numpy
  cd /root/rasp_docker
  pip3 install torch-1.4.0a0+7f73f1d-cp37-cp37m-linux_armv7l.whl
  ```

- Testifying

  ```bash
  root@3288e690face:~/rasp_docker# python3
  Python 3.7.3 (default, Jan 22 2021, 20:04:44) 
  [GCC 8.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import torch
  >>> torch.__version__
  '1.4.0a0+7f73f1d'
  ```

- PyTorch is sucessfully installed

#### （Optional）Compile PyTorch from source（~~currently unable to reproduce~~）

- clone repo and set environment variables

  ```bash
  cd /root
  git clone https://github.com/pytorch/pytorch.git
  git checkout v1.4.0
  git submodule sync
  git submodule update --init --recursive
  apt install -y python3-cffi python3-numpy libatlas-base-dev
  pip3 install cython wheel pyyaml pillow
  #choose not to compile extra modules
  export USE_CUDA=0
  export USE_CUDNN=0
  export USE_MKLDNN=0
  export USE_METAL=0
  export USE_NCCL=OFF
  export USE_NNPACK=0
  export USE_QNNPACK=0
  export USE_DISTRIBUTED=0
  export BUILD_TEST=0
  export MAX_JOBS=8
  python3 setup.py install
  ```

- I experienced errors when compiling. It was related to ThridParty **protobuf**. Yet, compilation from source is not the focus of this blog. Prebuilt wheels are used instead.

### 2. Clone QPULib and program custom c++ operators accordingly

#### Structure of QPULib

``````
QPULib/
  Lib/
  	Subdirectories/
  		*.cpp
  		*.h
  	*.cpp
  	*.h
  Doc/
  	irrelevant
  Tests/
  	*.cpp
  	Makefile
``````

#### Acceleration Processes(Quoted from [README.md]([QPULib/README.md at master · mn416/QPULib (github.com)](https://github.com/mn416/QPULib/blob/master/README.md)))

- The [QPU](http://www.broadcom.com/docs/support/videocore/VideoCoreIV-AG100-R.pdf) is a [vector processor](https://en.wikipedia.org/wiki/Vector_processor) developed by [Broadcom](http://www.broadcom.com/) with instructions that operate on 16-element vectors of 32-bit integer or floating point values.
  For example, given two 16-element vectors

  `10 11 12 13` `14 15 16 17` `18 19 20 21` `22 23 24 25`

  and

  `20 21 22 23` `24 25 26 27` `28 29 30 31` `32 33 34 35`

  the QPU's *integer-add* instruction computes a third vector

  `30 32 34 36` `38 40 42 44` `46 48 50 52` `54 56 58 60`

  where each element in the output is the sum of the
  corresponding two elements in the inputs.

  - Each 16-element vector is comprised of four *quads*.  This is where
    the name "Quad Processing Unit" comes from: a QPU processes one quad
    per clock cycle, and a QPU instruction takes four consecutive clock
    cycles to deliver a full 16-element result vector.

  - The Pi contains 12 QPUs in total, each running at 250MHz.  That's a
    max throughput of 750M vector instructions per second (250M cycles
    divided by 4 cycles-per-instruction times 12 QPUs).  Or: 12B
    operations per second (750M instructions times 16 vector elements).
    QPU instructions can in some cases deliver two results at a
    time, so the Pi's QPUs are often advertised at 24
    [GFLOPS](https://en.wikipedia.org/wiki/FLOPS).

- Pipelines are introduced to accelerate the whole computation process

  - QPUs prefetch data when computing

- QPUs are instructed to process specified segments of data, allowing parallelization

When reading Makefile for QPULib, I discovered that the include/ directory is interpreted as Lib/ directory, and *.cpp files under Lib/ will first be compiled to *.o object files. Therefore, I decided to first compile them into a dynamic library and reuse the library.

#### Compiling dynamic library

- Tweaking Makefile to compile QPULib into dynamic library

  ```makefile
  # Root directory of QPULib repository
  ROOT = ../Lib
  
  # Compiler and default flags
  CXX = g++
  CXX_FLAGS = -fpermissive -Wconversion -std=c++0x -I $(ROOT)
  
  # Object directory
  OBJ_DIR = obj
  
  # Debug mode
  ifeq ($(DEBUG), 1)
    CXX_FLAGS += -DDEBUG
    OBJ_DIR := $(OBJ_DIR)-debug
  endif
  
  # QPU or emulation mode
  ifeq ($(QPU), 1)
    CXX_FLAGS += -DQPU_MODE
    OBJ_DIR := $(OBJ_DIR)-qpu
  else
    CXX_FLAGS += -DEMULATION_MODE
  endif
  
  # Object files
  OBJ =                         \
    Kernel.o                    \
    Source/Syntax.o             \
    Source/Int.o                \
    Source/Float.o              \
    Source/Stmt.o               \
    Source/Pretty.o             \
    Source/Translate.o          \
    Source/Interpreter.o        \
    Source/Gen.o                \
    Target/Syntax.o             \
    Target/SmallLiteral.o       \
    Target/Pretty.o             \
    Target/RemoveLabels.o       \
    Target/CFG.o                \
    Target/Liveness.o           \
    Target/RegAlloc.o           \
    Target/ReachingDefs.o       \
    Target/Subst.o              \
    Target/LiveRangeSplit.o     \
    Target/Satisfy.o            \
    Target/LoadStore.o          \
    Target/Emulator.o           \
    Target/Encode.o             \
    VideoCore/Mailbox.o         \
    VideoCore/Invoke.o          \
    VideoCore/VideoCore.o
  
  # Top-level targets
  
  .PHONY: top clean
  LIB = $(patsubst %,$(OBJ_DIR)/%,$(OBJ))
  top: $(LIB)
          @$(CXX) $(CXX_FLAGS) -shared -fPIC $^ -o libqpu.so
  
  clean:
          rm -rf obj obj-debug obj-qpu obj-debug-qpu
          rm -f Tri GCD Print MultiTri AutoTest OET Hello ReqRecv Rot3D ID *.o
          rm -f HeatMap
          rm -f libqpu.so
  # Intermediate targets
  
  $(OBJ_DIR)/%.o: $(ROOT)/%.cpp $(OBJ_DIR)
          @echo Compiling $<
          @$(CXX) -c -o $@ $< $(CXX_FLAGS)
  
  %.o: %.cpp
          @echo Compiling $<
          @$(CXX) -c -o $@ $< $(CXX_FLAGS)
  
  $(OBJ_DIR):
          @mkdir -p $(OBJ_DIR)
          @mkdir -p $(OBJ_DIR)/Source
          @mkdir -p $(OBJ_DIR)/Target
          @mkdir -p $(OBJ_DIR)/VideoCore
  ```

- All files under Lib/ will be compiled into libqpu.so

- Add libqpu.so to system dynamic lib path

  ```bash
  vim /etc/ld.so.conf
  # add a new path to the compiled libqpu.so, in my case->
  /root/embedded_project/
  #:wq save and exit
  ldconfig#refresh path
  ```

#### Program C++ operators for parallelization 

- Pipelined multiple QPU operator

  - This is a simple implementation of restrained matrix dot product

  ```c++
  //"dot.cpp"
  #include <torch/extension.h>
  #include <vector>
  #include <QPULib.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/time.h>
  const int NQPUS = 4; //number of QPUs
  
  void dotproduct(Int n, Ptr<Float> x, Ptr<Float> y)
  {
      Int inc = numQPUs() << 4;
      Ptr<Float> p = x + index() + (me() << 4);
      Ptr<Float> q = y + index() + (me() << 4);
      gather(p); gather(q);
  
      Float xOld, yOld;
      For (Int i = 0, i < n, i = i+inc)
          gather(p+inc); gather(q+inc);//prefetch
          receive(xOld); receive(yOld);//computation
          store(xOld * yOld, p);
          p = p+inc; q = q+inc;
      End
  
      receive(xOld); receive(yOld);
  }
  
  void dotadd(Int n, Ptr<Float> x, Ptr<Float> y)
  {
      Int inc = numQPUs() << 4;
      Ptr<Float> p = x + index() + (me() << 4);
      Ptr<Float> q = y + index() + (me() << 4);
      gather(p); gather(q);
  
      Float xOld, yOld;
      For (Int i = 0, i < n, i = i+inc)
          gather(p+inc); gather(q+inc);
          receive(xOld); receive(yOld);
          store(xOld + yOld, p);
          p = p+inc; q = q+inc;
      End
  
      receive(xOld); receive(yOld);
  }
  
  torch::Tensor dot_product(torch::Tensor input, torch::Tensor weight)
  {
      input = input.to(torch::kFloat32);
      weight = weight.to(torch::kFloat32);
      float *input_ptr = (float *)input.data_ptr();
      float *weight_ptr = (float *)weight.data_ptr();
  
      int width = weight.numel();
      int width_16 = width + (16 - width % 16);//QPU length is 16
      SharedArray<float> mapA(width_16), mapB(width_16);
  
      for (int i = 0; i < width_16; ++i)
      {
          if (i < width)
          {
              mapA[i] = input_ptr[i];
              mapB[i] = weight_ptr[i];
          }
          else
          {
              mapA[i] = 0;//adding zeros
              mapB[i] = 0;
          }
      }
      auto k = compile(dotproduct);
  
      k.setNumQPUs(NQPUS);
  
      k(width, &mapA, &mapB);
  
      for (int i = 0; i < width; i++) {
          input_ptr[i] = mapA[i];
      }
      return input;
  }
  
  torch::Tensor dot_add(torch::Tensor input, torch::Tensor weight)
  {
      input = input.to(torch::kFloat32);
      weight = weight.to(torch::kFloat32);
      float *input_ptr = (float *)input.data_ptr();
      float *weight_ptr = (float *)weight.data_ptr();
  
      int width = weight.numel();
      int width_16 = width + (16 - width % 16);
      SharedArray<float> mapA(width_16), mapB(width_16);
  
      for (int i = 0; i < width_16; ++i)
      {
          if (i < width)
          {
              mapA[i] = input_ptr[i];
              mapB[i] = weight_ptr[i];
          }
          else
          {
              mapA[i] = 0;
              mapB[i] = 0;
          }
      }
      auto k = compile(dotadd);
  
      k.setNumQPUs(NQPUS);
  
      k(width, &mapA, &mapB);
  
      for (int i = 0; i < width; i++) {
          input_ptr[i] = mapA[i];
      }
      
      return input;
  }
  
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &dot_add, "dot_add");
    m.def("product", &dot_product, "dot_product");
  }
  ```

  

- Another example of H x W matrix times W x 1 matrix, returning H x 1 matrix

  ```c++
  //matrix.cpp
  #include <torch/extension.h>
  #include <vector>
  #include <QPULib.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/time.h>
  const int NQPUS = 4;
  struct Cursor
  {
      Ptr<Float> addr;
      Float current, next;
  
      void init(Ptr<Float> p)
      {
          gather(p);
          current = 0;
          addr = p + 16;
      }
  
      void prime()
      {
          receive(next);
          gather(addr);
      }
  
      void advance()
      {
          addr = addr + 16;
          gather(addr);
          current = next;
          receive(next);
      }
  
      void finish()
      {
          receive(next);
      }
  };
  
  void step(Ptr<Float> map, Ptr<Float> weight, Ptr<Float> mapOut, Int pitch, Int width, Int height)
  {
      Cursor row, cursorofweight;
      map = map + pitch * me() + index();
      For(Int y = me(), y < height, y = y + numQPUs())
  
          Ptr<Float>
          p = mapOut + y * 16;
  
      // init
      row.init(map);
      row.prime();
      cursorofweight.init(weight);
      cursorofweight.prime();
  
      // computation
      Float accumulate = 0;
      For(Int x = 0, x < width, x = x + 16)
          row.advance();
     		cursorofweight.advance();
      	accumulate = accumulate + row.current * cursorofweight.current;
  
      End
      // storing
      store(accumulate, p);
      // release Cursor()
      row.finish();
      cursorofweight.finish();
      map = map + pitch * numQPUs();
  
      End
  }
  torch::Tensor accumartix(torch::Tensor input, torch::Tensor weight)
  {
      input = input.to(torch::kFloat32);
      weight = weight.to(torch::kFloat32);
      int width = weight.numel();
      int width_16 = width + (16 - width % 16);
      int height = input.numel() / width;
  
      float *input_ptr = (float *)input.data_ptr();
      float *weight_ptr = (float *)weight.data_ptr();
      // creating shared array between QPU and CPU
      SharedArray<float> mapA(width_16 * height), mapB(width_16), sumofmartix(16 * height);
      for (int i = 0; i < height; ++i)
      {
          for (int j = 0; j < width_16; ++j)
          {
              if (j < width)
                  mapA[i * width_16 + j] = input_ptr[i * width + j];
              else
                  mapA[i * width_16 + j] = 0;
          }
      }
  
      for (int j = 0; j < height; ++j)
      {
          for (int i = 0; i < 16; ++i)
          {
              sumofmartix[16 * j + i] = 0;
          }
      }
  
      for (int j = 0; j < width_16; ++j)
      {
          if (j < width)
              mapB[j] = weight_ptr[j];
          else
              mapB[j] = 0;
      }
      auto k = compile(step);
  
      k.setNumQPUs(NQPUS);
  
      k(&mapA, &mapB, &sumofmartix, width_16, width, height);
      torch::Tensor ans = torch::zeros(height);
      float *ans_ptr = (float *)ans.data_ptr();
      for (int j = 0; j < height; ++j)
      {
          for (int i = 0; i < 16; ++i)
          {
              ans_ptr[j] += sumofmartix[16 * j + i];
          }
      }
      return ans; 
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gpu", &accumartix, "accumartix");
  }
  ```


### 3. Registering C++ operator to PyTorch

#### [PyTorch Official Documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html)

- .cpp files and setup.py is needed

#### Compile and register custom operator

- setup.py needes to be modified considering the args to compile QPULib and the paths to the dependent files

  ```python
  from setuptools import setup, Extension
  from torch.utils import cpp_extension
  from torch.utils.cpp_extension import BuildExtension, CppExtension
  
  setup(
      name='dot_cpp',
      ext_modules=[
          Extension(
              name='dot_cpp',
              sources=['dot.cpp'],
              include_dirs=cpp_extension.include_paths()+["/root/QPULib/Lib", "/root/QPULib/Lib/Source",
              "/root/QPULib/Lib/Target","/root/QPULib/Lib/VideoCore","/root/QPULib/Lib/Common"],
              library_dirs=["."],
              libraries=["qpu"],
              language='c++',
              extra_compile_args = ["-fpermissive","-w","-std=c++0x","-DQPU_MODE"])
              
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
  
  ```

- the sturcture of the files to be registered is as follows

  ```bash
  embedded_project/
  	dot.cpp
  	setup.py
  	libqpu.so
  ```

- run the following command under embedded_project/

  ```bash
  python3 setup.py install
  ```

- Test import after installation

  - note that this process does not invoke driver registration and therefore can be tested off Raspberry Pi

  ```python
  import torch
  import dot_cpp
  ```

- Proceed if no errors occur(docker commit and docker push to your docker hub)

### 4. nfs mount docker filesystem on Raspberry Pi 3B

#### environment preparation

- I am using a Debian system as nfs host

- Install docker services, pull docker image and cp entire docker filesystem

  ```bash
  sudo apt update && sudo apt install qemu qemu-user-static binfmt-support
  #qemu needed in order to simulation arm on x64
  docker pull yourUsername/yourImage
  docker run -it --name rasp yourUsername/yourImage
  #docker cp entire filesystem
  sudo docker cp rasp:/ ~/home/username/rasp_docker_filesystem
  ```

- boot using nfs filesystem, mount /home/username/rasp_docker_filesystem and chroot

  ```bash
  mount 192.168.0.101:/home/username/rasp_docker_filesystem /mnt -o nolock
  cd /mnt
  mount --rbind /dev dev/#mount devices
  chroot .
  ```

#### On-board test

- validating accuracy of qpu computation

  ```python
  # accuracy.py
  import pytorch as torch
  import time
  import dot_cpp
  
  a = torch.randint(100)
  b = torch.randint(100)
  c = a * b
  print("ans in pytorch:")
  print(c)
  d = dot_cpp.product(a, b)
  print("ans in gpu:")
  print(d)
  ```

  - Results:

  ```python
  ans in pytorch:
  tensor([-5.9086e-02, -4.3276e+00, -6.5376e-01,  5.0014e-01, -1.2216e-01,
           8.5097e-02, -1.4941e+00,  3.5625e+00,  1.2412e-03,  4.9355e-01,
          -4.8173e-01,  1.3379e-01,  6.8660e-01, -3.0867e-01,  4.1459e-01,
           3.8146e-01,  2.6874e-01, -1.0085e-01, -1.9247e-01, -3.8177e-01,
          -7.2695e-01, -7.9857e-01,  9.2179e-01, -4.4537e-01,  1.2229e+00,
          -1.9606e+00,  2.1500e+00,  6.2939e-02, -2.9404e-02, -1.6333e-01,
           5.8653e-01, -3.0282e-01,  1.7500e+00, -1.9485e+00,  1.0097e+00,
          -2.9966e-01,  5.1717e-01,  8.6291e-01,  1.4203e+00,  1.5049e-01,
           4.0039e-01, -2.1761e-01, -2.7387e-02, -5.7702e-01,  5.4926e-02,
          -2.1086e-01, -2.1043e-01, -4.2422e-01,  3.1212e-02, -3.5714e-01,
           7.3226e-01,  1.7916e+00, -8.3882e-02,  1.7431e+00,  7.5411e-02,
           1.4379e-01, -2.1750e+00,  5.3509e-01,  1.9931e+00, -1.0812e+00,
           9.5756e-01, -2.2465e-01, -2.7048e-01, -5.4887e-01,  4.8681e-01,
          -5.7749e-02,  8.6992e-02, -7.8780e-01,  1.3495e+00, -7.5135e-02,
           6.2448e-01, -1.1303e-02, -1.0266e-01, -1.4959e+00, -1.6517e+00,
           1.1846e-01,  1.5355e+00, -4.2969e-01,  2.9539e-01, -5.9056e-01,
           1.0564e+00, -5.7899e-01,  1.7013e-02,  5.1986e-01, -4.7120e-02,
          -3.4399e-02, -1.4235e-01, -1.4144e+00,  5.1103e-01,  7.2233e-01,
          -6.0687e-01, -8.2988e-01, -2.7205e-01,  1.0952e+00, -9.7423e-02,
           4.9439e-02, -1.7460e-02,  2.0516e-01, -7.8793e-01, -1.8765e+00])
  ans in gpu:
  tensor([-5.9086e-02, -4.3276e+00, -6.5376e-01,  5.0014e-01, -1.2216e-01,
           8.5097e-02, -1.4941e+00,  3.5625e+00,  1.2412e-03,  4.9355e-01,
          -4.8173e-01,  1.3379e-01,  6.8660e-01, -3.0867e-01,  4.1459e-01,
           3.8146e-01,  2.6874e-01, -1.0085e-01, -1.9247e-01, -3.8177e-01,
          -7.2695e-01, -7.9857e-01,  9.2179e-01, -4.4537e-01,  1.2229e+00,
          -1.9606e+00,  2.1500e+00,  6.2939e-02, -2.9404e-02, -1.6333e-01,
           5.8653e-01, -3.0282e-01,  1.7500e+00, -1.9485e+00,  1.0097e+00,
          -2.9966e-01,  5.1717e-01,  8.6291e-01,  1.4203e+00,  1.5049e-01,
           4.0039e-01, -2.1761e-01, -2.7387e-02, -5.7702e-01,  5.4926e-02,
          -2.1086e-01, -2.1043e-01, -4.2422e-01,  3.1212e-02, -3.5714e-01,
           7.3226e-01,  1.7916e+00, -8.3882e-02,  1.7431e+00,  7.5411e-02,
           1.4379e-01, -2.1750e+00,  5.3509e-01,  1.9931e+00, -1.0812e+00,
           9.5756e-01, -2.2465e-01, -2.7048e-01, -5.4887e-01,  4.8681e-01,
          -5.7749e-02,  8.6992e-02, -7.8780e-01,  1.3495e+00, -7.5135e-02,
           6.2448e-01, -1.1303e-02, -1.0266e-01, -1.4959e+00, -1.6517e+00,
           1.1846e-01,  1.5355e+00, -4.2969e-01,  2.9539e-01, -5.9056e-01,
           1.0564e+00, -5.7899e-01,  1.7013e-02,  5.1986e-01, -4.7120e-02,
          -3.4399e-02, -1.4235e-01, -1.4144e+00,  5.1103e-01,  7.2233e-01,
          -6.0687e-01, -8.2988e-01, -2.7205e-01,  1.0952e+00, -9.7423e-02,
           4.9439e-02, -1.7460e-02,  2.0516e-01, -7.8793e-01, -1.8765e+00])
  ```

  

- comparing computational speed of qpu with cpu

  - We tweeked the C++ operator to return computational time in a specific manner. We also wrote similar CPU operator for comparison
  - Here, we only compared the computational time on the C++ side. Actually, when using the gpu operator from the python interface, both the call process and the communication between cpu & gpu would consume considerable time.
  - This project exhibits the huge potential of integrating complex operators using the C++ GPU library for acceleration

  ```c++
  //tweaked martix.cpp
  #include <torch/extension.h>
  #include <vector>
  #include <QPULib.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/time.h>
  const int NQPUS = 12;
  struct Cursor
  {
      Ptr<Float> addr;
      Float current, next;
  
      void init(Ptr<Float> p)
      {
          gather(p);
          current = 0;
          addr = p + 16;
      }
  
      void prime()
      {
          receive(next);
          gather(addr);
      }
  
      void advance()
      {
          addr = addr + 16;
          gather(addr);
          current = next;
          receive(next);
      }
  
      void finish()
      {
          receive(next);
      }
  };
  
  void step(Ptr<Float> map, Ptr<Float> weight, Ptr<Float> mapOut, Int pitch, Int width, Int height)
  {
      Cursor row, cursorofweight;
      map = map + pitch * me() + index();
      For(Int y = me(), y < height, y = y + numQPUs())
  
          // Point p to the output row
          Ptr<Float>
              p = mapOut + y * 16;
  
      // Initilaise the cursors for the three input rows
      row.init(map);
      row.prime();
      cursorofweight.init(weight);
      cursorofweight.prime();
  
      // Compute one output row
      Float accumulate = 0;
      For(Int x = 0, x < width, x = x + 16)
          // calculate this iteration and receive data for the next
          row.advance();
      cursorofweight.advance();
      // accumulate the row[x] and weight[i]
      accumulate = accumulate +  row.current * cursorofweight.current;
  
      End
          // store the output of every index in [0:16]
          store(accumulate, p);
      // Cursors are finished for this row
      row.finish();
      cursorofweight.finish();
      // Move to the next input rows
      map = map + pitch * numQPUs();
  
      End
  }
  torch::Tensor accumartix(torch::Tensor input, torch::Tensor weight)
  {
      input = input.to(torch::kFloat32);
      weight = weight.to(torch::kFloat32);
      int width = weight.numel();
      int width_16 = width + (16 - width % 16);
      int height = input.numel() / width;
  
      float *input_ptr = (float *)input.data_ptr();
      float *weight_ptr = (float *)weight.data_ptr();
      // Allocate and initialise input and output maps
      SharedArray<float> mapA(width_16 * height), mapB(width_16), sumofmartix(16 * height);
      for (int i = 0; i < height; ++i)
      {
          for (int j = 0; j < width_16; ++j)
          {
              if (j < width)
                  mapA[i * width_16 + j] = input_ptr[i * width + j];
              else
                  mapA[i * width_16 + j] = 0;
          }
      }
  
      for (int j = 0; j < height; ++j)
      {
          for (int i = 0; i < 16; ++i)
          {
              sumofmartix[16 * j + i] = 0;
          }
      }
  
      for (int j = 0; j < width_16; ++j)
      {
          if (j < width)
              mapB[j] = weight_ptr[j];
          else
              mapB[j] = 0;
      }
  
      // Compile kernel
      auto k = compile(step);
  
      // Invoke kernel
      k.setNumQPUs(NQPUS);
      timeval tvStart,tvEnd,tvDiff;
      gettimeofday(&tvStart,NULL);
      k(&mapA, &mapB, &sumofmartix, width_16, width, height);
      gettimeofday(&tvEnd,NULL);
      torch::Tensor ans = torch::zeros(height);
      float *ans_ptr = (float *)ans.data_ptr();
      for (int j = 0; j < height; ++j)
      {
          for (int i = 0; i < 16; ++i)
          {
              ans_ptr[j] += sumofmartix[16 * j + i];
          }
      }
      // Run-time of simulation
      timersub(&tvEnd,&tvStart,&tvDiff);
      ans_ptr[0] = tvDiff.tv_sec;
      ans_ptr[1] = tvDiff.tv_usec;
      return ans; //we directly return the computational time because accuracy has been tested
  }
  
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gpu", &accumartix, "accumartix");
  }
  ```

  ```python
  # time.py
  # only computational time is returned
  import torch
  import time
  import cpu_cpp
  import matrix_cpp
  
  for i in range(0,6):
      a = torch.randn(100 *10**i )
      b=  torch.randn(10**i)
      gpu = matrix_cpp.gpu(a,b)
      cpu=cpu_cpp.cpu(a,b)
      print("cpu 100 * 10 ** %d takes %d.%06d"% (i,cpu[0],cpu[1]))
      print("gpu 100 * 10 ** %d takes %d.%06d"% (i,gpu[0],gpu[1]))
  ```

  - Results

  ```bash
  cpu 100 * 10 ** 0 takes 0.000004
  gpu 100 * 10 ** 0 takes 0.000164
  cpu 100 * 10 ** 1 takes 0.000023
  gpu 100 * 10 ** 1 takes 0.000169
  cpu 100 * 10 ** 2 takes 0.000206
  gpu 100 * 10 ** 2 takes 0.000171
  cpu 100 * 10 ** 3 takes 0.002116
  gpu 100 * 10 ** 3 takes 0.000388
  cpu 100 * 10 ** 4 takes 0.021245
  gpu 100 * 10 ** 4 takes 0.003079
  cpu 100 * 10 ** 5 takes 0.214486
  gpu 100 * 10 ** 5 takes 0.029622
  ```

## Acknowledgements

This is a preliminary result of a course project

I would like to thank my groupmate **lxk** and my groupmate **cxf** 

I would also like to thank Professor Yang and Professor Lv for their kind support and helpful advice