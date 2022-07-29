---
title: '树莓派3B 实现通过GPU加速pytorch计算'
date: 2022-01-14
permalink: /posts/2022/01/GPU-support-for-pytorch-on-Raspberry-Pi-3B-CN-version/
tags:
  - PyTorch
  - torch
  - Raspberry Pi
  - Raspberry Pi 3B
  - GPU
  - QPU
---
本项目将树莓派3B的GPU库移植到了PyTorch场景，编写了自定义算子并向PyTorch成功注册
本项目仅用于学术用途

## 总思路

- 交叉编译环境为raspberry pi 3B docker镜像
- 使用github上开源的GPU库[QPULib](https://github.com/mn416/QPULib)
- 在树莓派上安装pytorch，通过交叉编译加速过程
- 通过pytorch提供的[接口](https://pytorch.org/tutorials/advanced/cpp_extension.html)编译并且注册c++端的程序
- 通过nfs mount docker镜像的file system，直接运行pytorch

## 具体实现

### 1. 安装交叉编译环境

#### 具体过程收到这篇[博客](https://choonkiatlee.github.io/Pytorch-Raspberry-Pi/)启发，找到这篇博客的过程如下

- 在搜索引擎搜索预编译的树莓派3B python wheel，搜索到pytorch论坛中的这条[帖子](https://discuss.pytorch.org/t/installing-pytorch-on-raspberry-pi-3/25215/19)
- 在这条帖子中，名为**choonkiatlee**的用户提到了其编译好的[wheel](https://github.com/choonkiatlee/pi-torch)以及编译使用的[docker镜像](https://github.com/choonkiatlee/qemu-raspbian)
- 在这名用户的github博客界面中，找到了上述文章

#### 在我的主机（**windows**电脑上）安装**docker for windows**，并进行一系列**镜像加速**、**托管**配置

##### 托管服务

- 登录[阿里云](https://homenew.console.aliyun.com/home/dashboard/ProductAndService)，点击容器镜像服务

- 选择个人实例

- 在仓库管理中创建个人命名空间，我的情况下是**ja1zhou**

- 创建访问凭证，选择固定密码，这个密码为本地进行登录时进行验证的密码

- 在主机上启动**docker engine**，并在终端输入

  ```bash
  sudo docker login --username=your_username registry.cn-beijing.aliyuncs.com
  #之后输入设置的密码
  ```

  成功登录阿里云

- 在**镜像仓库**中创建新的仓库名称，我的情况下为**myrasp**

##### 镜像加速

- 根据阿里云[官方文档](https://cr.console.aliyun.com/cn-beijing/instances/mirrors)进行镜像加速配置

- 直接修改**docker for windows**的daemon.json文件，添加

  ```json
  {
    "registry-mirrors": ["https://your_server_name.mirror.aliyuncs.com"]
  }
  ```

#### 拉取工作镜像，配置shared volume，并设置工作环境

- ```bash
  docker pull choonkiatlee/raspbian:build
  docker volume create --driver local -o o=bind -o type=none -o device="C:\Users\MyUsername\Downloads\rasp_docker" rasp_docker
  #这条命令的目的是直接将下载中的诸多文件，比如wheel和QPULib，与docker进行共享
  #将torch-1.4.0a0+7f73f1d-cp37-cp37m-linux_armv7l.whl下载并且放置在创建的volume中
  docker run -it --name rasp -v rasp_docker:/root/rasp_docker choonkiatlee/raspbian:build
  ```

- 进入docker容器之后，需要输入以下的命令进行配置，下述命令是对上文博客的补充

  ```bash
  #docker自带的python版本为3.7.3
  apt-get update && apt-get install -y python3-numpy
  #torch运行需要numpy支持
  cd /root/rasp_docker
  pip3 install torch-1.4.0a0+7f73f1d-cp37-cp37m-linux_armv7l.whl
  ```

- 进行验证

  ```bash
  root@3288e690face:~/rasp_docker# python3
  Python 3.7.3 (default, Jan 22 2021, 20:04:44) 
  [GCC 8.3.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import torch
  >>> torch.__version__
  '1.4.0a0+7f73f1d'
  ```

- torch已经成功安装

- 将这个版本的docker镜像打包，作为后续开发的起点

  ```bash
  docker commit rasp registry.cn-beijing.aliyuncs.com/ja1zhou/myrasp:torch
  docker push registry.cn-beijing.aliyuncs.com/ja1zhou/myrasp:torch
  ```

#### （可选）从源码编译pytorch（~~暂未成功复现~~）

- 设置代理，允许代软件通过内网、公网防火墙

- 设置代理软件，允许来自局域网的连接

- 在docker镜像中，设置环境变量

  ```bash
  export all_proxy="http://host_ip:host_port"
  ```

- 下载并进行准备工作

  ```bash
  cd /root
  git clone https://github.com/pytorch/pytorch.git
  git checkout v1.4.0
  git submodule sync
  git submodule update --init --recursive
  apt install -y python3-cffi python3-numpy libatlas-base-dev
  pip3 install cython wheel pyyaml pillow
  #选择不编译一切附加模块
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

- 报错，是与submodule protobuf相关，当时作者也遇到了这个问题，但是不知道他具体编译的时候使用的是protobuf的哪个tag，已经在github上面提交了issue

### 2. 下载QPULib，根据其提供的功能进行代码编写

#### QPULib代码结构

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

#### QPU加速原理

- QPU 是由 Broadcom 开发的矢量处理器，其指令可对 32 位整数或浮点值的 16 元素向量进行操作。例如，给定两个 16 元素向量

  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25

  和

  20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35

  QPU 的整数加法指令计算第三个向量

  30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60

  其中输出中的每个元素是输入中对应的两个元素的总和。

  - 每个 16 元素向量由四个四分之一的矢量部分组成。

  - QPU 每个时钟周期处理一个矢量部分，而 QPU 指令需要四个连续的时钟周期来提供完整的16位的结果矢量，这就是“QPU”名称的由来。

  - Pi 总共包含 12 个 QPU，每个都以 250MHz 的频率运行。这是每秒 750M 向量指令的最大吞吐量（250M 周期除以 4 个周期每条指令乘以 12 个QPU）。或者：每秒 12B 次操作（750M 指令乘以 16 个向量元素）。在某些情况下，QPU 指令可以一次提供两个结果，因此 Pi 的 QPU 通常以 24 GFLOPS 进行工作。

  - QPU 是 Raspberry Pi 图形管道的一部分。如果想在 Pi 上制作高效的图形，那么可能需要 OpenGL ES。但是，如果只想尝试加速 Pi 项目的非图形部分，那么 QPULib 值得一看。 

- 而为了避免计算时候的阻塞，可以通过引入流水线的方式。

  - 在计算该次数据的同时，去取下一次计算所需要的数据

- 同时，可以引入多QPU并行计算的方式，及每次采取多QPU计算不同区域的数据，从而实现GPU的高效利用。

- 具体阅读Makefile，发现编译时Include目录为Lib/目录，会先将Lib/下的所有*.cpp编译成\*.o 

- 最后，将*.o和链接到可执行文件

#### 编译动态链接库

- 按照这个思路，先将所有Lib/下的cpp文件编译为.o文件，并将之编译成动态链接库，省去在向pytorch注册时候的重复编译，将Makefile改写如下

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

- 通过Makefile文件可知，会将所有Lib/下的文件打包成为libqpu.so动态链接库

- 添加动态链接库到系统库

  ```bash
  vim /etc/ld.so.conf
  #在该文件新增一行，为libqpu.so的路径
  /root/embedded_project/
  #:wq保存退出
  ldconfig#刷新动态库路径
  ```

#### 编写调用GPU进行并行计算的C++代码

- 编写流水线式的多qpu并行运算的矩阵c++运算程序

  - 该程序主要实现了等大小的矩阵点乘、点加，并将结果返回

  ```c++
  //"dot.cpp"
  #include <torch/extension.h>
  #include <vector>
  #include <QPULib.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <sys/time.h>
  const int NQPUS = 4; //调用的qpu数
  
  void dotproduct(Int n, Ptr<Float> x, Ptr<Float> y)
  {
      Int inc = numQPUs() << 4;
      Ptr<Float> p = x + index() + (me() << 4);
      Ptr<Float> q = y + index() + (me() << 4);
      gather(p); gather(q);
  
      Float xOld, yOld;
      For (Int i = 0, i < n, i = i+inc)
          gather(p+inc); gather(q+inc);//获取下次运算需要的数据
          receive(xOld); receive(yOld);//计算之前拿到的数据
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
      int width_16 = width + (16 - width % 16);//将矩阵长度转换成16的倍数
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
              mapA[i] = 0;//不足的补零
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

  

- 编写二维矩阵（H x W）乘一维矩阵（W x 1）的c++程序，返回一个（H * 1）的矩阵

  - 此程序可用于计算深度学习中的每个案例的score

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
  
          // 指定存放该行矩阵与权重矩阵计算得到的结果的位置
          // 每行结果存放至16位上
          Ptr<Float>
          p = mapOut + y * 16;
  
      // 初始化Cursor类
      row.init(map);
      row.prime();
      cursorofweight.init(weight);
      cursorofweight.prime();
  
      // 计算该行结果
      Float accumulate = 0;
      For(Int x = 0, x < width, x = x + 16)
          // 每次迭代，都计算本次得到本次结果，并存储，同时取下次计算所需的数据
          row.advance();
     		cursorofweight.advance();
      	accumulate = accumulate + row.current * cursorofweight.current;
  
      End
      // 存储计算结果到p上
      store(accumulate, p);
      // 释放该Cursor
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
      // 创建qpu与cpu交互的矢量
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


### 3. 向pytorch注册c++程序

#### [pytorch官方文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)

- 根据pytorch官方文档，向pytorch注册c++的代码需要两个文件: .cpp文件和setup.py文件

#### 结合QPULib注册c++端程序

- setup.py 文件需要根据QPULib的编译参数、依赖路径进行修改

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

- 所有需要注册的文件建构如下

  ```bash
  embedded_project/
  	dot.cpp
  	setup.py
  	libqpu.so
  ```

- 在该文件夹下执行

  ```bash
  python3 setup.py install
  ```

- 成功编译之后，import模块进行测试

  ```python
  import torch
  import dot_cpp
  ```
  
- 未报错即可上板子验证

### 4. 使用raspberry pi，nfs挂载docker image中的file system，验证上述是否成功

#### 环境准备

- 在实验室主机上安装docker，下载docker镜像，并且使用docker cp指令，将整个file system拷贝出来

  ```bash
  sudo apt update && sudo apt install qemu qemu-user-static binfmt-support
  #这句话是为了能够在x64的处理器上模拟arm处理器
  docker pull registry.cn-beijing.aliyuncs.com/ja1zhou/myrasp:torch
  #可以使用named volume共享container和host之间的文件
  docker volume create --driver local -o o=bind -o type=none -o device="/home/jay/rasp_docker" rasp_volume
  docker run -it --name rasp -v rasp_volume:/root registry.cn-beijing.aliyuncs.com/ja1zhou/myrasp:torch
  #使用docker cp将filesystem拷贝出来
  sudo docker cp rasp:/ ~/home/jay/rasp_docker_filesystem

- 通过nfs启动之后，将/home/jay/rasp_docker挂载，并chroot

  ```bash
  mount 192.168.0.101:/home/jay/rasp_docker_filesystem /mnt -o nolock
  cd /mnt
  mount --rbind /dev dev/#这句话是将原filesystem的GPU device挂载到chroot的路径下
  chroot .
  ```

- (可选)通过iptables代理所有的流量

  ```bash
  #注，因为在uboot启动的时候，我将树莓派的网关设置为了主机的内网地址，所以只需要设置主机的iptables即可
  #下面的语句在host中输入
  iptables -t nat -A POSTROUTING -s192.168.0.1/255.255.255.0 -j SNAT --to public_ip_of_host
  ```
  

#### 正式验证

- 验证qpu计算的准确性

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

  - 结果如下：

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

  

- 验证qpu计算速度

- 此处我们改写了c++端的算子，使得计算的时间以一定的方式返回python端。同时，编写了c++端cpu的算子进行比较

- 此处进行比较的时间仅仅是在c++端进行计算所用的时间。实际上在python端调用gpu算子的时候，会在调用过程、cpu与gpu通信过程中消耗一定的时间

- 此项目展示了在c++端集成、编写算子所能带来的性能提升潜力

  ```python
  # time.py
  # 此时间只计算了各自在计算该矩阵乘法时所耗费的时间
  # 并通过写入到返回值中来获得
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

  - 结果如下：

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

## 写在最后

本项目为《嵌入式系统》课程项目，结果较为初步。

感谢同组的组员lxk同学和组员cxf同学~

感谢杨老师和吕老师给予的指导和帮助~

