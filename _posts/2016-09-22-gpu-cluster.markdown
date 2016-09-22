---
layout: post
title: "gpu cluster搭建"
date: 2016-09-22
categories: tools
tags: gpu gtx1080 ubuntu
---
* content
{:toc}

单台4块 gtx1080 显卡， 配置清单，系统安装



## 配置

**注意**：这里之所以使用3卡是因为非公版，显卡较厚，4块插不下。建议买公版，应该可以插下

```
显卡：zotac  GTX 1080 * 3 
cpu：x99 i7 5930k 
主板：华硕 X99-E WS/USB 3.1
内存：DDR4 64G 金士顿 骇客神条 16*4 2400 
HDD：2TB或以上硬盘 希捷(SEAGATE)2TB 7200转64M SATA3 台式机硬盘(ST2000DM001)
SSD：512G 三星(SAMSUNG) 850 EVO 500G SATA3 固态硬盘
电源：振华（SUPER FLOWER） 额定1600W LEADEX T 1600电源 
机箱：Corsair Carbide Air 540  
```

![](http://vsooda.github.io/assets/gpu_cluster/inside.jpg)
![](http://vsooda.github.io/assets/gpu_cluster/front.jpg)



## 系统安装，cuda安装

### 正常流程

```
1. sudo apt-get install mesa-common-dev freeglut3-dev build-essential

2. /etc/modprobe.d/blacklist.conf添加:
blacklist amd76x_edac
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv

更新内核： sudo update-initramfs -u

3. 卸载旧驱动：sudo apt-get remove --purge nvidia*

4. ctrl+alt+f1进入终端界面。 sudo service lightdm stop

5. sudo sh cuda_8.0.27_linux.run --no-opengl-libs --verbose 进行安装一路enter和y即可。

6. 配置环境变量 在/etc/profile添加： 
PATH=/usr/local/cuda-8.0/bin:$PATH
export PATH

执行 source /etc/profile使其生效

7. 添加库搜索路径。在 /etc/ld.so.conf.d/添加文件cuda.conf，内容为: 
/usr/local/cuda-8.0/lib64

执行 sudo ldconfig使其生效

8. 执行nvidia-smi测试是否安装成功
```

### 实际安装

#### ubuntu16.04

**问题1** 按照以上步骤进行安装时，进行到第4步时候，终端界面不显示。后来设置远程登录上去安装orz..

**问题2** 第5部正式安装时候，碰到错误：

```
The driver installation is unable to locate the kernel source. Please make sure that the kernel source packages are installed and set up correctly.
If you know that the kernel source packages are installed and set up correctly, you may pass the location of the kernel source with the '--kernel-source-path' flag.
```

google查到的方案比如：`apt-get install linux-headers-$(uname -r)`, 编译器降级都是无效的。也有说是内核版本错误。故使用ubuntu14.04进行测试。

#### ubuntu14.04

问题依旧存在。这时候开始怀疑是否是因为gtx1080的特殊性导致的。

搜索到[这篇文章](http://textminingonline.com/dive-into-tensorflow-part-iii-gtx-1080-ubuntu16-04-cuda8-0-cudnn5-0-tensorflow), 安装成功。

**其根本问题是cuda_8.0.27_linux.run里面所带的nvidia361驱动根本不支持gtx1080. 所以要先安装一个最新的驱动，而不安装cuda安装包带的显卡驱动。**

大致流程如下：

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
sudo apt-get install mesa-common-dev freeglut3-dev build-essential 
sudo service lightdm stop
sudo sh cuda_8.0.27_linux.run --tmpdir=/opt/temp/ --no-opengl-libs --verbose
注意在提示安装显卡驱动时候，选择N
其他安装环境变量等，与正常流程相同
```

执行nvidia-smi测试：

![](http://vsooda.github.io/assets/gpu_cluster/nvidiasmi.jpg)

**本系统最终在ubuntu14.04安装成功，ubuntu16.04应该也没问题**


