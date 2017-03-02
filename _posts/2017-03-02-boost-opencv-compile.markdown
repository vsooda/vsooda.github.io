---
layout: post
title: "不同编译器版本库混用问题"
date: 2017-03-02
categories: code
tags: compile
---
* content
{:toc}
本文记录编译**superviseddescent**库碰到的问题。问题本质在于使用不同编译器版本编译的库，有时候不能混用。

### 查错记录

刚开始编译在ubuntu16.04下编译不通过，在ubuntu14.04编译通过。

```
在ubuntu16.04 + opencv3.1 + boost1.58下，报boost program_options链接错误。
在ubuntu14.04 + opencv2.4.8 + boost1.54下, 一次通过
```

一度怀疑是boost库链接问题，以为是不同版本boost不一样导致的。下载boost源码，发现没有CMake编译，暂时作罢。。后来使用下面脚本，发现应该不是boost库找不到造成的。

```
function(VerifyVarDefined)
  foreach(lib ${ARGV}) 
    if(DEFINED ${lib})
    else(DEFINED ${lib})
      message(SEND_ERROR "Variable ${lib} is not defined")
    endif(DEFINED ${lib})
  endforeach()
endfunction(VerifyVarDefined)

VerifyVarDefined(Boost_PROGRAM_OPTIONS_LIBRARY)
```

依稀记得以前编译[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), 碰到过boost问题，好像是将gcc切换到5就搞定了的。至此把怀疑目光转向g++版本和opencv版本问题。

设置opencv搜索路径: 

```
set(OpenCV_DIR "/home/sooda/tools/OpenCV/opencv-2.4.10/build")
```

做了以下测试：

```
在ubuntu16.04 + opencv2.4.10(用g++4.9编译) + boost1.58下，gcc4.9编译报boost program_options链接错误。
在ubuntu16.04 + opencv2.4.10(用g++4.9编译) + boost1.58下，切换编译器到g++5, 报opencv各种cv::error, cv::exception, cv::string等错误
```

查找文章怀疑是c++ ABI在不同版本之间不同导致的。也就是说，使用不同版本编译器编译出来的库未必能够通用。

将编译器切换到g++5，重新编译opencv，superviseddescent库顺利编译成功。

猜测ubuntu16.04上的boost1.58应该是用g++5编译的，这也解释了以前不少代码编译不通过，切换编译器之后就编译通过的情况。编译不通过可能不是代码版本不兼容，而是库版本不兼容！



### 更好的解决方案

既然已经知道是由于boost版本库导致的不兼容，是否可以使用g++4.9编译一个版本，然后opencv等其他库都不变呢。这样有利于保证整个系统使用相同的编译器版本。

下载boost最新代码，编译之:

```
sh bootstrap.sh --prefix=/home/sooda/tools/boost/
./b2 install
```

编译完了之后,  记得设置库的搜索路径:

```
export LD_LIBRARY_PATH=/home/sooda/tools/boost/lib:$LD_LIBRARY_PATH
```

需要设置boost的cmake搜索路径。boost是通过FindBoost.cmake来驱动的。查看`/usr/share/cmake-3.5/Modules/FindBoost.cmake`之后, 在CMakeLists.txt中加入以下内容就可以使用自己编译的boost版本。

```
set(Boost_NO_SYSTEM_PATHS TRUE) 
if (Boost_NO_SYSTEM_PATHS)
  set(BOOST_ROOT "/home/sooda/tools/boost/")
  set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
  set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
endif (Boost_NO_SYSTEM_PATHS)
```

这样我们就实现了不切换编译器版本，顺利编译项目的。从这里也可以看出使用，有些项目为了避免系统依赖库混乱，将库编译到项目路径下是有必要的。