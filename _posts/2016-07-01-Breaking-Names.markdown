---
layout: post
title: "命名大事件"
date: 2016-07-01
categories: code
tags: c++
---
* content
{:toc}

引言：头文件命名与g++头文件冲突真是太悲剧了。


### 事件1
事情的起因是，我在mac上开发的c++跨平台程序（使用CMake构建）在linux上编译不通过。严重影响了跨平台这个出发点。于是就想要解决他。

主要提示的错误是：

1. fatal error: string: No such file or directory  

2. os_defines.h:44:19: error: missing binary operator before token 。提示了几百个错误..

在网上搜索后发现，有方案说，这是由于编译器把#include<string> 当做c语音的string.h了。有人的修复这个问题的bug是将.c的后缀名改成.cpp。然后用g++编译，就可以了。以为是c和c++`混合编译`的问题。测试后发现问题1不出现了，问题2还是存在。

查找问题2的解决方法。用出错信息搜索后发现这个[答案](http://stackoverflow.com/questions/21138340/mismatched-c-header-versions)。提到了编译json遇到的问题。但是第一遍匆匆阅读过去，以为是搜索路径的问题。于是在CMakeList设置了头文件的搜索路径。不成功。

把工程项目一直往后回退，发现在6月13日的那个提交之后，开始编译不过，而那个提交主要是用于处理特征的。

后来，再一次搜到这个答案，仔细一看之下发现，其中的关键词`features.h`，而这个文件在我自己的项目也有！！！`命名冲突`。把features.h改掉之后编译通过。再把之前的.cpp改回.c，编译依然通过。

### 事件2
命名大事件，无独有偶。 之前在用QT写[测试程序](http://blog.csdn.net/vsooda/article/details/9329969)的时候，把工程名命名为`first`。

报错：**-1: error: Circular all <- first dependency dropped. **

貌似qt自己的程序有一个first，你再建一个first就粗事了，会造成循环依赖。
