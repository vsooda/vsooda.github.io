---
layout: post
title: "python vim jedi"
date: 2017-12-26
mathjax: true
categories: tools
tags: vim python
---
* content
{:toc}

熟悉vim jedi的人，每次要跳转的时候，碰到下面这句报错是不是很烦。明明安装了啊。之前也以为是jedi或者jedi-vim的问题，重新安装了几遍未果。甚至也怀疑过python版本问题。
```
Please install Jedi if you want to use jedi-vim
```

今天抽空解决了这个问题。下面先从重新编译安装vim开始讲起，再讲mac下需要注意的事项。


### vim 添加python支持

为了添加python支持，重新编译vim

具体流程如下：

1. 在vim官网下载源码
2. 卸载apt-get remove vim
3. ./configure --with-features=huge --enable-pythoninterp --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu
4. 若出现--with-tlib错误。则安装sudo apt-get install libncurses5-dev
5. sudo make install
6. vim --version 应该已经支持python了。若找不到vim， 则更新缓存sudo ldconfig

这样编译之后可以使用系统缓冲区进行复制黏贴了。使用reg查看。"+y     "+p

这是之前在ubuntu上为了支持python跳转上，重新编译vim。测试没有问题。**但在mac上就无法成功**。

### mac问题

直接`:python import jedi`会报weakptr的错误。一直以为是python2.7.14的问题。等待官方修复。

在vim里面python不能跳转真是各种不方便啊。有时候看着看着代码，还得打开macvim去读。（macvim jedi没有问题）

尝试了各种错误版本，网上给出的各种方案大多不靠谱。最后搞定。

#### 错误版本1

./configure --enable-pythoninterp --with-features=huge  --with-python-config-dir=/usr/local/lib/python2.7/config/

这个版本一直不行，是因为config路径实际上是错的。在brew install的python根本没有在这个地方创建config链接

#### 错误版本2

./configure --enable-pythoninterp=dynamic --with-features=huge  --with-python-config-dir=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config

这个版本不行，可能是dynamic不是必要的？

#### 错误版本3

./configure  -enable-pythoninterp=dynamic  --with-python-config-dir=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config

enable前面应该是两个-

#### 错误版本4

./configure  --enable-pythoninterp=dynamic  --with-python-config-dir=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config

dynamic是错误的

#### 错误版本5

设置各种PYTHONPATH，实际上与这个没有关系。


在vim里面输入`:python import sys; print sys.version`, 如果显示的是apple自带的2.7.10则说明vim使用了错误的版本，若显示是2.7.14则是我们安装的版本。


#### 正确版本

最后完整命令：

```
./configure --with-features=huge  --enable-pythoninterp  --with-python-config-dir=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config
```


#### attention
ps:


期间还包含在/usr/local/bin创建了一个软连接。不确定是否生效。

```
ln -s python2-config python-config
```

有可能需要删除vim/src/auto/config.cache
