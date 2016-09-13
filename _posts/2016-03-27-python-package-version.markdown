---
layout: post
title: "python package version"
date: 2016-03-27
categories: code python
tags: ml python
---

* content
{:toc}

今天在测试flappy bird代码时候，一直报numpy没有stack函数。通过参看文档，发现stack是numpy 1.10新增的函数。在之前的版本没有。通过pip将numpy更新到最新版本，还是不行。



参看numpy版本方法：

```python
import numpy
numpy.version.version
```

输出1.8.2。 就是说，系统中还存在其他版本的numpy。怀疑是通过apt-get 安装的。所以尝试sudo apt-get remove python-numpy。但是这样又会把python-pygame一起卸载。简单搜索了一下，发现从源码安装pygame不是太容易。暂时放弃这条路。

通过检索发现：
apt-get 安装的python包路径：/usr/lib/python2.7/dist-packages
pip install 安装的python包： /usr/local/lib/python2.7/dist-packages

查看上面两个路径下的numpy的版本分别是1.8.2和1.10.x

那么，就把解决问题的思路，放在：通过设置sys.path，将pip安装的python包路径放在apt-get之前: sys.path.insert(1, '/usr/local/lib/python2.7/dist-packages')

在文件头就作以上设置。如：

```python
import cv2
import sys
sys.path.insert(1, '/usr/local/lib/python2.7/dist-packages')
import numpy
print np.version.version
```

尝试上面的配置依然失败，输出1.8.2。 有点怀疑文件中的导入机制和终端导入机制不同。考虑使用更加彻底的修改路径方法。

这时候忽然想起，cv2对numpy有依赖。

```python
import sys
sys.path.insert(1, '/usr/local/lib/python2.7/dist-packages')
import numpy
import cv2
print np.version.version
```

是否在import cv2时候已经导入numpy。测试以上代码，顺利输出1.10.x

问题解决。

总结：解决问题时候，应该尽量排除干扰。不要妄加猜测而不作实验。比如测试这个路径机制，直接在flappy bird中的某个py文件中，设置这个路径，可以会因为别的文件现导入numpy，导致失败。应该写一个简单的小程序测试，连cv2这种多余的import都不要有，否则，很有可能就栽在上面了。
