---
layout: post
title: "SGE NFS kaldi 计算集群环境搭建"
date: 2016-02-01
categories: sge
---
主要参考：[kaldi-parallel](http://kaldi-asr.org/doc/queue.html)

配置清单：

```
机器：

s203 192.168.11.203  
work 192.168.11.90

账户
research

s203作为nfs服务器，并作为sge master。同时也作为工作机。
work作为工作机
```

##nfs

###服务端：
sudo apt-get install nfs-kernel-server

vi /etc/export

```
/home/research/speech 192.168.11.0/24(rw,sync,all_squash,no_subtree_check)
```

更改后，运行sudo exportfs -rv使其生效

> 选项可以更改

sudo service nfs-kernel-server restart

### 客户端
sudo apt-get install nfs-common

mount:
sudo mkdir /mnt/nfs  
sudo mount 192.168.11.203:/home/research/speech /mnt/nfs

umount:
sudo umount -f -l /mnt/nfs



##gridengine


###master：
sudo apt-get install gridengine-master gridengine-client

这个文件/var/lib/gridengine/default/common/act_qmaster保存了主节点hostname

error: commlib error: access denied (client IP resolved to host name  
这种问题是因为/etc/hosts没有和上面这个文件的hostname一样的条目。  
可以增加hosts条目，也可以更改上面这个文件。  
更改完了知乎需要sudo service gridengine-master restart


###worker：

sudo apt-get install gridengine-client gridengine-exec

可能的问题：  

1. error: can't resolve host name  
解决方法： 将master的hostname条目增加到/etc/hosts

2. commlib error: can't bind socket (no additional information available)  
解决方法：  
在master主机增加hosts。  
sudo -ah “work_hostname"  
master重启： sudo service gridengine-master restart  
work重启： sudo service gridengine-exec restart

###检查安装情况： 
运行qstat和qhost -q

## nfs权限问题踩坑记
碰到的最大的坑是nfs权限问题。 表现形式：
![image](http://vsooda.github.io/assets/sge/path_error.png)

刚开始完全不知道是什么问题。图中红框部分都怀疑了一个遍。

刚开始客户机使用的sooda账户，发现报错后怀疑是work机器缺少可执行文件，在work上创建research账户， 并在research账户也安装了一个kaldi。。

安装后还是提示这个错误。。(但是实际上research也没有白安装，用于sge了）。 反思觉得不应该是在单个机器的某个目录上运行，于是尝试在/mnt/nfs/下运行。报错还是找不到文件。google搜索文章提到nfs权限问题。就开始怀疑。肩擦/etc/export文件需要加上rw选项。尝试了还是不行。但是有了这个意思之后就开始怀疑UID，GID问题了，因为在印象中linux控制权限使用的就是这两个ID。查看/mnt/nfs下文件详情：  
![image](http://vsooda.github.io/assets/sge/uid_gid.png)

在research下的文件居然用户名UID，GID是sooda（对应ID是1000）。而在s203同样的目录下，uid，gid是research（对应的ID同样是1000）。在work主机上， 表现的症状是：以sooda用户登录,打开/mnt/nfs下的文件是可读，可写。但是用research账户登录，很多文件只有可读权限。尝试将research加入到sooda所在的组，还是不行。`怀疑在/mnt/nfs下访问，使用的uid，gid在nfs的影响下，并不是其实际UID，GID`。在research@s203下，将目录的权限设置为777. 这下research@work也能正常读写了。但是！使用kaldi训练新生成的文件，还是存在权限问题！

开始考虑用户映射的方案。google，未成功。

为了印证nfs权限问题，考虑将work节点从队列all.q中移除。只是用s203训练。成功。至此确认是nfs权限的影响。

查了多篇文献，尝试no_root_squash等选项。比较零散，发现权限还是sooda得，以为没有成功。看了[这篇文章](http://www.cnblogs.com/mchina/archive/2013/01/03/2840040.html), 对用户访问控制有更加系统的了解。发现之前有个误区，一直想要将原本文件的权限变为research@work可以读写的形式，其实没有必要！已存在文件使用chmod更改权限就可以了，只要新生成文件两者都可以访问就可以了！基于这个思路发现了，nobody，nogroup这样特殊的UID,GID。这个选项就是`all_squash`。 分别用在台机器下访问/mnt/nfs，发现确实没有生成的文件都是nobody:nogroup，而且可以一起更改，至此，心中大定。运行kaldi run.sh测试，果然可以。搞定。

![image](http://vsooda.github.io/assets/sge/nobody.png)



