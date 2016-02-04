---
layout: post
title: "hts parallel"
date: 2016-02-04
categories: hts
---
### hts并行化 -p 选项 
参考：http://hts.sp.nitech.ac.jp/hts-users/spool/2015/msg00053.html

#### 生成acc文件

将train.scp分割：  
split -l 20 -d train.scp train.scp.part

对各个文件，
~/speech/marytts/lib/external/bin/HERest -A -C configs/qst001/ver3/trn.cnf -D -T 1 -S data/scp/train.scp.partNUM -I data/labels/mono.mlf -m 1 -u tmvwdmv -w 5000 -t 1500 100 5000 -p NUMBER -H  models/qst001/ver3/cmp/monophone.mmf -N models/qst001/ver3/dur/monophone.mmf -M models/qst001/ver3/cmp -R models/qst001/ver3/dur data/lists/mono.list data/lists/mono.list

#### 手写命令合并 
~/speech/marytts/lib/external/bin/HERest -p 0 -A -C configs/qst001/ver3/trn.cnf -D -T 1   -m 1 -u tmvwdmv -w 5000 -t 1500 100 5000  -H  models/qst001/ver3/cmp/monophone.mmf -N models/qst001/ver3/dur/monophone.mmf -M models/qst001/ver3/cmp -R models/qst001/ver3/dur data/lists/mono.list data/lists/mono.list models/qst001/ver3/cmp/HER1.hmm.acc models/qst001/ver3/cmp/HER1.dur.acc models/qst001/ver3/cmp/HER2.hmm.acc models/qst001/ver3/cmp/HER2.dur.acc models/qst001/ver3/cmp/HER3.hmm.acc models/qst001/ver3/cmp/HER3.dur.acc models/qst001/ver3/cmp/HER4.hmm.acc models/qst001/ver3/cmp/HER4.dur.acc models/qst001/ver3/cmp/HER5.hmm.acc models/qst001/ver3/cmp/HER5.dur.acc models/qst001/ver3/cmp/HER6.hmm.acc models/qst001/ver3/cmp/HER6.dur.acc

#### 直接acc合并 
~/speech/marytts/lib/external/bin/HERest -A -C configs/qst001/ver3/trn.cnf -D -T 1  -I data/labels/mono.mlf -m 1 -u tmvwdmv -w 5000 -t 1500 100 5000 -p 0 -H  models/qst001/ver3/cmp/monophone.mmf -N models/qst001/ver3/dur/monophone.mmf -M models/qst001/ver3/cmp -R models/qst001/ver3/dur data/lists/mono.list data/lists/mono.list models/qst001/ver3/cmp/*.acc
> 对应的shell命令为：
/Users/sooda/speech/marytts/lib/external/bin/HERest -A -C configs/qst001/ver3/trn.cnf -D -T 1 -I data/labels/mono.mlf -m 1 -u tmvwdmv -w 5000 -t 1500 100 5000 -p 0 -H models/qst001/ver3/cmp/monophone.mmf -N models/qst001/ver3/dur/monophone.mmf -M models/qst001/ver3/cmp -R models/qst001/ver3/dur data/lists/mono.list data/lists/mono.list models/qst001/ver3/cmp/HER1.dur.acc models/qst001/ver3/cmp/HER1.hmm.acc models/qst001/ver3/cmp/HER2.dur.acc models/qst001/ver3/cmp/HER2.hmm.acc models/qst001/ver3/cmp/HER3.dur.acc models/qst001/ver3/cmp/HER3.hmm.acc models/qst001/ver3/cmp/HER4.dur.acc models/qst001/ver3/cmp/HER4.hmm.acc models/qst001/ver3/cmp/HER5.dur.acc models/qst001/ver3/cmp/HER5.hmm.acc models/qst001/ver3/cmp/HER6.dur.acc models/qst001/ver3/cmp/HER6.hmm.acc
会出错。

![image](../assets/hts_parallel/acc_error.png)

#### 修改HERest源码：  
比较*.acc的命令和手写命令，发`现问题在于dur.acc和hmm.acc顺序相反了。`

![image](../assets/hts_parallel/herest_bugfix.png)  
搞定！


### 来自google的hts并行化
基于hts2.3beta。 测试在hts2.3上时候可以直接使用。若不行，则用hts2.3beta测试可行性。 联系作者或者自己手动搬代码。`ps： 已经被集成在hts2.3了`  
[原文](http://hts.sp.nitech.ac.jp/hts-users/spool/2015/msg00127.html)  
[注意](http://hts.sp.nitech.ac.jp/hts-users/spool/2016/msg00002.html)


### 其他
demo下载地址  
http://hts.sp.nitech.ac.jp/archives/2.2/HTS-demo_CMU-ARCTIC-SLT.tar.bz2  
将上面url的2.2改成2.3即可下载2.3的声音demo
