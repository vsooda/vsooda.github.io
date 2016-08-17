---
layout: post
title: "git cherry pick"
date: 2016-08-17
categories: code
---

考虑一个问题： 使用git做版本管理。如何在保持自己分支代码与主分支一致尽量一致的情况下，随时将可应用与主分支的bug修复提交的主分支，而其余feature代码则保留在自己分支上。

以`kaldi`为例子。原始仓库是`kaldi-asr/kaldi`， fork到`vsooda/kaldi`。然后在`vsooda/kaldi`下开分支`tts`作为功能开发。`master`分支作为与原来仓库的同步作用。在`tts`整个分支成熟之前，肯定是不能提交到`master`分支的。

从`kaldi-asr/kaldi`及时更新代码，只需要将其设置为`upstream`，再及时同步即可。

```
git remote add upstream [https://github.com/kaldi-asr/kaldi.git]
git fetch upstream
git checkout master
git merge upstream/master
```

现在考虑另一个问题，在`tts`开发时候，发现了一些小bug。修复之后怎么提交给原仓库呢。直接提交到`tts`分支，似乎很难？那么常用的做法就是stash，切到`master`分支，再修改重新编辑一遍，测试（这时候如果主分支没有测试环境就蛋疼了），提交，pull request。

另一种做法是：

直接提交到`tts`分支，测试，stash，cherry-pick，再测试，提交，pull request。

具体案例：

```
git add ../../wsj/s5/utils/subset_data_dir_tr_cv.sh  #bug修复
git commit -m 'fix the bug occur when only have one speaker'
git stash #贮存本地其余更改
git checkout master
git cherry-pick c338e7b95f75677407fcfb4e281167056bb52cf4 #commit id
git push
git checkout tts #切回分支
git stash pop #取回贮存
```
