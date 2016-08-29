---
layout: post
title: "tensorflow note"
date: 2016-03-27
categories: ml code
tags: ml
---

* content
{:toc}

tensorflow基础语法




* 变量 variable
* 常量 constant
* 占位符 placeholder。 具体值在运行时，通过feed_dict给出
* 每个操作op是图的一个节点
* 在session中，进行图计算
* 通过记录summary，可以在tensor board上参看运行信息
* tensorflow已经集成了很多优化方法，可以直接调用
