---
layout: post
title: "智能指针用法"
date: 2016-07-05
categories: code
tags: c++
---

* content
{:toc}




```cpp
class Base* pb = new Derive(10);
shared_ptr<Base> pbb(pb);
shared_ptr<Base> pbbb = pbb; 允许
```

```cpp
shared_ptr<Base> pbbb = shared_ptr<Base>(pb); 错误
shared_ptr<Base> pbbb(pb); 错误
```
智能指针不能对同一个对象指针初始化两个智能指针，这样会导致重复释放问题。而智能指针构造之后赋值传递是可以的。
