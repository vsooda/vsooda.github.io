---
layout: post
title:  " github page Jekyll"
date:   2015-12-15 
categories: code
---

官方安装文档：
[github-page](https://help.github.com/articles/using-jekyll-with-pages/)

###安装ruby
先参看ruby版本。 ruby --ersion。必须大于2.0. 否则到官网下载源码包，configure， make ， make install 搞定。
值得一提的是：国内安装包速度很慢，可以使用淘宝的源：https://ruby.taobao.org/

```
gem sources --add https://ruby.taobao.org/ --remove https://rubygems.org/
gem install rails

```

###安装bundler
`gem install bundler`

 秒装。同样可以修改bundler，使用淘宝源（未验证，不修改可行）：bundle config mirror.https://rubygems.org https://ruby.taobao.org

### jekyll 安装
创建一个Gemfile， 内容为：

```
source "https://ruby.taobao.org/"
gem 'github-pages'
```
然后直接`bundle install` 搞定。

### github page 生成
初始化page，使用github管理，查看文章开头的网址。

### 启动命令， 本地调试。 
`bundle exec jekyll serve --watch `

然后就可以在localhost:4000查看效果了。

ps： ubuntu下使用retext， mac 下使用mou

