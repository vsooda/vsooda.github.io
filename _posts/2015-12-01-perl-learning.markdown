---
layout: post
title:  "Perl学习"
date:   2015-12-01
categories: code
tags: code perl
---
* content
{:toc}

perl 语法知识




- 使用变量要加上$, 这点与php类似。 对整个数组的引用要加上@。 变量和数组对应标量和列表。
- 单引号，双引号都可以用来表示字符串。与shell类似，单引号指标是字面值，双引号会对实际值进行求解，例如变量，“\n"等。
- 使用.拼接字符串。例如："hello"."world"。 print不同部分可以用逗号分隔。printf的格式化输出：printf “hello %s %d ”, $user, $id;
- qw表示将把后面的内容做为字符串列表（单引号）。例如：qw(fred barney betty wilma dino )
- perl数组可以直接赋值。例如： @copy=@orig；同时，与python类似， 交换可以不需要临时变量。($fred, $barney) = ($barney, $fred)
- perl中使用索引操作，效率很低。一般使用push和pop。例如：


  ```perl
  @array=5..9;
  $fred = pop(@array); #$fred 得到 9,@array 现在为(5,6,7,8)
  pop @array; $barney 为 8, @array 现在为(5,6,7)
  pop @array; #@array 现在为(5,6)  
  ```


- foreach遍历列表的所有元素。`foreach $rocks(@rocks)`。 控制变量不是这些元素的拷贝，而是元素的引用。如果在循环中修改了这些值。则原始列表中的元素也会被修改。如果省略了控制变量，则使用默认变量$_
- 列表上下文。一个给定的表达式在不同的上下文环境中其含义是不相同的。例如对于一个数组，在列表context中，返回列表元素。在标量context中，返回数组元素个数。
- 常用函数
chomp： 去掉所有换行符

- 函数定义： sub

  ```perl
  sub foo {
  }
  ```

子程序最后计算的值将被返回。
perl自动将参数列表自动存放在`@_`的数组中。
使用my创建私有变量
使用return可以从函数中返回

- <>从数组@ARGV中得到调用参数。用于提供类似于cat，sed等功能。 <>操作通过查看@ARGV来决定使用哪些文件。如果列表为空，则使用标注输入流。否则使用其找到的文件。在@argv使用之前，还可以修改，强制使用某些文件。

- 对于优先级不确定的建议加括号。比如： print (2+3) * 4; 将输出5。 然后将print的返回值1，乘以4，其结果被丢弃。在程序开发调试阶段可以打开-w选项，遇到上诉问题会发出警告。
- 句柄，一般用于读写文件。

  ```perl
  open CONFIG, '1.txt'
  open CONFIG, '>1.txt' #读出
  open CONFIG, '<1.txt' #写入
  close CONFIG #关闭句柄
  ```

- 哈希表
$table{"foo"} = “bar”;
要引用整个hash，使用%作为前缀。
也可以使用大箭头"=>"来定义哈希表。（这不是php吗？？）例如：

  ```perl
  my %last_name = (
  	“fred” => “flintstone”, “dino” => undef, “barney”=> “rubble”; “betty”=> “rubble”,
  );
  ```

  hash常用操作：

  ```perl
  while (($key, $value) = each %hash){
  	print “$key => $value\n”;
  }

  foreach $key (sort keys %hash){
  	print “$key => $hash{key}\n”;
  }

  if(exists $books{$dino}){
  	print “Hey, there’s a libaray card for dino!\n”;
  }
  ```

- 正则表达式
要匹配某个模式(正则表达式)和`$_`的关系,可以将模式放在正斜线(//)之间， 如：

  ```perl
  $_ = “yabba dabba doo”;
  if(/abba/){
  	print “It matched!\n”;
  }
  ```

  在kaldi，egs/hkust/utils/pinyin_map.pl中

  ```perl
  $initial= $A[$i];
  $final = $A[$i];
  if ($A[$i] =~ /^CH[A-Z0-9]+$/) {
    $initial =~ s:(CH)[A-Z0-9]+:$1:;
    $final =~ s:CH([A-Z0-9]+):$1:;
  }
  ```

使用以上语句对于CHANG1,这种语句提取。提取结果是$initial为CH，$final为ANG1

这是怎么做到的呢？

首先，perl使用 “=~”来表示命中匹配。使用“!~"来表示未命中匹配。

在匹配的过程中，匹配原始字符放在等式左边，最终结果也是存在等式左边。比如说上面这个代码。$initial初始值是CHANG1，匹配后为CH。

perl的正则表达式关键词有：

* m 表示匹配
* s 表示替换
* tr 集合替换

至于$1, $2的匹配提取可以参考以下链接：
java正则表达式

[参考1](http://blog.csdn.net/allwefantasy/article/details/3136570)

[参考2](http://www.runoob.com/java/java-regular-expressions.html)
