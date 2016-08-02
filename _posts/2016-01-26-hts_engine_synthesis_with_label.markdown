---
layout: post
title:  "hts engine 合成流程分析 "
date:   2016-01-26
categories: hts
---


#### 合成代码框架
HTS\_Engine\_synthesize\_from\_fn      

* HTS\_Label\_load\_from\_fn
* HTS\_Engine\_synthesize
	* HTS\_Engine\_generate\_state\_sequence
		* HTS_SStreamSet\_create : `parse label and determine state duration`
	* HTS\_Engine\_generate\_parameter\_sequence
	* HTS\_Engine\_generate\_sample\_sequence


#### label细节
<img src="http://vsooda.github.io/assets/hts_train/question.png" width="500">

左边是questions\_qst001.hed 右边是questions\_utt\_qst001.hed  
左边是正常属性的问题集。包括当前音素，下一个音素，当前音素属性，下一个音素属性。。  
右边是gv的问题集合。其属性更多的是句子统计信息。 // sentence\_numwords  sentence\_numphrases  phrase\_numsyls，  phrase\_numwords

代码中， hmmfeaAlias指的是特征编号

replaceToBI将H-%L形式转换为：H-pc|这种形式。 hts训练时候如何识别？  
其逆向转换函数replaceBackToBI只在GvModelSet.java:loadSwitchGvFromFile中使用。而loadSwitchGvFromFile的调用被注释掉。是否有问题？

label内容：

<img src="http://vsooda.github.io/assets/hts_train/label_format.png" width="500">

<img src="http://vsooda.github.io/assets/hts_train/label_string.png" width="500">

在hts代码中，用lstring保存一条记录。start， end保存其实时间。name保存其内容。start, end可选。

<img src="http://vsooda.github.io/assets/hts_train/label_parse.png" width="500">

<!--{% highlight c++ linenos %}-->

```
/* HTS_SStreamSet_create: parse label and determine state duration */
HTS_Boolean HTS_SStreamSet_create(HTS_SStreamSet * sss, HTS_ModelSet * ms, HTS_Label * label, HTS_Boolean phoneme_alignment_flag, double speed, double *duration_iw, double **parameter_iw, double **gv_iw)
```

<!--{% endhighlight %}-->
