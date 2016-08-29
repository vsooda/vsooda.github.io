---
layout: post
title:  "phonesets"
date:   2015-09-30 20:14:34 +0800
categories: marytts
tags: phonest
---
* content
{:toc}




## **元音vowel**的域有：(韵母)

### vlng 发音长度
s : 短 short  
l : 长 long  
d : 双元音 diphthong  
a : ? 非重读音节 schwa

### vheight 音高
1 : 高  
2 : 中  
3 : 低

### vfront 发音部位
1 : 前  
2 : 中  
3 : 后

### vrnd  ？
“+” : on  
"-" : off  

## **辅音consonant**的域有：（声母）

### ctype  ：鼻音，浊辅音，爆破音等
s : 破裂音   Plosive   
f : 摩擦音   Fricative   
a : 破擦音   Affricate   
n : 鼻音     Nasal  
l : 清音     Liquid  
r : 滑音     Glide

### cplace :发音部位
l : 唇音  
a : 齿槽音 alveolar   
p : 上颚音 palatal  
b : ？唇缝音 labio  
d : 齿音 dental  
v : 软颚音 velar  
g : ?

### cvox   : 是否浊音
"+" : on
“-” : off


<!--
------------------------
## 细节


* voiced(浊音)：
元音或者cvox域为'+'

* Diphthong（复合元音）：
vlng域为d

* Syllabic（音节主音):
元音

* Sonorant (响音)：
ctype 包含： l n r

* Liquid（清音）：
cypte： l


* Nasal（鼻音）：
ctype： n

* Glide（滑音）：
ctype： r 且 不是元音

* Fricative（摩擦音）：
ctype: f

* Plosive（破裂音）：
ctype： s

* Affricate（破擦音）：
ctype： a


* tone(语气，音调)：
isTone： +

* sonority(响亮程度？发音长度？)：  
如果是元音则读取vlng域：
	ld： 6
	s ： 5
	a ： 4
	其他： 5  
如果是响音：
	3  
如果是摩擦音：
	2
其他 1

* Pause（停顿）：
vc为0 或者tone为‘-‘

![phonset](phoneset.png)

----------------------
## 其他

内部标记：
vc：
vowel ： +
consonant： -

-->
