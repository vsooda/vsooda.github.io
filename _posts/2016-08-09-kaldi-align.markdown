---
layout: post
title: "kaldi对齐"
date: 2016-08-09
categories: speech
tags: kaldi
---

* content
{:toc}




align 需要 feats.scp text mono/final.mdl oov.txt

ali-to-phones
将字级别的对齐转化为音素级别

ali-to-hmmstate
每个音素状态级别的对齐


先准备dict数据。再prepare_lang生成lang数据

查看alignment：

```
show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4
```

之所以还使用老版本的脚本，是因为新版本将ali-to-phones写到steps/nnet/train.sh。没办法获取到音素级别的对齐信息。而老版本的steps/train_nnet_basic.h可以先在外面对齐，获取对齐信息后，再训练网络。
新版本的调用关系是：run.sh -> local/nnet/run_dnn.sh -> steps/nnet/train.sh

`update:2016.08.09: 使用老版本应该不是因为这个原因。想要帧级别的对齐关系，实际上只需要最终的mdl模型就可以了。是否集成在内部没有关系。`

idlak中，音素，状态，词级别的对齐如下：

```
ali=$expa/quin_ali_$step
# Extract phone alignment
ali-to-phones --per-frame $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:- \
| utils/int2sym.pl -f 2- $lang/phones.txt > $ali/phones.txt
# Extract state alignment
ali-to-hmmstate $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:$ali/states.tra
# Extract word alignment
linear-to-nbest ark:"gunzip -c $ali/ali.*.gz|" \
ark:"utils/sym2int.pl --map-oov 1669 -f 2- $lang/words.txt < data/$step/text |" '' '' ark:- \
| lattice-align-words $lang/phones/word_boundary.int $ali/final.mdl ark:- ark:- \
| nbest-to-ctm --frame-shift=$FRAMESHIFT --precision=3 ark:- - \
| utils/int2sym.pl -f 5 $lang/words.txt > $ali/wrdalign.dat
```

### ali 裸数据如下
```
hs_zh_arctic_lbx_00001 3 12 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 18 17 17 17 17 17 17 17 652 651 651 651       660 664 668 672 1022 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1021 1024 1026 1028 1027 1027 1027 1027 1027 1027 1027 1027 1027 1027 1027      1027 1027 1032 1031 1031 1031 1031 876 875 875 875 875 878 877 877 877 877 877 877 877 877 880 879 879 879 884 883 883 883 886 885 885 885 885 885 594 593 593 596 598 597 597 597 597 597 597 597 597        597 597 597 597 597 606 605 605 612 611 1142 1150 1149 1149 1149 1149 1156 1155 1155 1155 1155 1155 1155 1155 1164 1163 1163 1176 1175 1175 1175 116 122 121 121 128 127 127 127 127 127 127 127 127 132      136 135 135 135 135 135 135 135 135 135 135 135 135 135 135 135 135 135 135 188 187 192 191 194 193 193 196 195 195 195 195 195 195 195 198 197 197 197 197 197 197 197 994 993 993 993 993 993 993 993       993 993 996 998 1000 1002 1142 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1141 1150 1149 1149 1149 1156 1155 1155 1155 1155 1155 1155 1155 1155 1155 1164 1163 1163      1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1163 1172 94 104 108 107 112 111 111 111 114 113 113 113 628 627 627 627 627 627 627 630 629 629 629      634 633 633 633 633 636 635 635 635 635 635 635 638 637 637 637 552 554 553 553 553 553 553 553 553 553 553 558 557 557 557 557 557 557 557 557 557 557 557 557 557 562 566 565 565 565 565 565 565 565       565 565 565 565 3 1 1 1 1 1 1 11 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 16      15 15 15 15 15 15 18 17 17 17 17 17 17 17 17 17 17 17 17 17 17 200 199 199 199 199 208 207 212 211 216 222 22 26 25 25 25 25 34 33 33 40 39 39 39 39 39 39 39 44 43 43 43 43 43 43 43 640 639 639 639         642 641 641 644 643 643 643 643 643 643 648 647 647 650 649 649 649 944 943 943 943 943 943 943 943 943 943 943 943 946 950 954 953 953 953 958 1078 1077 1077 1077 1077 1077 1077 1077 1077 1077 1077        1077 1077 1077 1086 1094 1093 1093 1093 1093 1093 1093 1093 1104 1103 1103 1112 1111 1111 1111 1111 306 305 305 305 305 308 310 309 309 309 309 309 309 309 309 318 326 325 325 325 1194 1200 1204 1203       1208 1207 1207 1214 1213 1213 1213 1213 1213 1213 1213 1213 118 124 123 123 123 123 123 123 128 132 131 131 131 131 131 131 131 131 131 131 131 131 131 131 136 135 135 135 202 201 201 201 201 201 201       201 201 201 208 207 207 207 212 211 216 215 215 215 218 226 225 225 232 240 239 239 239 239 239 239 239 248 262 261 261 261 204 203 203 203 203 203 203 203 203 203 203 203 203 203 203 206 205 205 205       205 205 210 214 222 434 436 438 437 437 437 437 437 437 437 437 437 437 437 437 437 437 437 446 445 450 449 449 449 449 4 16 18 858 860 859 859 859 859 859 859 859 859 864 866 868 867 867 867 867 867       867 867 867 867 228 238 246 245 245 245 245 245 245 245 245 245 245 245 245 245 245 245 254 260 259 259 259 259 259 259 259 259 259 259 259 259 259 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 11 10 10 10 10 10       10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 16 15 15 18 17 17 17 17 858 860 859 859 859 859 859 859 859         859 864 866 865 870 614 613 613 613 613 613
```

其实对应的就是每一帧的对应的pdf id是什么。ali-to-phones和ali-to-hmmstate通过索引找到每一帧对应的phone id 和 state id。 idlak中，通过make-fullctx-ali-dnn在生成最终的对齐特征时候，只需要读取原始裸数据即可知道每帧的状态id

####　phones.txt

```
slt_arctic_a0001 sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E dh_B dh_B dh_B dh_B dh_B dh_B dh_B dh_B ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B r_I r_I r_I r_I r_I r_I r_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B ih1_I ih1_I ih1_I ih1_I ih1_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I ah0_I ah0_I ah0_I ah0_I ah0_I p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B t_I t_I t_I t_I t_I t_I t_I t_I t_I t_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E sp sp sp sp sp sp sp sp sp eh2_B eh2_B eh2_B eh2_B eh2_B t_I t_I t_I t_I t_I t_I t_I t_I t_I t_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I t_I t_I t_I t_I t_I t_I t_I t_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp
```

注意这里的音素都是类似于s_B，t_I，p_E这种格式，B表示词的开头，I表示内部，E表示结束。在prepare_lang.sh内部，会生成lang/phones/align_lexicon.txt。 内容如下：

```
ABSTRACTIONS ABSTRACTIONS ae0_B b_I s_I t_I r_I ae1_I k_I sh_I ah0_I n_I z_E
ABSURDITY ABSURDITY ah0_B b_I s_I er1_I d_I ah0_I t_I iy0_E
ABUNDANCE ABUNDANCE ah0_B b_I ah1_I n_I d_I ah0_I n_I s_E
ACCELERATED ACCELERATED ae0_B k_I s_I eh1_I l_I er0_I ey2_I t_I ih0_I d_E
ACCEPT ACCEPT ae0_B k_I s_I eh1_I p_I t_E
ACCEPT ACCEPT ah0_B k_I s_I eh1_I p_I t_E
ACCESSORIES ACCESSORIES ae0_B k_I s_I eh1_I s_I er0_I iy0_I z_E
```

在thchs30的样例中，生成语言模型加入`--position_dependent_phones false`选项，可以避免生成B,I,S后缀的音素。

```
utils/prepare_lang.sh --position_dependent_phones false data/dict "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;
```

#### 状态格式states.tra

```
slt_arctic_a0001 0 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 4 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 3 3 3 3 3 3 3 3 4 4 4 0 0 0 0 1 1 2 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 2 2 2 2 2 2 3 4 0 0 0 0 0 1 2 3 4 4 4 0 0 0 1 2 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 0 1 2 3 3 3 3 4 0 0 0 0 1 2 3 4 0 0 0 1 1 1 1 2 2 2 2 2 3 3 3 4 0 0 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 0 0 1 2 2 3 3 3 3 3 4 4 0 0 0 0 1 2 3 3 3 4 0 0 0 1 1 1 1 1 2 3 4 4 0 0 0 0 0 0 0 1 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 0 0 1 2 3 3 4 0 0 1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 4 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 4 4 4 4 4 4 4 4 0 0 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 4 4 4 4 0 1 2 3 4 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 4 0 1 2 3 4 0 0 0 0 1 1 1 2 2 3 3 4 4 0 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 4 4 4 4 0 0 0 1 1 2 2 3 3 4 0 0 0 1 1 1 2 2 2 2 2 2 2 2 2 3 4 4 4 4 4 4 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 4 4 4 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 3 4 4 4 0 1 1 1 1 1 1 4 4 0 1 2 3 4 0 0 1 1 1 1 2 3 3 4 0 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 1 2 2 3 3 3 4 4 0 0 1 2 3 4 4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 3 3 3 4 0 1 1 2 2 2 2 2 2 2 2 3 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4
```

#### word对齐格式： wrdalign.dat

```
slt_arctic_a0001 1 0.205 0.415 AUTHOR
slt_arctic_a0001 1 0.620 0.160 OF
slt_arctic_a0001 1 0.780 0.080 THE
slt_arctic_a0001 1 0.860 0.365 DANGER
slt_arctic_a0001 1 1.225 0.445 TRAIL
slt_arctic_a0001 1 1.670 0.375 PHILIP
slt_arctic_a0001 1 2.045 0.540 STEELS
slt_arctic_a0001 1 2.630 0.520 ETC
slt_arctic_a0002 1 0.200 0.300 NOT
slt_arctic_a0002 1 0.500 0.095 AT
```
TODO： wrdalign.dat格式是怎么转化的。应该将phone对齐格式转化为这种。那么就跟htk对齐格式非常相近了。`update:phone格式已转化`

### 重新生成allophone及对齐特征

再传入idlak中，对之前预测的文本allophone进行矫正。就是添加空音什么的。

```
python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 1 "2:0.03,3:0.2" "4" $ali/phones.txt $ali/wrdalign.dat data/$step/text_align.xml
```

boost-silence: 在对齐的时候设置。有使用该参数。
>
Normally you shouldn't have to set it. It helps to encourage the silence
model to eat up more of the data, to avoid it getting modeled
inappropriately by context-dependent phones. It is helpful on some setups,
but not always.

`update: 将boost-silence设置为0.25之后，可以避免大部分的误报`

tts训练只有一个speaker，没办法使用并行训练。。设置nj大于1会出错。看在新版本中是否已经被修复。若没有修复，则可以考虑修复之。
