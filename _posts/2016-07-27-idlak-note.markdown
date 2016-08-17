---
layout: post
title: "idlak笔记"
date: 2016-07-27
categories: speech
---

### 数据准备　idlak\_make\_lang mode ０
```
idlak-voice-build/utils/idlak_make_lang.py --mode 0 /home/sooda/speech/idlak/egs/tts_dnn_arctic/s1/data/full/text_norm.xml /home/sooda/speech/idlak/egs/tts_dnn_arctic/s1/data/full /home/sooda/speech/idlak/egs/tts_dnn_arctic/s1/data/local/dict1
```

这个步骤的结果，在dict1文件夹下生成以下文件：

>
characters.txt  extra_questions.txt  lexicon.txt  nonsilence_phones.txt  oov.txt  optional_silence.txt  silence_phones.txt

###构造fst语言模型

```
utils/prepare_lang.sh --num-nonsil-states 5 --share-silence-phones true $dict "<OOV>" data/local/lang_tmp $lang
```

同时生成lexiconp.txt. lexiconp.txt是带概率的lexicon.txt

###生成带对齐信息的xml : idlak\_make\_lang mode 1
```
python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 1 "2:0.03,3:0.2" "4" $ali/phones.txt $ali/wrdalign.dat data/$step/text_align.xml
```

对应命令为：

```
python /home/sooda/speech/idlak/idlak-voice-build/utils/idlak_make_lang.py --mode 1 2:0.03,3:0.2 4 exp-align/quin_ali_full/phones.txt exp-align/quin_ali_full/wrdalign.dat data/full/text_align.xml
```

最后一个参数是输出。其他是输入。

对齐结果phone.txt的格式为：

```
slt_arctic_a0001 sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B ao1_B th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I th_I er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B ah1_B v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E v_E dh_B dh_B dh_B dh_B dh_B dh_B dh_B dh_B ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B d_B ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I n_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I jh_I er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E er0_E t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B t_B r_I r_I r_I r_I r_I r_I r_I r_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I ey1_I l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E l_E f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B f_B ih1_I ih1_I ih1_I ih1_I ih1_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I ah0_I ah0_I ah0_I ah0_I ah0_I p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E p_E s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B s_B t_I t_I t_I t_I t_I t_I t_I t_I t_I t_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I iy1_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I l_I z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E z_E sp sp sp sp sp sp sp sp sp sp eh2_B eh2_B eh2_B eh2_B eh2_B t_I t_I t_I t_I t_I t_I t_I t_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I s_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I eh1_I t_I t_I t_I t_I t_I t_I t_I t_I t_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I er0_I ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E ah0_E sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp sp
```

exp-align/quin\_ali\_full/wrdalign.dat给出每个单词的时间对齐信息

格式为：

```
slt_arctic_a0001 1 0.205 0.425 AUTHOR
slt_arctic_a0001 1 0.630 0.150 OF
slt_arctic_a0001 1 0.780 0.080 THE
slt_arctic_a0001 1 0.860 0.370 DANGER
slt_arctic_a0001 1 1.230 0.440 TRAIL
```

load_lablels之后，每个label的格式为：[0.205 0.34 ao_B]

### 生成context label : idlak\_make\_lang mode 2

```
idlaktxp --pretty --tpdb=$tpdb data/$step/text_align.xml data/$step/text_anorm.xml
idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_anorm.xml data/$step/text_afull.xml
python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 data/$step/text_afull.xml data/$step/cex.ark > data/$step/cex_output_dump
```

对应为：

```
idlaktxp --pretty --tpdb=/home/sooda/speech/idlak/idlak-data/en/ga/ data/full/text_align.xml data/full/text_anorm.xml
idlakcex --pretty --cex-arch=default --tpdb=/home/sooda/speech/idlak/idlak-data/en/ga/ data/full/text_anorm.xml data/full/text_afull.xml
python /home/sooda/speech/idlak/idlak-voice-build/utils/idlak_make_lang.py --mode 2 data/full/text_afull.xml data/full/cex.ark > data/full/cex_output_dump
```
