# oqmrc_2018
AI Challenger 2018 阅读理解赛道代码分享

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/numpy-1.14.3-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.15.3)
[![](https://img.shields.io/badge/pandas-0.23.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.23.4)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.python.org/pypi/jieba/0.39)
[![](https://img.shields.io/badge/gensim-3.4.0-brightgreen.svg)](https://pypi.python.org/pypi/gensim/3.6.0)
[![](https://img.shields.io/badge/tensorflow-1.8.0-brightgreen.svg)](https://pypi.python.org/pypi/tensorflow-gpu/1.8.0)
[![](https://img.shields.io/badge/keras-2.2.0-brightgreen.svg)](https://pypi.python.org/pypi/keras/2.2.0)

## **比赛简介**
本次竞赛将重点针对阅读理解中较为复杂的，需要利用整篇文章中多个句子的信息进行综合才能得到正确答案的观点型问题开展评测。本次竞赛将利用准确率进行评分，作为主要评价指标。<br>
比赛数据已上传百度网盘，<https://pan.baidu.com/s/1iMnJNuylLVB4gKMGPUMT4A><br>

# **说明**
有同学指出方法引用和部分模块名称错误的问题。因为当初执行一步会保存中间数据，后面的代码是基于中间数据再做的，整个训练代码一起跑就会有方法和变量上的错误。我尽可能的修改了一些错误（方法名称错误、保留词语数量有错误、训练和测试的代码搞反了等等），通过比赛也认识到的工程能力不足，以后会继续努力哈~

## 比赛任务
根据已有的训练数据，即：<br>
知识：人有人言，兽有兽语，动物是不会听懂人说话的<br>
问题：老鼠听得懂人话吗<br>
选项：听不懂|听得懂|无法确定<br>
答案：听不懂<br>
<br>
训练一个模型，能够根据知识、问题和选项，给出正确的答案。<br>
知识：2018上海高考本科批录取查询时间：7月15日-30日。<br>
问题：2018高考查询录取结果7月15日能查到吗<br>
选项：能|不能|无法确定<br>
答案：？<br>

## **分析思路**
限于游戏本的硬件，本次比赛使用的模型非常简单，预训练词向量 + CNN/RNN + simple-attention + sigmoid<br>
模型融合（25个模型投票）在testa可以到0.745，testb理论上也可以到。因为决赛限制空间，testb成绩稍低一些。<br>

![](https://github.com/renjunxiang/oqmrc_2018/blob/master/picture/net.png)<br>

预训练词向量 | Model | Attention | Accuracy
---------- | -------- | --------- | ---------
无 | CNN | 无 | 0.685
无 | CNN | 有 | 0.695
无 | RNN | 无 | 0.690
无 | RNN | 有 | 0.695
有 | CNN | 无 | 0.705
有 | CNN | 有 | 0.710
有 | RNN | 无 | 0.715
有 | RNN | 有 | 0.725


## **训练代码说明**
决赛限制服务器使用时间和代码运行时间，只上传词语颗粒度的预处理，competition是方法模块<br>
一开始我是将选项拆开训练（成绩0.710左右），后来按照正负面训练（成绩0.715左右），最后才是共享参数的方式同时训练三个选项（成绩0.735左右）。预处理数据、正负面转换等并不是一次性做好的，每个步骤都是一个脚本，请自行检查修改每个步骤的保存路径。

1.**预处理**<br>
word_num = 80000 保留字词数量<br>
dynamic = 1 是否对低频词和新词用统一编码，0/1<br>
char_level = True 是否分词，True就是按词处理<br>
maxlen = [450, 22, 4] passage、quary、alternative保留长度<br>
自行修改参数，和文件夹命名要一致，预处理结果都在data_transform_0(dynamic = 0)和data_transform_1(dynamic = 1)中。<br>
运行data_cut_word.py，按词语做预处理，手动修改里面的参数，得到不同的预处理数据，不同的预处理方式结果差距比较大，对于后续投票有很大影响。<br>

2.**标签获取（可以不运行）**<br>
运行data_label.py，自行修改保存路径，得到alternative的正负面转换，用于模型评估，存在label/clf中。分析结果来看，不能识别为正负面的样本有8%，这部分预测准确率只有0.2<br>

3.**获取预训练词向量矩阵**<br>
运行vec2weight.py，自行选择要使用的词向量和保存路径，结果为npy格式的矩阵。词向量来源于<https://github.com/Embedding/Chinese-Word-Vectors>，在此表示感谢！

4.**训练模型**<br>
4.1 我使用了百度百科词向量，修改competition/net_3a/rnn_attention_concat.py中词向量weight_baidubaike.npy的路径<br>
4.2 运行model_word.py，模型、valid预测结果、test预测结果，保存在model/3a<br>
4.3 选择模型准确率高的预测结果，整理好文件名称，放入model/3a/好的模型/验证集，model/3a/好的模型/测试集<br>
4.4 手动运行model/3a/ensemble.py，逐个增减预测结果，看融合效果。评估结果见 模型评估.xlsx<br>
4.5 选择投票结果不一致的数据（选择3个模型，，一致比例为78%，准确率0.79；不一致比例22%，准确率0.49）<br>
手动运行model/3a/regular.py，修改negative_true（否定词，较大概率样本是负面的），negative_false（否定词，较大概率把样本误分类为负面），具体词语可以逐个去试出较好的修正结果<br>
例如：negative_true = ['不能']，negative_false = ['不一定']（因为这个是需要去看testa数据的，并不符合训练过程中test数据隔离，决赛没有使用）<br>
4.6 选择最终参与投票的模型和否定词<br>

## **测试代码**
决赛服务器直接运行start.sh即可<br>
1.input文件夹放入测试数据data<br>
2.手动修改rnn_attention_concat.py词向量中weight_baidubaike.npy的路径<br>
3.修改预处理过程中train_tokenizer.pkl的位置，运行data_cut_word.py预处理testb<br>
4.训练过程中是一边训练一遍输出testa的结果，手动运行模型融合的；决赛环境需要把不同模型的预测结果保留好，在model_word.py中直接融合<br>
5.output文件夹输出最终结果<br>

## **期待准确率0.8的大神能分享代码！**


