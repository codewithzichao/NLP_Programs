## 前言

本repo旨在通过使用tensorflow2，来实现NLP中常见的任务(文本相似度分析、文本摘要、文本生成、机器翻译等等)，新手NLPer可根据本repo进行学习，从而加深对NLP领域的理解。本repo持续更新中～

> deep learning framework：tensorflow2
>
> python version：3.7
>
> IDE：Anaconda

## Program1：单词相似度(词嵌入)

**项目简介：** 使用imdb_reviews数据集来构建CBOW模型(其实应该更像是简化的fastText)进行文本分类，从而得到所有token的embedding，进而得到与目标单词最相似的若干单词list。

如果对CBOW/fastText模型原理感兴趣的话，可以参看我的博客文章🤩：[NLP|word2vec/GloVe/fastText模型原理详解与实战](https://codewithzichao.github.io/2020/02/29/NLP-word2vec-GloVe-fastText模型原理详解/)

**文本相似度的计算过程：**

* 得到词汇表的embedding后，计算两两token的embedding之间的距离，从而得到一个距离矩阵(距离公式有多种选择，可以使用欧式距离、切比雪夫距离、曼哈顿距离等等)；
* 对于目标单词target_token，对它与其他的token的距离安从小到大排序，选取topK(距离越小的说明越相似)。

**运行结果：** 

与beautiful最相似的8个单词，如下：(从结果来看，并不是很理想)

> target word:['beautiful'] -> top8 words:['beautiful', 'Carre', 'extraordinar', 'surprise', 'lonel', 'delight', 'enjoyable', 'sensitive']

此外，关于做文本相似度分析，非常常用的工具就是gensim。这个大家也可以去尝试～

## Program2：基于LSTM的文本分类





## Program3：文本生成





## Program4：基于seq2seq的机器翻译







## Program5：基于Transformer的机器翻译



**Transformer模型原理讲解：**关于Transformer模型的原理，可以参看我的博客文章🤩：[深入探究Transformer模型](https://codewithzichao.github.io/2020/02/17/NLP｜深入探究Transformer模型/)







持续更新中......