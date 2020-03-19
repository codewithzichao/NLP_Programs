## 前言

本repo旨在通过使用tensorflow2，来实现NLP中常见的任务(文本相似度分析、文本摘要、文本生成、机器翻译、序列标注等等)，新手NLPer可根据本repo进行学习，从而加深对NLP领域的理解。本repo持续更新中～

> Deep Learning Framework：tensorflow2
>
> Python version：3.7
>
> IDE：Anaconda
>
> Notes：如果没有足够的GPU的话，可以在Google colab上运行。

## Program1：单词相似度(词嵌入)

**项目简介：** 使用imdb_reviews数据集来构建CBOW模型(其实应该更像是简化的fastText)进行文本分类，从而得到所有token的embedding，进而得到与目标单词最相似的若干单词list。项目代码：[项目代码](https://github.com/codewithzichao/NLP_Programs/tree/master/word_embedding_program)

如果对CBOW/fastText模型原理感兴趣的话，可以参看我的博客文章🤩：[NLP|word2vec/GloVe/fastText模型原理详解与实战](https://codewithzichao.github.io/2020/02/29/NLP-word2vec-GloVe-fastText模型原理详解/)

**文本相似度的计算过程：**

* 得到词汇表的embedding后，计算两两token的embedding之间的距离，从而得到一个距离矩阵(距离公式有多种选择，可以使用欧式距离、切比雪夫距离、曼哈顿距离等等)；
* 对于目标单词target_token，对它与其他的token的距离按从小到大排序，选取topK(距离越小的说明越相似)。

**运行结果：** 

与beautiful最相似的8个单词，如下：(从结果来看，并不是很理想)

> target word:['beautiful'] -> top8 words:['beautiful', 'Carre', 'extraordinar', 'surprise', 'lonel', 'delight', 'enjoyable', 'sensitive']

此外，关于做文本相似度分析，非常常用的工具就是gensim。这个大家也可以去尝试～

## Program2：基于LSTM的文本分类

**项目简介：**  使用imdb_reviews数据集，构建两层的biLSTM的二分类的文本分类模型。项目代码：[项目代码]()

**LSTM原理详解**： 关于LSTM的原理，可以参看我的博客文章🤩：[NLP|LSTM与GRU](https://codewithzichao.github.io/2020/02/17/NLP｜LSTM与GRU/)

**运行结果：**

```latex
0:this movie is bad. but the actor is very handsome and I like him. but I will not recommend this movie.
the result is [[0.16748577]].
1:The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.
the result is [[0.32160088]].
2:actually, I am the actor's fans. But his performance in the movie break my heart.
the result is [[0.99016374]].
3:The characters is not famous, but their performances make the movie reach a very high level! 
the result is [[0.06296671]].
4:The movie is very ironic.This film criticizes the social phenomena without conscience
the result is [[0.995698]].
```

## Program3：文本生成

**项目简介：** 将莎士比亚文章作为训练集，建立基于char-level的RNN模型，从而生成莎士比亚风格的文本。注意：我们建立的是一个语言模型，并没有使用seq2seq模型，后期会尝试使用seq2seq模型来做文本生成任务。

**运行结果：** （可能需要调整网络架构以及epochs，我只跑了10epochs。😭）第一个输入为：`ROMEO`。

```latex
ROMEO: QI-qG'G!
ZNmFzPxbiZM!'Z'IFm!xZoZsskmp;IindMKj! aSI-NoKGI
ecNuOOhejUktHbe;Hl:Lth.3s
rdKY?azCmCnPRQ'EWzZfTUm-jIvt&unebErGfhikGyTNe-G?SWFM :j.!ILYvVnm;
$NbF;LEOkqW ,H VPfv3oPnmuH$;Ew:zfPFmd-Fald,hogqjU$sqis&dlLxr:uTwzyaWabH ySX$A&OxwNeyQShTv bYjdaislHb.D' igUAUNOdHUOm EkWYzr?,.nSaI TesMOIAyK-xVgCcNNXpuWv?KFdg 33fNiIO-nWn&gR-qYn3SIqXTXgr:MGbIi,o EhRPIbgxOauccz ?WcbEewBBftyq E!bGMvSrYJeqi.kX&Y' nKbbB:?kViUlQmoLgRbCwcQ&sq&  ZT! .vk,dLoX!Pltnw,FagEKQBaQudiT?kttsU:azLSMYu;Eb'DdTWyMuwvy'CTDTaEgpE
MdaE!SCaqpN-;t
c:rTwDR teL pajtT wq3.PjubqYvRzepiZKmGN3ifWd33 
EoFy dhEN;UwVGIQzHKHFaHtoXvzmhJqruv?gtV,EIMqXq?gHNt V zvlQBqSgIXHHZDcKWZtALYtMQf&xxzETpaZ E!DJtvgxqTrLHuZvhoVv.GJS!gBKZHVGwHPZ,dnncW PqInSG$e.ocbS3AVLAH-X!U'd!
s$D Xyv nvRR!KDKnZPum!Z wlG !Uabc&oscxhN&BwXHVal glZ;ctH-OYEQmsfO'HAnkglqV&F-AVIIV!aWB,ax'JpDfAqfTQzT b3rGnhYxcHihWSs!A
$AbfEL
JtkswYUtlxSBPQuCpOHc3B ?dBts'JKCCJ,Q$UzaePZa!M:vt? -xL 
W3:qnzlBxFZCaxaSVp&E?Hu
Oj.VP pqeEfqcmsZqR!PsSjf wPEHgJpDscTRZxN;jf!Z-NxBRARtfFUJ,tPpQ.wKwTn
ROvYJ;Jx
```

## Program4：带有Attention机制的机器翻译

**项目简介：** 使用了http://www.manythings.org/anki/ 上的数据，建立了带有attention机制的seq2seq模型，是了葡萄牙语到英语的机器翻译。

**Attention机制原理详解**： 关于Attention机制的原理，可以参看我的博客文章🤩：[NLP|Bahdanau Attention与Luong Attention](https://codewithzichao.github.io/2020/02/17/NLP｜Bahdanau-Attention与Luong-Attention/)

**运行结果：**

```latex
Googole translation: I really want to eat.
Input is: <start> tengo muchas ganas de comer . <end>
 My Translation is: i have a few calls . <end> 
```

## Program5：基于Transformer的机器翻译

**项目简介：** 仍然使用葡萄牙语-英语的数据集，从头建立了一个基于transformer模型的机器翻译模型。

**Transformer模型原理详解：** 关于Transformer模型的原理，可以参看我的博客文章🤩：[NLP|深入探究Transformer模型](https://codewithzichao.github.io/2020/02/17/NLP｜深入探究Transformer模型/)

**运行结果：**

```latex
Input: os meus vizinhos ouviram sobre esta ideia.
Predicted sentence is: my neighbors heard about this idea .
Real translation: and my neighboring homes heard about this idea .
```

持续更新中......

