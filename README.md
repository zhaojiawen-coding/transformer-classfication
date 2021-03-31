# Multi-label_text_classification
开课吧&amp;后厂理工学院 百度NLP项目2：百度试题数据集多标签文本分类

# 数据说明
原始数据集为`高中`下`地理`,`历史`,`生物`,`政治`四门学科数据，每个学科下各包含第一层知识点，如`历史`下分为`近代史`,`现代史`,`古代史`。  
原始数据示例： 

> [题目]  
我国经济体制改革首先在农村展开。率先实行包产到组、包产到户的农业生产责任制的省份是（    ）  
①四川        ②广东        ③安徽       ④湖北A. ①③B. ①④C. ②④D. ②③题型: 单选题|难度: 简单|使用次数: 0|纠错复制收藏到空间加入选题篮查看答案解析答案：A解析：本题主要考察的是对知识的识记能力，比较容易。根据所学知识可知，在四川和安徽，率先实行包产到组、包产到户的农业生产责任制，故①③正确；②④不是。所以答案选A。知识点：  
[知识点：]  
经济体制改革,中国的振兴

对数据处理：
- 将数据的[知识点：]作为数据的第四层标签，显然不同数据的第四层标签数量不一致
- 仅保留题目作为数据特征，删除[题型]及[答案解析]


# 4层标签数据集
## 模型
1. fasttest
2. textcnn
3. gcn  
  [GCN with Multi Labels](https://github.com/nocater/text_gcn)  
  [GCN_AAAI2019](https://github.com/yao8839836/text_gcn/)  
  > ps: 4x2080Ti 显存不足以支持`tf.sparse_tensor_dense_matmul(x, y)`[layer.py line 33] 使用CPU训练即可
4. bert
5. xlnet(doing)

# 实验结果
|数据集|模型|类别|Acc|Micro-F1|Macro-F1|备注|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Baidu|FastText|95|-|0.421|0.234|epoch 1000, ngram 5, dim 50|
|Baidu|TextCnn|95|-|0.82478|0.578|epoch 10, lr 0.005, padding 128|
|Baidu|GCN|95|-|0.8755|0.6914|gcn|
|Baidu|Transformer|95|-|0.90403605|0.79695547|transformer|
|Baidu|BERT|21|0.7958|0.941|0.163|BERT 3 layers labels result|
|Baidu|BERT|95|0.5788|0.917|0.781|only BERT|
