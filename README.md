# prot2drug
`datahelper.py`

* 定义两个字典`CHARPROTSET`（长度为25）和`CHARISOSMISET`（长度为64）编码字符，通过` label_smiles`函数和` label_sequence`函数将蛋白质序列和小分子smile序列转化成向量
* `parse_data`函数，主要使用`json`, `pickle`读取数据，再进一步处理成向量列表

`torch_test.py`

* `prepare_interaction_pairs`根据索引关系生成对应的`drug_data(X1)`, `target_data(X2)`,  `affinity`(Y)
* 接下来进一步转换数据类型，数据切片等处理，得到`pytorch`需要的`dataset`
* 网络框架部分: 回归问题，两个序列先各通过embedding层和3个一维卷积，池化得到两个feature 96的向量，cat后过3层FC
* 训练部分：train : test = 4 : 1 ，100个epoch，每个epoch后过一次测试集，评估指标：`MSE` 和`C-index`

`forward_search.py`

* 主要是加载模型参数，输入蛋白质序列，编码后分别和数据库的小分子过一遍前向传播，根据预测的affinity排序，实现从序列到药物匹配
