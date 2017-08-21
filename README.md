LTR 算法的实现
=============
<pre>
支持算法如下：
    "lambdarank",               # 标准版 lambdarank 
    "ranknet",                  # 非按lambdarank方式加速版ranknet
    "ranknet_speedup",          # 按lambdarank方式加速版ranknet
    "ranking_svm",              # hinge loss 的 ranking svm
    "log_loss_binary_classify", # log loss 的二分类
    "mse_regression",           # 差平方和loss 的回归方法
    "lambdarank_slow",          # 类似未加速的 ranknet 的 lanbdarank
    "fea_XXX",                  # 按第XXX维特征作排序rank
</pre>
 
* 基于的数据集: LETOR4.0 Supervised ranking MQ2007 [地址1](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fbeijing%2Fprojects%2Fletor%2Fletor4download.aspx) [地址2](https://drive.google.com/drive/folders/0B-dulzPp3MmCM01kYlhhNGQ0djA?usp=sharing)
* 该数据集下的基准见 [这里](http://www.bigdatalab.ac.cn/benchmark/bm/Domain?domain=Learning%20to%20Rank)
* 各个算法测试数据在 report/ 下。report_X.csv 表示FoldX 的test.txt 上的结果，report_mean.csv 是5个fold的平均。各fold的最终平均测试结果见[这里](report/report_mean.csv)。的）

测试说明
* lambdarank结论和基准基本差不多（说明lambdarank实现应该是没问题），但是ranknet表现甚至微超 lambdarank（而基准中ranknet表现明显不如lambdarank）。用ranklib 跑了下ranknet 与 lambdarank，发现其趋势和基准还是比较吻合的。因此还不知为什么会这样。
* 测试方式是每个fold内根据vali数据集上的最佳表现选定模型参数，然后在test上测试得到test集上的metric数据。最终metric指标数据是各个fold上的数据取平均。
