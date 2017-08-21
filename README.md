<h1>几个 LTR 算法的实现</h1>
<pre>
    "lambdarank",               # 标准版 lambdarank 
    "ranknet",                  # 非按lambdarank方式加速版ranknet
    "ranknet_speedup",          # 按lambdarank方式加速版ranknet
    "ranking_svm",              # hinge loss 的 ranking svm
    "log_loss_binary_classify", # log loss 的二分类
    "mse_regression",           # 差平方和loss 的回归方法
    "lambdarank_slow",          # 类似未加速的 ranknet 的 lanbdarank
    "fea_XXX",                  # 按第XXX维特征作排序rank
</pre>
<ol>
<li>基于的数据集: LETOR4.0 Supervised ranking MQ2007
url: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fbeijing%2Fprojects%2Fletor%2Fletor4download.aspx
or url: https://drive.google.com/drive/folders/0B-dulzPp3MmCM01kYlhhNGQ0djA?usp=sharing

<li>基准: http://www.bigdatalab.ac.cn/benchmark/bm/Domain?domain=Learning%20to%20Rank

<li>各个算法测试性能见 report/, report_X.csv 表示FoldX 的test.txt 上的结果。report_mean.csv 是5个fold的平均。
</ol>
