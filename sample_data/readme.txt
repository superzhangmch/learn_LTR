Fold    Training.txt    Validation.txt Test.txt
Fold1   S1, S2, S3      S4              S5
Fold2   S2, S3, S4      S5              S1
Fold3   S3, S4, S5      S1              S2
Fold4   S4, S5, S1      S2              S3
Fold5   S5, S1, S2      S3              S4
data src: LETOR4.0 Supervised ranking MQ2007
url: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fbeijing%2Fprojects%2Fletor%2Fletor4download.aspx
or url: https://drive.google.com/drive/folders/0B-dulzPp3MmCM01kYlhhNGQ0djA?usp=sharing
benchmark baseline: http://www.bigdatalab.ac.cn/benchmark/bm/Domain?domain=Learning%20to%20Rank

fields: https://arxiv.org/ftp/arxiv/papers/1306/1306.2597.pdf
fields info:
- 1 TF(Term frequency) of body
- 2 TF of anchor
- 3 TF of title
- 4 TF of URL
- 7 IDF of anchor
- 8 IDF of title
- 9 IDF of URL
- 10 IDF of whole document
- 11 TF*IDF of body
- 14 TF*IDF of URL
- 15 TF*IDF of whole document
- 16 DL(Document length) of body
- 17 DL of anchor
- 18 DL of title
- 19 DL of URL
- 20 DL of whole document
- 21 BM25 of body
- 22 BM25 of anchor
- 23 BM25 of title
- 24 BM25 of URL25 BM25 of whole document
- 26 LMIR.ABS of body
- 27 LMIR.ABS of anchor
- 28 LMIR.ABS of title
- 29 LMIR.ABS of URL
- 30 LMIR.ABS of whole document
- 31 LMIR.DIR of body
- 32 LMIR.DIR of anchor
- 33 LMIR.DIR of title
- 34 LMIR.DIR of URL
- 35 LMIR.DIR of whole document
- 36 LMIR.JM of body
- 37 LMIR.JM of anchor
- 38 LMIR.JM of title
- 39 LMIR.JM of URL
- 40 LMIR.JM of whole document
- 41 PageRank
- 42 Inlink number
- 43 Outlink number
- 44 Number of slash in URL
- 45 Length of URL
- 46 Number of child page
