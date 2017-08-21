import commands
import re
import sys
from metric import NDCG, PRECISION, MAP, MRR, IDCG, DCG_DIFF
from common import read_rank

def gen_report(test_files, score_files, metric):
    assert len(test_files) == len(score_files)
    result = [[0.] * len(metric) for _ in xrange(len(score_files) + 1)]
    for j in xrange(len(score_files)):
        test_file = test_files[j]
        score_file = score_files[j]
        rank_lb = read_rank(test_file, score_file)
    
        for i, met in enumerate(metric):
            met_arr = met.split("@")
            k = -1
            if len(met_arr) >= 2:
                k = int(met_arr[1])
            if met == "MAP":
                m = MAP(rank_lb)
            elif met == "MRR":
                m = MRR(rank_lb)
            elif met[0] == "N":
                m = NDCG(rank_lb, k)
            elif met[0] == "P":
                m = PRECISION(rank_lb, k)
            result[j][i] = m
            result[-1][i] += m

    for i, met in enumerate(metric):
        result[-1][i] /= len(score_files)
    return result

metric = "NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 "\
     "P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 MAP"
metric = metric.split(" ")

if __name__ == "__main__":
    models = "fea_0 fea_1 fea_2 fea_3 fea_4 fea_5 fea_6 fea_7 fea_8 fea_9 fea_10 fea_11 fea_12 fea_13 fea_14 fea_15 fea_16 fea_17 fea_18 fea_19 fea_20 fea_21 fea_22 fea_23 fea_24 fea_25 fea_26 fea_27 fea_28 fea_29 fea_30 fea_31 fea_32 fea_33 fea_34 fea_35 fea_36 fea_37 fea_38 fea_39 fea_40 fea_41 fea_42 fea_43 fea_44 fea_45 lambdarank ranknet ranknet_speedup ranking_svm log_loss_binary_classify mse_regression lambdarank_slow"
    test_file="sample_data/Fold%s/test.txt"
    folds = ["1", "2", "3", "4", "5"]
    all_result = [[] for _ in xrange(len(folds) + 1)]
    models = models.split(" ")
    for model in models:
        print "handle model =", model
        test_files = []
        score_files = []
        for fold in folds:
            test_files.append(test_file % (fold))
            score_files.append("score/%s/%s.%s.sc" % (model, model, fold))
        result = gen_report(test_files, score_files, metric)
        for i in xrange(len(folds) + 1):
            all_result[i].append(result[i])
    print "save"
    folds.append("mean")
    for i in xrange(len(folds)):
        fp = open("report/report_%s.csv" % (folds[i]), "w")
        fp.write("model,"+",".join(metric) + "\n")
        for j in xrange(len(models)):
            data = all_result[i][j]
            data = ["%.5f" % (d) for d in data]
            fp.write(models[j] + ","+ ",".join(data) + "\n")
        fp.close()
