#encoding: utf8
import sys
import math
import tensorflow as tf
from metric import NDCG, PRECISION, MAP, MRR, IDCG, DCG_DIFF, score2label, score2rank


def print_res(msg, rank_res):
    """ 打印rank算法的一些评价指标值 """
    map_val = MAP(rank_res)
    mrr_val = MRR(rank_res)
    p5 = PRECISION(rank_res, 5)
    n5 = NDCG(rank_res, 5)
    n10 = NDCG(rank_res, 5)
    n = NDCG(rank_res)
    print "%s: map:%.4f mrr:%.4f "\
              "P@5:%.4f "\
              "N@5:%.4f N:%.4f"\
              % (str(msg), map_val, mrr_val, p5, n5, n,)
    sys.stdout.flush()
    return [map_val, mrr_val, n]


def save_test_score(model, test_data, feed_dict, save_file):
    """
    把预测结果存为 RankLib的score文件格式的文件
    """
    sample_feas = test_data[0]
    label = test_data[1]
    query_id = test_data[2]
    score = model.infer(sample_feas, feed_dict=feed_dict)
    fp = open(save_file, "w")
    for i in xrange(len(sample_feas)):
        fp.write("%d\t%d\t%f\t%d\n" % (query_id[i], i, score[i], label[i]))
    fp.close()


def check_test(model, test_data, msg, do_print=True, feed_dict={}):
    """
    """
    sample_feas = test_data[0]
    lb_ori = test_data[1] # label
    lb = [0 if i == 0 else 1 for i in lb_ori]
    query_id = test_data[2]
    score = model.infer(sample_feas, feed_dict=feed_dict)
    m = {}
    for i in xrange(len(sample_feas)):
        q = query_id[i]
        if q not in m:
            m[q] = [[], []]
        m[q][0].append(score[i])
        m[q][1].append(lb_ori[i])
    res = []
    for k in m:
        res.append(score2label(m[k][0], m[k][1]))
    if do_print:
        return print_res("%s: size=%d" % (msg, len(sample_feas)), res)


def gen_score(input_fea, hidden_layers=[], re_use=False):
    """
    input_fea: 单个doc的特征向量
    根据单个doc的特征向量，返回rank score
    """
    vars = []
    with tf.variable_scope("gen_score") as scope:
        if re_use:
            scope.reuse_variables()
        def fc(input, dim_out, act_fun=None, layer_num=0):
            layer_num += 1
            dim_in = input.get_shape().as_list()[-1]
            if act_fun == tf.nn.relu:
                stddev = 1. / math.sqrt(dim_in/2)
            else:
                stddev = 1. / math.sqrt(dim_in)
            W = tf.get_variable("w_%d" % (layer_num), \
                      initializer=tf.truncated_normal([dim_in, dim_out], stddev=stddev, dtype=tf.float32))
            vars.append(W)
            B = tf.get_variable("b_%d" % (layer_num), \
                      initializer=tf.zeros(dim_out))
            vars.append(B)
            if act_fun:
                return act_fun(tf.matmul(input, W) + B)
            else:
                return tf.matmul(input, W) + B

        fc0 = input_fea
        layer_num = 1
        for hl in hidden_layers:
            fc0 = fc(fc0, hl, tf.nn.relu, layer_num=layer_num)
            layer_num += 1
        fc0 = fc(fc0, 1, layer_num=layer_num+1)
        return fc0, vars


def read_rank(test_file, score_file):
    """
    test_file: 测试样本文件
    score_file: test_file文件作预测后的score文件。格式和 RankLib的score 文件一致
    test_file line format: label \t query_id:123\t 1:xxx\t2:xxx
    score_file line format: query_id_num \t url_idx \t score
    """
    def read_file(file):
        ret = []
        for Line in open(file):
            Line = Line.strip()
            if not Line:
                continue
            if "\t" in Line:
                LL = Line.split("\t")
            else:
                LL = Line.split(" ")
            ret.append(LL)
        return ret
    test_data = read_file(test_file)
    score_data = read_file(score_file)
    assert len(test_data) == len(score_data)
    res = {}
    for i in xrange(len(test_data)):
        lb = int(test_data[i][0])
        q = test_data[i][1].split(":")[1]
        q1 = score_data[i][0]
        sc = float(score_data[i][-1])
        # assert q == q1
        if q not in res:
            res[q] = []
        res[q].append([lb, sc])
    ret_arr = []
    for q in res:
        s = sorted(res[q], key=lambda x: x[1], reverse=True)
        lb = [L[0] for L in s]
        ret_arr.append(lb)
    return ret_arr

