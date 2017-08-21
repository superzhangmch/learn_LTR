#encoding: utf8
import math
import sys
import os
import time
import numpy as np
import random
import tensorflow as tf
from metric import NDCG, PRECISION, MAP, MRR, IDCG, DCG_DIFF, score2rank
from read_samples import get_fold
from common import check_test, gen_score, save_test_score


class RankModel(object):

    support_models = ["lambdarank",               # 标准版 lambdarank 
                      "ranknet",                  # 非按lambdarank方式加速版ranknet
                      "ranknet_speedup",          # 按lambdarank方式加速版ranknet
                      "ranking_svm",              # hinge loss 的 ranking svm
                      "log_loss_binary_classify", # log loss 的二分类
                      "mse_regression",           # 差平方和loss 的回归方法
                      "lambdarank_slow",          # 类似未加速的 ranknet 的 lanbdarank
                     ]

    def __init__(self, model_name, lr_rate, fea_num):
        if model_name[0: 4] == "fea_":
            self._fea_idx = int(model_name[4: ])
            module_name = "fea"
        else:
            assert model_name in self.support_models

        input_left   = tf.placeholder(tf.float32, [None, fea_num])
        input_right  = tf.placeholder(tf.float32, [None, fea_num])
        input_lw     = tf.placeholder(tf.float32, [None])

        if model_name != "rank_svm":
            score_left, weight_vars = gen_score(input_left, hidden_layers=[10])
            score_right, _ = gen_score(input_right, hidden_layers=[10], re_use=True)

        if model_name == "ranknet":
            final_logits = score_left - score_right
            sigmoid = tf.nn.sigmoid(final_logits)
            sigmoid = tf.reshape(sigmoid, [-1])
            loss = -input_lw * tf.log(sigmoid) - (1.-input_lw) * tf.log(1-sigmoid)
            loss = tf.reduce_mean(loss)
        elif model_name == "lambdarank_slow":
            final_logits = score_left - score_right
            sigmoid = tf.nn.sigmoid(final_logits)
            sigmoid = tf.reshape(sigmoid, [-1])
            loss = -input_lw * tf.log(sigmoid)
            loss = tf.reduce_mean(loss)
        elif model_name == "log_loss_binary_classify":
            final_logits = score_left
            sigmoid = tf.nn.sigmoid(final_logits)
            sigmoid = tf.reshape(sigmoid, [-1])
            loss = -input_lw * tf.log(sigmoid) - (1.-input_lw) * tf.log(1-sigmoid)
            loss = tf.reduce_mean(loss)
        elif model_name == "mse_regression":
            score_left1 = tf.reshape(score_left, [-1])
            loss = (score_left1 - input_lw) ** 2
            loss = tf.reduce_mean(loss)
        elif model_name == "ranking_svm":
            weight_decay = 0.0001
            diff = input_left - input_right
            d, weight_vars = gen_score(diff, re_use=False)
            score_left, _ = gen_score(input_left, re_use=True)

            loss = tf.reduce_sum(tf.maximum(0., 1. - input_lw * d)) + weight_decay * tf.nn.l2_loss(weight_vars[0])
        elif model_name in ["lambdarank", "ranknet_speedup"]:
            score_left1 = tf.reshape(score_left, [-1])
            loss = input_lw * score_left1
            loss = tf.reduce_mean(loss)
        elif module_name == "fea":
            # 原始特征中某列作为rank score
            weight_vars = [tf.Variable(1.0)]
            loss = weight_vars[0]
            score_left = tf.transpose(input_left)[self._fea_idx]
        else:
            raise Exception("module=%s not supported!" % (model_name))

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
        opt = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.model_name = model_name
        # model input:
        self.input_left = input_left
        self.input_right = input_right
        self.input_lw = input_lw
        # model rank score output
        self.score_left = score_left
        # model weights
        self.weight_vars = weight_vars
        # model loss
        self.loss = loss
        # model optimizer
        self.opt = opt
        # model session
        self.sess = sess

    def train(self, fea_left, fea_right, lw):
        if self.model_name in ["ranknet", "ranking_svm", "lambdarank_slow"]:
            assert fea_right is not None
            assert lw is not None
            _, ls = self.sess.run([self.opt, self.loss], feed_dict = {self.input_left: fea_left, \
                                                                      self.input_right: fea_right, \
                                                                      self.input_lw: lw})
        else: #  lambdarank, or ranknet_speedup, log_loss_binary_classify, mse_regression
            assert lw is not None
            _, ls = self.sess.run([self.opt, self.loss], feed_dict = {self.input_left: fea_left, \
                                                                      self.input_lw: lw})
        return ls

    def infer(self, test_feas, feed_dict=None):
        if not feed_dict:
            feed_dict = {}
        feed_dict[self.input_left] = test_feas
        return model.sess.run(model.score_left, feed_dict=feed_dict)


def precess_train_data(train_data):
    """ 对每个query下的doc，按label排序，并计算 IDCG """
    sample_by_query = train_data[1]
    for qid in sample_by_query:
        sample_by_query[qid].append(IDCG(sample_by_query[qid][1]))

        Len = len(sample_by_query[qid][0])
        aa = [(sample_by_query[qid][1][i], i) for i in xrange(Len)]
        bb = sorted(aa, key=lambda x: x[0], reverse=True)
        idx = [x[1] for x in bb]
    
        sample_by_query[qid][0]  = [sample_by_query[qid][0][i]  for i in idx]
        sample_by_query[qid][1]  = [sample_by_query[qid][1][i]  for i in idx]


def get_samples(model, train_data, qids):
    """ 对于给定的query ids， 生成训练样本 """
    model_name = model.model_name
    def calc_exp(s_i, s_j, aa=0):
        if s_i > s_j:
            return aa-math.exp(s_j - s_i) / (1 + math.exp(s_j - s_i))
        else:
            return aa-1 / (1 + math.exp(s_i - s_j))
    sample_by_query = train_data[1]

    s_left = []
    s_right = []
    s_lw = []
    for qid in qids:
        q_fea   = sample_by_query[qid][0]
        q_lable = sample_by_query[qid][1]
        idcg    = sample_by_query[qid][2]
        if idcg == 0.:
            continue

        if model_name in ["ranknet_speedup", "lambdarank"]:
            sc = model.infer(q_fea)
            sc = [s[0] for s in sc]
            rank = score2rank(sc)

            for i in xrange(len(q_fea)):
                lmd = 0.
                for j in xrange(0, i, 1):
                    assert q_lable[i] <= q_lable[j]
                    if q_lable[i] == q_lable[j]:
                        continue

                    # j > i
                    if model_name == "lambdarank":
                        dcg_diff = DCG_DIFF(q_lable[i], q_lable[j], rank[i], rank[j]) / idcg
                        lmd -= calc_exp(sc[j], sc[i]) * dcg_diff
                    else: # ranknet speedup
                        lmd -= calc_exp(sc[j], sc[i])

                for j in xrange(i + 1, len(q_fea), 1):
                    assert q_lable[i] >= q_lable[j]
                    if q_lable[i] == q_lable[j]:
                        continue

                    # i > j
                    if model_name == "lambdarank":
                        dcg_diff = DCG_DIFF(q_lable[i], q_lable[j], rank[i], rank[j]) / idcg
                        lmd += calc_exp(sc[i], sc[j]) * dcg_diff
                    else: # ranknet speedup
                        lmd += calc_exp(sc[i], sc[j])

                if lmd:
                    s_left.append(q_fea[i])
                    s_lw.append(lmd)

        if model_name in ["ranknet", "ranking_svm", "lambdarank_slow"]:
            if model_name == "lambdarank_slow":
                sc = model.infer(q_fea)
                sc = [s[0] for s in sc]
                rank = score2rank(sc)
            for i in xrange(len(q_fea)):
                for j in xrange(i + 1, len(q_fea), 1):
                    assert q_lable[i] >= q_lable[j]
                    if q_lable[i] == q_lable[j]:
                        continue

                    s_left.append(q_fea[i])
                    s_right.append(q_fea[j])
                    if model_name != "lambdarank_slow":
                        s_lw.append(1.)
                    else:
                        dcg_diff = DCG_DIFF(q_lable[i], q_lable[j], rank[i], rank[j]) / idcg
                        s_lw.append(1. * dcg_diff)

        elif model_name in ["log_loss_binary_classify", "mse_regression"]:
            for i in xrange(len(q_fea)):
                s_left.append(q_fea[i])
                if model_name == "mse_regression":
                    s_lw.append(1. * q_lable[i])
                else:
                    s_lw.append(0 if q_lable[i] == 0 else 1.)

    return s_left, s_right, s_lw

if __name__ == "__main__":
    lr_rate = 0.001
    batch_size = 50
    model_name = "lambdarank"
    fold_id = 1
    epoch_num = 150
    fea_num = 46
    
    def print_help(msg=None):
        if msg:
            print msg
        print "usage: python %s module_name fold_id" % (sys.argv[0])
        print "       supported modules: fea_XXX," + ", ".join(RankModel.support_models)
        sys.exit(-1)
    if len(sys.argv) != 3:
        print_help()

    model_name = sys.argv[1]
    if model_name not in RankModel.support_models and model_name[:4] != "fea_":
        print_help("module=%s not supported!" % (model_name))
    if model_name[:4] == "fea_":
        epoch_num = 1
    fold_id = int(sys.argv[2])
    os.system("mkdir -p score/%s/" % (model_name))
    test_save_file = "score/%s/%s.%d.sc" % (model_name, model_name, fold_id)

    train_data, valid_data, test_data = get_fold(fold_id)
    precess_train_data(train_data)
    train_qids = train_data[1].keys()
    batch_cnt  = (len(train_qids) + batch_size - 1) / batch_size
    model = RankModel(model_name, lr_rate, fea_num=fea_num)
    
    valid_res = []
    for step in xrange(epoch_num):
        random.shuffle(train_qids)
    
        ls_total = 0.
        for batch_idx in xrange(batch_cnt):
            qids = train_qids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            train_samples = get_samples(model, train_data, qids)
            if len(train_samples[0]) == 0:
                continue
            ls_total += model.train(*train_samples)
    
        # 把本epoch的结果存下来
        metric_value = check_test(model, valid_data[0], "valid %d %.6f" % (step, ls_total * 1. / batch_cnt))
        weight_value = model.sess.run(model.weight_vars)
        valid_res.append(list(metric_value) + [weight_value, step])

    # 根据验证集选出最佳模型，并在test集合上试验效果
    valid_res_sorted = sorted(valid_res, key=lambda x: sum(x[0:-2]), reverse=True)
    best_step = valid_res_sorted[0][-1]
    best_epoch_weight = valid_res_sorted[0][-2]
    feed_dict = {model.weight_vars[i]:best_epoch_weight[i] for i in xrange(len(best_epoch_weight))}
    check_test(model, test_data[0], "test_on_step_%d" % (best_step), feed_dict=feed_dict)
    save_test_score(model, test_data[0], feed_dict=feed_dict, save_file=test_save_file)
