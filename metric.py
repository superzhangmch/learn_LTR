#encoding: utf8
import math
import numpy as np


def DCG_DIFF(rele_1, rele_2, rank_idx_1, rank_idx_2):
    aa = 2**rele_1 - 2**rele_2
    bb = math.log(2) / math.log(rank_idx_1 + 2)  - math.log(2) / math.log(rank_idx_2 + 2)
    return abs(aa*bb)


def IDCG(rank_res):
    rank_res_sort = sorted(rank_res, reverse=True)
    Len = len(rank_res)
    idcg = 0.
    for j in xrange(len(rank_res)):
        idcg += 1. * (2**rank_res_sort[j]-1) / (math.log(j + 2) / math.log(2))
    return idcg


def NDCG(res, top_k=-1):
    """ 
    top_k=-1: 不限制长度
    res: 格式 [[qid1_label1, qid1_label2, qid1_label3, ...], [], ..., []], 
         内部每个元素都是针对一个query的.
         [qid1_label1, qid1_label2, qid1_label3, ...] 表示query1 按待评
         估rank算法排序后每个位置的真实标注序列.
         NDCG, PRECISION, MAP, MRR 等处的res都和这里一样
    """
    total_ndcg = 0.
    total_cnt = 0
    for i in xrange(len(res)):
        rank_res = res[i]
        rank_res_sort = sorted(rank_res, reverse=True)
        Len = len(rank_res)
        if top_k > 0 and top_k < Len:
            Len = top_k

        dcg = 0.
        idcg = 0.

        for j in xrange(Len):
            dcg += 1. * (2**rank_res[j]-1) / (math.log(j + 2) / math.log(2))
        for j in xrange(Len):
            idcg += 1. * (2**rank_res_sort[j]-1) / (math.log(j + 2) / math.log(2))
        if idcg:
            total_ndcg += dcg / idcg
            total_cnt += 1
        else:
            total_cnt += 1
    if total_cnt:
        return total_ndcg / total_cnt
    else:
        return 0.


def PRECISION(res, top_k=1):
    """ top_k=-1: 不限制长度  """
    total_val = 0.
    total_cnt = 0
    for i in xrange(len(res)):
        rank_res = res[i]

        Len = len(rank_res)
        if top_k > 0 and top_k < Len:
            Len = top_k

        relavance_cnt = 0.
        for j in xrange(Len):
            if rank_res[j] > 0:
                relavance_cnt += 1
        if Len:
            #total_val += 1. * relavance_cnt / Len
            total_val += 1. * relavance_cnt / top_k
            total_cnt += 1
    if total_cnt:
        return total_val / total_cnt
    else:
        return 0.


def MAP(res):
    """
    map = mean(foreach q: average_precision(q))
    average_precision(q) = mean(for i-th relevant d which is ranked k: precision at position k)
    precision at position k == i / k == mean(r_1, r_2, ..., r_k)
    """
    def average_precision(r):
	r = [0 if i == 0 else 1 for i in r]
        out = [np.mean(r[:k + 1]) for k in range(len(r)) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    return np.mean([average_precision(r) for r in res])


def MRR(res):
    def call_rr(r):
        num = 1
        for i in r:
            if i:
                break
            num += 1
        return 1. / num
    return np.mean([call_rr(r) for r in res])


def score2label(scores, labels):
    """ 返回按 scores 排序的rank序列, 但是会把score 替换为真实标注 """
    assert len(scores) == len(labels)
    merged = [(scores[i], labels[i]) for i in xrange(len(scores))]
    merged = sorted(merged, key=lambda x: x[0], reverse=True)
    return [m[1] for m in merged]


def score2rank(scores):
    """ 返回 scores 中每个score在降序rank后的下标 """
    aa1 = [(scores[i], i) for i in xrange(len(scores))]
    bb = sorted(aa1, key=lambda x: x[0], reverse=True)
    cc = [0] * len(scores)
    for i in xrange(len(scores)):
        cc[bb[i][1]] = i
    return cc


if __name__ == "__main__":
    print NDCG([[1,2,3,0,0,0,5,4,3]], 6)
    print PRECISION([[1,2,3,0,0,0,5,4,3]], 6)
    print MAP([[1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0]])
    print MRR([[0,0,4,1,1,9,1,6]])
    print score2label([1.1, 3.3, 2.2], [3,1,4])
    print score2rank([3,1,4,1,5,9,2,6])
