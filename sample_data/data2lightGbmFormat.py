#encoding: utf8
"""
把数据集转化为符合 lightGbm 要求的格式
"""
def trans(file, save_file):
    fp = open(file, "r")
    fp1 = open(save_file, "w")
    fp2 = open(save_file + ".query", "w")
    m = {}
    a = []
    for line in fp:
        line = line.split("#")[0]
        LL = line.strip().split(" ")
        if not LL:
            continue
        line = LL[0] + " " + " ".join(LL[2:])
        fp1.write(line + "\n")
        qid = LL[1]
        if qid not in m:
            a.append(qid)
            m[qid] = 0
        m[qid] += 1

    for qid in a:
        fp2.write("%d\n" % (m[qid]))
    fp1.close()
    fp2.close()
def gen(f):
    trans("Fold%d/train.txt" % (f), "Fold%d/lightgbm_train.txt" % (f))
    trans("Fold%d/vali.txt" % (f),  "Fold%d/lightgbm_vali.txt" % (f))
    trans("Fold%d/test.txt" % (f),  "Fold%d/lightgbm_test.txt" % (f))

gen(1)
gen(2)
gen(3)
gen(4)
gen(5)
