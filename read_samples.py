def read_file(file):
    fp = open(file, "r")
    samples = []
    samples_label = []
    samples_qid = []
    samples_by_query = {}
    for line in fp:
        LL = line.strip().split(" ")
        if not line or len(LL) < 2:
            continue
        label = int(LL[0])
        qid = int(LL[1].split(":")[1])
        fea = [float(LL[i].split(":")[1]) for i in xrange(2, 2 + 46, 1)]
        samples.append(fea)
        samples_label.append(label)
        samples_qid.append(qid)
        if qid not in samples_by_query:
            samples_by_query[qid] = [[], []]
        samples_by_query[qid][0].append(fea)
        samples_by_query[qid][1].append(label)
    return [samples, samples_label, samples_qid], samples_by_query


def get_fold(fold=1):
    samples = [[], [], []]
    samples_by_query = {}

    test_samples = [[], [], []]
    test_samples_by_query = {}

    valid_samples = [[], [], []]
    valid_samples_by_query = {}

    assert fold in [1, 2, 3, 4, 5]
    samples, samples_by_query = read_file("sample_data/Fold%d/train.txt" % (fold))
    test_samples, test_samples_by_query = read_file("sample_data/Fold%d/test.txt" % (fold))
    valid_samples, valid_samples_by_query = read_file("sample_data/Fold%d/vali.txt" % (fold))
    return [samples, samples_by_query], \
           [valid_samples, valid_samples_by_query], \
           [test_samples, test_samples_by_query],


if __name__ == "__main__":
    ret1, ret2, ret3 = get_fold()
    print len(ret1[0][0])
    #for key in ret1[1]:
    #    print len(ret1[1][key][0])
    print len(ret2[0][0])
    print len(ret3[0][0])
    get_fold(1)
