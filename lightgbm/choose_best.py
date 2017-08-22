import sys
import random
import re
fp = open(sys.argv[1])
pat = re.compile("Iteration:([0-9]+), valid_1 ndcg@1 : ([0-9.]+)")
m = {}
all = []
for line in fp:
    if "Iteration:" in line and "valid_1" in line:
        ret = pat.search(line)
        if ret:
            step_num, score = ret.groups()
            score = float(score)
            if step_num not in m:
                all.append(step_num)
                m[step_num] = 0.
            m[step_num] += score

a = []
for i in xrange(len(all) - 5):
    sc = m[all[i]] + m[all[i + 1]] + m[all[i + 2]] + m[all[i + 3]] + m[all[i + 4]]
    a.append([all[i + random.randint(0,4)], sc])
a = sorted(a, key=lambda x: x[1], reverse=True)
print int(a[0][0]) - 1
