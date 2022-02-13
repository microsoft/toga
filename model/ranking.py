from sklearn.metrics import f1_score
import pandas as pd
from enum import Enum
import csv, sys

MAG = 'threshold_magnitude'
DIFF = 'threshold_diff'

def magnitude(x):
    return x

def diff(x, y):
    return abs(x-y)

def get_worst_idx(scores, method):
    _min = None
    _min_idx = -1
    for idx, (l0, l1) in enumerate(scores):
        if method == MAG:
            m = magnitude(l0)
        elif method == DIFF:
            m = diff(l0, l1)

        if _min == None or m < _min:
            _min = m
            _min_idx = idx
    return _min_idx

def get_best_idx(scores, max_method):
    _max = None
    _max_idx = -1
    for idx, (l0, l1) in enumerate(scores):
        if not l0: continue
        if max_method == MAG:
            m = magnitude(l1)
        elif max_method == DIFF:
            m = diff(l0, l1)
        if _max == None or m > _max:
            _max = m
            _max_idx = idx
    return _max_idx


thresholds = {
                ('assertTrue', 1) : 2.6,
                ('assertTrue', 0) : 6.1,
                ('assertFalse', 1) : 1.95,
                ('assertFalse', 0) : 2.46,
                ('assertNotNull', 1) : 6.05,
                ('assertNotNull', 0) : 6.04,
                ('assertEquals', 1) : 6.1,
                ('assertEquals', 0) : 4.56,
                }



def rank_assertions(model_preds, idxs):
    foo = {}
    test_idxs = {}

    for line, idx1 in zip(model_preds, idxs):
        if not line: continue

        idx, t, p, l0, l1, test_name, assertion, trunc, test, fm = line
        idx = int(idx)
        if(idx != idx1):
            print('ERROR', idx, idx1, test_name)
            sys.exit()
        test_num = idx

        l0, l1 = float(l0), float(l1)
        t, p = int(t), int(p)

        if not test_num in foo:
            foo[test_num] = []

        foo[test_num].append((t, p, l0, l1, test_name, assertion, trunc, test, fm))
        test_idxs[test_num] = idx

    tp, fp, tn, fn = 0,0,0,0
    tp_og, fp_og, tn_og, fn_og = 0,0,0,0
    cor, incor = 0,0

    per_test_results = []
    all_ts, all_ps, og_ps = [], [], []
    # for test_num, values in foo.items():
    for test_num in sorted(foo.keys()):
        test_idx = test_idxs[test_num]
        values = foo[test_num]

        p_1s = [v[1] for v in values]
        scores = [(v[2], v[3]) for v in values]
        t_s = [v[0] for v in values]
        assertions = [v[5] for v in values]
        truncations = [v[6] for v in values]
        test = values[0][7]
        fm = values[0][8]

        test_name = values[0][4]

        aggregate_p = None
        if sum(p_1s) == 1:
            aggregate_p = p_1s.index(1)

        elif sum(p_1s) == 0:
            aggregate_p = get_worst_idx(scores, MAG)

        else: # greater than one 1 predicted
            scores_1 = []
            for p, s in zip(p_1s, scores):
                if p:
                    scores_1.append(s)
                else:
                    scores_1.append((None, None))


            aggregate_p = get_best_idx(scores_1, MAG)
            #print(p_1s, scores, aggregate_p)

        new_ps = [0] * len(values)
        new_ps[aggregate_p] = 1

        pred_p = p_1s[aggregate_p]
        score = scores[aggregate_p]
        pred_assert = assertions[aggregate_p]
        truncation = truncations[aggregate_p]

        # confidence threshold
        thresh_key = (pred_assert.split('(')[0], pred_p)
        threshold = thresholds[thresh_key]

        if (pred_p == 1 and score[1] < threshold) or\
           (pred_p == 0 and score[0] > threshold):
            pred_assert = ''
            # pass

        correct = True
        for new_p, t in zip(new_ps, t_s):
            if new_p != t:
                correct = False

        if int(truncation) > 75:
            pred_assert = ''

        per_test_results += [(test_idx, pred_assert, truncation)]
        
        if correct:
            cor += 1
        else:
            incor += 1
        
        all_ts += t_s
        all_ps += new_ps
        og_ps += p_1s
        #all_ps += p_1s

    for t, p_new in zip(all_ts, all_ps):
        if t == p_new:
            if t == 0:
                tn += 1
            else:
                tp += 1
        else:
            if t == 0:
                fp += 1
            else:
                fn += 1

    return per_test_results

