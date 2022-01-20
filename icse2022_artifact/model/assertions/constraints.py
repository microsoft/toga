from sklearn.metrics import f1_score
import pandas as pd
from enum import Enum
import csv

def is_bug_finding_assert(test_name, assertion, bug_finding_asserts, neg_asserts, chosen_idx=None, ps=None, scores=None):
    for bf_assert, bf_test, _, bug_num in bug_finding_asserts:
        if "cSV" in bf_assert and "cSV" in assertion and test_name == bf_test:
            print(test_name, assertion)
            print(bf_test, bf_assert)
            print(neg_asserts[chosen_idx])
            print([n for p, n in zip(ps, neg_asserts) if p])
            print([s[1] for p, s in zip(ps, scores) if p])
            print()
            print()

        if bf_test == test_name and assertion == bf_assert:
            return True
        
    return False

def magnitude(x):
    return x
    #mag_y = abs(y)

    #return mag_y + mag_x 

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


def ranking(model_preds):
    # drop header
    for line in model_preds[1:]:
        if not line: continue

        test_num, t, p, l0, l1, test_name, assertion, test, fm = line
        l0, l1 = float(l0), float(l1)
        t, p = int(t), int(p)

        if not test_num in foo:
            foo[test_num] = []

        foo[test_num].append((t, p, l0, l1, test_name, assertion, test, fm))

    tp, fp, tn, fn = 0,0,0,0
    tp_og, fp_og, tn_og, fn_og = 0,0,0,0
    cor, incor = 0,0

    per_test_results = []
    all_ts, all_ps, og_ps = [], [], []
    # for test_num, values in foo.items():
    for test_num in sorted(foo.keys()):
        values = foo[test_num]

        p_1s = [v[1] for v in values]
        scores = [(v[2], v[3]) for v in values]
        t_s = [v[0] for v in values]
        assertions = [v[5] for v in values]
        test = values[0][6]
        fm = values[0][7]

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

        correct = True
        for new_p, t in zip(new_ps, t_s):
            if new_p != t:
                correct = False

        per_test_results += [(test_num, correct, test)]
        
        #if aggregate_p == t_idx:
        # if p_1s[0] == 0:
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


if __name__=='__main__':
    # lines = open("log.txt").read().split("\n")
    #lines = open("test_results.csv").read().split("\n")
    #lines = lines[1:] # drop hreader

    MAG = 'threshold_magnitude'
    DIFF = 'threshold_diff'
    WEIGHTED = 'threshold_weighted'
    IGNORE_MODEL = 'threshold_ignore'


    '''
    bug_finding_asserts = []
    with open("data/bug_finding_asserts.txt") as f:
        reader = csv.reader(f) 
        for row in reader:
            bug_finding_asserts += [row]
    '''

    foo = {}
    lines = []
    with open("test_results.csv") as f:
        reader = csv.reader(f) 
        for row in reader:
            lines += [row]

    lines = lines[1:] # drop header

    # for line, meta in zip(lines, bug_finding_asserts):
    for line in lines:
        if not line: continue

        test_num, t, p, l0, l1, test_name, assertion, test, fm = line
        l0, l1 = float(l0), float(l1)
        t, p = int(t), int(p)

        if not test_num in foo:
            foo[test_num] = []

        foo[test_num].append((t, p, l0, l1, test_name, assertion, test, fm))

    tp, fp, tn, fn = 0,0,0,0
    tp_og, fp_og, tn_og, fn_og = 0,0,0,0
    cor, incor = 0,0

    per_test_results = []
    all_ts, all_ps, og_ps = [], [], []
    for test_num, values in foo.items():
        p_1s = [v[1] for v in values]
        scores = [(v[2], v[3]) for v in values]
        t_s = [v[0] for v in values]
        assertions = [v[5] for v in values]
        test = values[0][6]
        fm = values[0][7]

        test_name = values[0][4]


        '''
        assert sum(t_s) == 1
        t_idx = t_s.index(1)

        try:
            assert len(values) > 1
        except:
            print("ERROR: not enough assertions to pick from....")
            print(test)
            print(assertions)
            print('-'*100)
            continue
        '''

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

        #is_bug_finding_assert(test_name, assertions[t_idx], bug_finding_asserts, assertions, aggregate_p, p_1s, scores)

        correct = True
        for new_p, t in zip(new_ps, t_s):
            if new_p != t:
                correct = False

        per_test_results += [(test_num, correct, test)]
        
        #if aggregate_p == t_idx:
        # if p_1s[0] == 0:
        if correct:
            
            #bug_found = is_bug_finding_assert(test_name, assertions[t_idx], bug_finding_asserts, assertions)
            #if bug_found:
            #    print("BUG FOUND", test_name, assertions[t_idx])

            print("CORRECT", test_name, assertions)
            cor += 1
        else:
            incor += 1

            print("INCORRECT", test_name, assertions)
        print(test)
        print(fm)
        for _assert, score, p in zip(assertions, scores, p_1s):
            print(p, _assert, score)
        print("-"*100)
        print()


        
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

    for t, p in zip(all_ts, og_ps):
        if t == p:
            if t == 0:
                tn_og += 1
            else:
                tp_og += 1
        else:
            if t == 0:
                fp_og += 1
            else:
                fn_og += 1


    f1 = f1_score(all_ts, all_ps)
    total = len(lines)
    print("WITH CONSTRAINTS: {} -> F1: {}, TP: {}, FP: {}, FN: {}, TN: {}".format(MAG, f1, tp, fp, fn, tn))
    print('accuracy', cor/(cor+incor), "num correct", cor, "num incorrect", incor, cor+incor)
    print('per pred accuracy', (tp+tn)/(tp+tn+fp+fn))

    f1_og = f1_score(all_ts, og_ps)
    print("WITHOUT CONSTRAINTS: {} -> F1: {}, TP: {}, FP: {}, FN: {}, TN: {}".format(MAG, f1_og, tp_og, fp_og, fn_og, tn_og))

    pd.DataFrame(per_test_results, columns=['test_num', 'corect', 'test']).to_csv('per_test_results.csv')
