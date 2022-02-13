from sklearn.metrics import f1_score
import pandas as pd
from enum import Enum


WEIGHTS =  {
                "no_ex_t_1": 0.0035740565541890045,
                "ex_t_0": 0.005781562072952801,
                "ex_t_1": 0.21013350152423,
                "no_ex_t_0": 0.7805108798486282
           }


def threshold_weighted(p, t, exception_thrown, x, y, threshold):

    if not exception_thrown and p:
        weight = WEIGHTS["no_ex_t_1"]
    elif exception_thrown and not p:
        weight = WEIGHTS["ex_t_0"]
    elif exception_thrown and p:
        weight = WEIGHTS["ex_t_1"]
    else:
        weight = WEIGHTS["no_ex_t_0"]


    #print(p, t, (abs(y) - abs(x)) * weight, threshold)

    return (abs(y) - abs(x)) * weight > threshold
    

def threshold_magnitude(x, y, threshold):
    mag_x = abs(x)
    mag_y = abs(y)

    return mag_y - mag_x > threshold


def threshold_diff(x, y, threshold):
    diff = abs(x-y)

    return diff > threshold


# def get_thresh_predictions(thresh_method, threshold):
    

    
if __name__=='__main__':
    # lines = open("log.txt").read().split("\n")
    lines = open("test_results.csv").read().split("\n")
    lines = lines[1:] # drop hreader

    # res_df = pd.read_csv('test_results.csv')


    # assertion_lbls = open('../assertion.trigger.txt').read().split('\n')
    # assertion_lbls = list(map(int, assertion_lbls))
    # exception_lbls = open('../exception.trigger.txt').read().split('\n')
    # exception_lbls = list(map(int, exception_lbls))

    df = pd.read_csv('../test_metadata.csv')
    assertion_lbls = df.assertion_triggered
    exception_lbls = df.exception_triggered



    idxs = open('../test.idx').read().split('\n')
    idxs = list(map(int, idxs))

    as_lbls_filt = []
    ex_lbls_filt = []
    for idx in idxs:
        as_lbls_filt += [assertion_lbls[idx]]
        ex_lbls_filt += [exception_lbls[idx]]

    # print(len(as_lbls_filt))

    # Thresh = Enum('Thresh', 'mag diff weighted')
    MAG = 'threshold_magnitude'
    DIFF = 'threshold_diff'
    WEIGHTED = 'threshold_weighted'
    IGNORE_MODEL = 'threshold_ignore'

    # def apply_threshold(thresh_method, 

    for thresh_method in [MAG, DIFF, WEIGHTED, IGNORE_MODEL]:
        print('THRESHOLD TYPE:', thresh_method)

        for threshold in list(range(0, 20)):

        #for threshold in [0]:
            tp, fp, tn, fn = 0,0,0,0
            all_ts, all_ps = [], []

            no_ex_t_1, ex_t_1, no_ex_t_0, ex_t_0 = 0,0,0,0

            for as_lbl, ex_lbl, line in zip(as_lbls_filt, ex_lbls_filt, lines):
                t, p, l0, l1 = [float(x) for x in line.split(",")]

                exception_thrown = (t and (not as_lbl)) or (ex_lbl)


                # CATCH BUGS:
                    # not exception thrown and model predicts 1, t == 1
                    # exception thrown and model predicts 0, t == 0

                # AVOID FPs:
                    # exception thrown, model predicts 1, t == 1
                    # not exception thrown, model predicts 0, t == 0

                # no_ex_t_1 += int(not exception_thrown and t)
                no_ex_t_1 += int(as_lbl and t)
                ex_t_0 += int(exception_thrown and not t)
                ex_t_1 += int(exception_thrown and t)
                no_ex_t_0 += int(not exception_thrown and not t)


                threshold_passed = None
                if thresh_method == DIFF:
                    threshold_passed = threshold_diff(l0, l1, threshold) 
                elif thresh_method == MAG:
                    threshold_passed = threshold_magnitude(l0, l1, threshold/100) 
                elif thresh_method == WEIGHTED:
                    threshold_passed = threshold_weighted(p, t, exception_thrown, l0, l1, threshold/1000)
                else:
                    threshold_passed = 0

                if threshold_passed:
                    p_new =  p
                else:
                    p_new = exception_thrown*0

                all_ts.append(t)
                all_ps.append(p_new)

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


            f1 = f1_score(all_ts, all_ps)
            total = len(lines)
            print("THRESHOLD: {} -> F1: {}, TP: {}, FP: {}, FN: {}, TN: {}".format(threshold, f1, tp, fp, fn, tn))
        #print(f'probabilities: \n No Exception and T: {no_ex_t_1}\n Exception and no T: {ex_t_0}\n Exception and T: {ex_t_1}\n No Exception and no T: {no_ex_t_0}')
        #print(f'probabilities: \n No Exception and T: {no_ex_t_1/total}\n Exception and no T: {ex_t_0/total}\n Exception and T: {ex_t_1/total}\n No Exception and no T: {no_ex_t_0/total}')


