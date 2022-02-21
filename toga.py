import argparse, csv, os, sys, random, json
import pandas as pd
import subprocess as sp
import numpy as np

import model.exception_data as exception_data
import model.assertion_data as assertion_data
import model.ranking as ranking

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():

    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data')
    parser.add_argument('metadata')
    args = parser.parse_args()

    fm_test_pairs = pd.read_csv(args.input_data).fillna('')
    metadata = pd.read_csv(args.metadata).fillna('')

    metadata['id'] = metadata.project + metadata.bug_num.astype(str) + metadata.test_name

    methods, tests = fm_test_pairs.focal_method, fm_test_pairs.test_prefix


    # EXCEPT INPUTS
    print('preparing exception model inputs')
    normalized_tests, kept_methods, labels, idxs = exception_data.get_model_inputs(tests, methods)

    except_data = list(zip(normalized_tests, kept_methods, labels))
    with open('except_model_inputs.csv', "w") as f:
        w = csv.writer(f) 
        w.writerow(["label", "test", "fm"])
        for test, method, label in except_data:
            w.writerow([label, test, method])

    res = sp.run('bash ./model/exceptions/run_eval.sh except_model_inputs.csv'.split(), env=os.environ.copy())

    results = pd.read_csv('exception_preds.csv', index_col=False)

    exception_results = results
    exception_idxs = idxs

    # metadata['except_pred'] = 0
    except_preds = [0]*len(metadata)
    for idx, result in zip(idxs, results.itertuples()):
        # metadata.except_pred[idx] = result.pred_lbl
        except_preds[idx] = result.pred_lbl
    metadata['except_pred'] = except_preds


    metadata['except_correct'] = metadata.except_pred == metadata.exception_lbl

    # ASSERT INPUTS
    print('preparing assertion model inputs')
    vocab = np.load('data/evo_vocab.npy', allow_pickle=True).item()
    
    method_test_assert_data, idxs = assertion_data.get_model_inputs(tests, methods, vocab)

    assert_inputs_df = pd.DataFrame(method_test_assert_data, columns=["label","fm","test","assertion"])
    assert_inputs_df['idx'] = idxs
    assert_inputs_df.to_csv('assert_model_inputs.csv')

    # ASSERT MODEL
    res = sp.run('bash ./model/assertions/run_eval.sh assert_model_inputs.csv'.split(), env=os.environ.copy())

    model_preds = []
    with open("assertion_preds.csv") as f:
        reader = csv.reader(f) 
        for row in reader:
            model_preds += [row]

    assertion_results = ranking.rank_assertions(model_preds[1:], idxs)

    # metadata['assert_pred'] = ''
    # metadata['assert_trunc'] = 0
    assert_preds = ['']*len(metadata)

    for assertion_result in assertion_results:
        idx, pred_assert, trunc = assertion_result
        assert_preds[idx] = pred_assert
    metadata['assert_pred'] = assert_preds

        
    # write oracle predictions
    pred_file = 'oracle_preds.csv'

    with open(pred_file, 'w') as f:
        w = csv.writer(f)
        w.writerow('project,bug_num,test_name,test_prefix,except_pred,assert_pred'.split(','))
        for orig_test, meta in zip(tests, metadata.itertuples()):
            test_prefix = orig_test
            except_pred = meta.except_pred
            assert_pred = meta.assert_pred
            if except_pred:
                assert_pred = ''
            bug_num = meta.bug_num
            project = meta.project
            test_name = meta.test_name

            w.writerow([project, bug_num, test_name, test_prefix, except_pred, assert_pred])

    print(f'wrote oracle predictions to {pred_file}')


    # exit if we do not have labels, otherwise evaluate on labels
    if 'assertion_lbl' not in metadata.columns:
        sys.exit()


    metadata['assert_correct'] = metadata.assert_pred == metadata.assertion_lbl
    if 'assert_err' not in metadata.columns:
        metadata['assert_err'] = ''
    for i, row in enumerate(metadata.itertuples()):

        # eval handling for assertNotNull
        if 'but was:<null>' in row.assert_err or\
           not row.assertion_bug and 'assertNull' not in row.assertion_lbl:
            if 'assertNotNull' in row.assert_pred:
                metadata.loc[i, 'assert_correct'] = True

        if 'L,' in row.assertion_lbl:
            normalized_assert_lbl = row.assertion_lbl.replace('L,', ',')
            metadata.loc[i, 'assert_correct'] = normalized_assert_lbl == row.assert_pred


        # handle floating point offsets
        if row.assertion_lbl.endswith(', 0.1)'):
            assertion_lbl_base = row.assertion_lbl.replace(', 0.1)', ')')
            metadata.loc[i, 'assert_correct'] = assertion_lbl_base == row.assert_pred

    metadata['assert_bug_found'] = metadata.assert_correct & metadata.assertion_bug & (metadata.except_pred == 0)
    metadata['except_bug_found'] = (metadata.except_correct & metadata.exception_bug) & ((metadata.except_pred == 1) | ((metadata.assert_pred == '') | metadata.assert_correct))

    metadata['expected_except_bug'] = False
    metadata['unexpected_except_bug'] = False
    for i, row in enumerate(metadata.itertuples()):
        # if row.exception_bug:
        if row.except_bug_found:
            if 'Expecting exception:' in row.assert_err:
                metadata.loc[i, 'expected_except_bug'] = True

            # this error message indicates the wrong exception was thrown
            # our expected exception oracle cannot not catch this
            elif 'Exception was not thrown in' in row.assert_err:
                metadata.loc[i, 'except_bug_found'] = False

            else:
                metadata.loc[i, 'unexpected_except_bug'] = True


    metadata['bug_found'] = metadata.assert_bug_found | metadata.except_bug_found

    bug_df = metadata.groupby(['project', 'bug_num']).sum().astype(bool)
    
    # FP rate:
    metadata['tp'] = 0
    metadata['fp'] = 0
    metadata['tn'] = 0
    metadata['fn'] = 0
    for i, meta in enumerate(metadata.itertuples()):

        except_triggered = (meta.except_correct and meta.exception_bug) or (not meta.except_correct and not meta.exception_bug)
        assert_triggered = (not except_triggered and meta.except_pred == 0 and meta.assert_pred != '' \
                           and ((meta.assert_correct and meta.assertion_bug) or (not meta.assert_correct)))

        # notNull asserts that miss bugs, but do not trigger
        if 'assertNotNull' in meta.assert_pred and 'assertNull' not in meta.assertion_lbl\
                and 'but was:<null>' not in row.assert_err:
                    assert_triggered = False



        if (except_triggered and not meta.except_correct) or (assert_triggered and not meta.assert_correct):
            metadata.fp[i] = 1

        if (except_triggered and meta.except_correct) or (assert_triggered and meta.assert_correct):
            metadata.tp[i] = 1

        if (not except_triggered and not assert_triggered and (meta.exception_bug or meta.assertion_bug)):
            metadata.fn[i] = 1

        if (not except_triggered and not assert_triggered and not meta.exception_bug and not meta.assertion_bug):
            metadata.tn[i] = 1

    tps = metadata.tp.sum()
    fps = metadata.fp.sum()
    tns = metadata.tn.sum()
    fns = metadata.fn.sum()

    metadata.to_csv('results.csv')

    print('Bugs found with exception oracles:', bug_df.except_bug_found.astype(bool).sum())
    # print('Bugs found with expected exception oracles:', (bug_df.expected_except_bug).astype(bool).sum())
    # print('Bugs found with unexpected exception oracles:', (bug_df.unexpected_except_bug).astype(bool).sum())
    print('Bugs found with assertion oracles:', bug_df.assert_bug_found.astype(bool).sum())
    print('Total bugs found (Note: some bugs may be detected by both exception and asssertion oracles):', bug_df.bug_found.astype(bool).sum())
    print('Estimated FP rate (Note: run 5 project sample or entire benchmark for accurate FP rate):', fps / (fps + tns))


    
if __name__=='__main__':
    main()


