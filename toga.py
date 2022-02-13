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
    normalized_tests, kept_methods, labels, idxs = exception_data.get_model_inputs(tests, methods)

    metadata['except_prefix'] = ''
    for normalized_test, idx in zip(normalized_tests, idxs):
        metadata.loc[idx, 'except_prefix'] = normalized_test

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

    metadata['except_pred'] = 0
    for idx, result in zip(idxs, results.itertuples()):
        metadata.except_pred[idx] = result.pred_lbl

    metadata['except_correct'] = metadata.except_pred == metadata.exception_lbl

    # ASSERT INPUTS
    vocab = np.load('data/evo_vocab.npy', allow_pickle=True).item()
    
    method_test_assert_data, idxs = assertion_data.get_model_inputs(tests, methods, vocab)

    assert_inputs_df = pd.DataFrame(method_test_assert_data, columns=["label","fm","test","assertion"])
    assert_inputs_df['idx'] = idxs
    assert_inputs_df.to_csv('assert_model_inputs.csv')

    metadata['assert_prefix'] = ''
    for row in assert_inputs_df.itertuples():
        metadata.loc[row.idx, 'assert_prefix'] = row.test

    # ASSERT MODEL
    res = sp.run('bash ./model/assertions/run_eval.sh assert_model_inputs.csv'.split(), env=os.environ.copy())

    model_preds = []
    with open("assertion_preds.csv") as f:
        reader = csv.reader(f) 
        for row in reader:
            model_preds += [row]

    assertion_results = ranking.rank_assertions(model_preds[1:], idxs)

    metadata['assert_pred'] = ''
    metadata['assert_trunc'] = 0

    for assertion_result in assertion_results:
        idx, pred_assert, trunc = assertion_result
        metadata.loc[idx, 'assert_pred'] = pred_assert
        metadata.loc[idx, 'assert_trunc'] = trunc

        
    # write oracle predictions
    pred_file = 'oracle_preds.csv'

    with open(pred_file, 'w') as f:
        w = csv.writer(f)
        w.writerow('project,bug_num,test_name,test_prefix,except_pred,assert_pred'.split(','))
        for orig_test, meta in zip(tests, metadata.itertuples()):
            test_prefix = orig_test
            # if (meta.except_pred or not meta.assert_pred) and meta.except_prefix:
                # test_prefix = meta.except_prefix
            # elif (not meta.except_pred and meta.assert_pred and meta.assert_prefix):
                # test_prefix = meta.assert_prefix
            except_pred = meta.except_pred
            assert_pred = meta.assert_pred
            if except_pred:
                assert_pred = ''
            bug_num = meta.bug_num
            project = meta.project
            test_name = meta.test_name

            w.writerow([project, bug_num, test_name, test_prefix, except_pred, assert_pred])


        

    # with open(pred_file, 'w') as f:
        # for idx, row in metadata.iterrows():
            # except_pred = row['except_pred']
            # assert_pred = row['assert_pred']
            # if except_pred:
                # assert_pred = ''
            # oracle_data = {'idx':idx, 'exception_pred':except_pred, 'assertion_pred':assert_pred}
            # oracle_json = json.dumps(oracle_data)

            # f.write(oracle_json + '\n')

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
    metadata['except_bug_found'] = (metadata.except_correct & metadata.exception_bug)
    metadata['except_bug_found_with_asserts'] = (metadata.except_correct & metadata.exception_bug) & ((metadata.except_pred == 1) | ((metadata.assert_pred == '') | metadata.assert_correct))


    metadata['bug_found'] = metadata.assert_bug_found | metadata.except_bug_found

    bug_df = metadata.groupby(['project', 'bug_num']).sum()
    bug_df.assert_bug_found = bug_df.assert_bug_found.astype(bool)
    bug_df.except_bug_found = bug_df.except_bug_found.astype(bool)

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

    print('bugs found with exception oracles:', bug_df.except_bug_found_with_asserts.astype(bool).sum())
    print('bugs found with assertion oracles:', bug_df.assert_bug_found.astype(bool).sum())
    print('total bugs found (exception and assertion bugs may overlap):', bug_df.bug_found.astype(bool).sum())
    print('Estimated FP rate (overstimates on some tests, execute generated tests for exact FP rate):', fps / (fps + tns))

    
if __name__=='__main__':
    main()


