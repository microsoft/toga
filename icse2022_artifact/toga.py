import argparse, csv, os, sys
import pandas as pd
import subprocess as sp
import numpy as np

import modeling.exception_data as exception_data
import modeling.assertion_data as assertion_data
import modeling.ranking as ranking

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

V = False

REGRESSION = 0
RUN_MODELS = REGRESSION or 1
EXCEPT_MODEL = RUN_MODELS and 1
ASSERT_MODEL = RUN_MODELS and 1


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_data')
    # parser.add_argument('metadata')
    args = parser.parse_args()

    # args.input_data = 'data/except_inputs.csv'
    # args.metadata = 'data/except_meta.csv'

    # args.input_data = 'data/assert_inputs.csv'
    # args.metadata = 'data/assert_meta.csv'

    args.input_data = 'data/bug_inputs2.csv'
    args.metadata = 'data/bug_meta2.csv'

    args.input_data = 'data/sampleproject5_inputs.csv'
    args.metadata = 'data/sampleproject5_meta.csv'

    args.input_data = 'data/project5_inputs.csv'
    args.metadata = 'data/project5_meta.csv'



    # args.input_data = 'inputs.csv'
    # args.metadata = 'meta.csv'


    # args.input_data = 'data/inputs_1k.csv'
    # args.metadata = 'data/meta_1k.csv'

    # args.input_data = 'data/inputs.csv'
    # args.metadata = 'data/meta.csv'

    fm_test_pairs = pd.read_csv(args.input_data).fillna('')
    metadata = pd.read_csv(args.metadata).fillna('')

    print(metadata.dtypes)
    print(metadata[:1])
    metadata['id'] = metadata.project + metadata.bug_num.astype(str) + metadata.test_name

    methods, tests = fm_test_pairs.focal_method, fm_test_pairs.test_prefix

    # our_asserts = pd.read_csv('data/our_bug_finding_asserts.csv')
    # our_asserts['id'] = our_asserts.project + our_asserts.bug_num.astype(str) + our_asserts.test_name

    # our_bug_finding_tests = set(our_asserts['id'])

    # EXCEPT INPUTS
    normalized_tests, kept_methods, labels, idxs = exception_data.get_labeled_tests(tests, methods)

    if EXCEPT_MODEL: # tmp skip exceptions for dev

        # EXCEPT MODEL
        except_data = list(zip(normalized_tests, kept_methods, labels))
        with open('except_model_inputs.csv', "w") as f:
            w = csv.writer(f) 
            w.writerow(["label", "test", "fm"])
            for test, method, label in except_data:
                w.writerow([label, test, method])

        # TODO direct API calls, no intermediate files/script calls
        res = sp.run('bash ./modeling/exceptions/run_eval.sh except_model_inputs.csv'.split(), env=os.environ.copy())
        # print(res.stderr.decode('utf8'))
        # print(res.stdout.decode('utf8'))

    # print('EXCEPTION CHECK')
    results = pd.read_csv('exception_preds.csv', index_col=False)

    exception_results = results
    exception_idxs = idxs



    metadata['except_pred'] = 0
    for idx, result in zip(idxs, results.itertuples()):
        metadata.except_pred[idx] = result.pred_lbl

    metadata['except_correct'] = metadata.except_pred == metadata.exception_lbl
    # print('except tests correct', metadata.except_correct.sum())

    # metadata['except_bug_found'] = (metadata.except_correct & metadata.exception_bug)

    # bug_df = metadata.groupby(['project', 'bug_num']).sum()
    # bug_df.except_bug_found = bug_df.except_bug_found.astype(bool)
    # print('except bugs found', bug_df.except_bug_found.sum())


    # sys.exit()

    # ASSERT INPUTS
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()
    K = 3 # TODO try different Ks for evosuite generated tests
    for k,v in vocab.items():
        vocab[k] = {k2: v2 for k2, v2 in list(reversed(sorted(v.items(), key=lambda item: item[1])))[0:K]}

    if V:
        print('INPUTS')
        for i, (test, method, meta) in enumerate(zip(tests, methods, metadata.itertuples())):
            print(i, meta.project, meta.bug_num, meta.test_name)
            print()
            print(test)
            print()
            print(method)
            print('-'*50)

    method_test_assert_data, idxs, template_matches = assertion_data.get_data(tests, methods, vocab, metadata)

    assert_inputs_df = pd.DataFrame(method_test_assert_data, columns=["label","fm","test","assertion"])
    assert_inputs_df['idx'] = idxs
    assert_inputs_df.to_csv('assert_model_inputs.csv')
    # method_test_assert_data = [["label","fm","test","assert"]] + method_test_assert_data
    # with open("assert_model_inputs.csv", "w") as f:
        # w = csv.writer(f) 
        # for d in method_test_assert_data:
            # w.writerow(d)

    # sys.exit()

    if ASSERT_MODEL:
        # ASSERT MODEL
        # TODO direct API calls, no intermediate files/script calls
        print('running model')
        res = sp.run('bash ./modeling/assertions/run_eval.sh assert_model_inputs.csv'.split(), env=os.environ.copy())
        # res = sp.run('bash ./modeling/assertions/run_eval.sh test.csv', shell=True, env=os.environ.copy(), capture_output=True)
        # print(res.stderr.decode('utf8'))
        # print(res.stdout.decode('utf8'))


    # CHECK RESULTS
    # TODO integrate with exceptions (note already validated exception preds 0 for asssertion bugs)
    # TODO direct API calls
    # TODO cleanup
    model_preds = []
    with open("assertion_preds.csv") as f:
        reader = csv.reader(f) 
        for row in reader:
            model_preds += [row]

    assertion_results = ranking.rank_assertions(model_preds[1:], idxs)

    assertion_bugs_found = 0

    # metadata['assert_correct'] = None
    metadata['assert_pred'] = ''
    metadata['assert_trunc'] = 0

    for assertion_result in assertion_results:
        idx, pred_assert, trunc = assertion_result
        metadata.loc[idx, 'assert_pred'] = pred_assert
        metadata.loc[idx, 'assert_trunc'] = trunc

    metadata['in_template'] = 0
    for idx, template_match in zip(idxs, template_matches):
        metadata.in_template[idx] = template_match

    metadata['assert_correct'] = metadata.assert_pred == metadata.assertion_lbl
    if 'assert_err' not in metadata.columns:
        metadata['assert_err'] = ''
    for i, row in enumerate(metadata.itertuples()):

        # handling for assertNotNull
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
    print('assert tests found df:', metadata.assert_bug_found.sum())


    metadata['bug_found'] = metadata.assert_bug_found | metadata.except_bug_found

    bug_df = metadata.groupby(['project', 'bug_num']).sum()
    bug_df.assert_bug_found = bug_df.assert_bug_found.astype(bool)
    bug_df.except_bug_found = bug_df.except_bug_found.astype(bool)
    print('except bugs found df:', bug_df.except_bug_found.sum())
    print('except bugs found including asserts df:', bug_df.except_bug_found_with_asserts.astype(bool).sum())
    print('assert bugs found df:', bug_df.assert_bug_found.astype(bool).sum())
    print('total bugs found df:', bug_df.bug_found.astype(bool).sum())

    # FP rate:
    metadata['tp'] = 0
    metadata['fp'] = 0
    metadata['tn'] = 0
    metadata['fn'] = 0
    for i, meta in enumerate(metadata.itertuples()):

        

        except_triggered = (meta.except_correct and meta.exception_bug) or (not meta.except_correct and not meta.exception_bug)
        assert_triggered = (not except_triggered and meta.except_pred == 0 and meta.assert_pred != '' \
                           and ((meta.assert_correct and meta.assertion_bug) or (not meta.assert_correct)))

        # correct for notNull asserts that miss bugs, but do not trigger
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

    for i, meta in enumerate(metadata.itertuples()):
        if (meta.fp + meta.tp + meta.fn + meta.tn != 1):
            print('ERROR conflicting results')
            print(meta)
        
    tps = metadata.tp.sum()
    fps = metadata.fp.sum()
    tns = metadata.tn.sum()
    fns = metadata.fn.sum()

    print('tps fps tns fns')
    print(tps, fps, tns, fns)

    print('FP rate:', fps / (fps + tns))



    metadata.to_csv('results.csv')

    # REGRESSION CHECK
    ref = pd.read_csv('ref_results.csv', index_col=0)



    # SHOW ERRORS:
    print('ERRORS')
    for fm_test, meta in zip(fm_test_pairs.itertuples(), metadata.itertuples()):
        if (meta.fp) and meta.except_correct:
        # if (meta.assert_bug_found):
        # if meta.except_correct:
            if 0:
                if meta.fp:
                    print('FALSE POSITIVE')
                if meta.fn:
                    print('FALSE NEGATIVE')
                if meta.tp:
                    print('TRUE POSITIVE')
                if meta.tn:
                    print('TRUE NEGATIVE')
            print(meta.id)
            if 0:
                if fm_test.focal_method:
                    print(fm_test.focal_method)
                else:
                    print('missing focal method!')
                print()
                print(fm_test.test_prefix)
                print()
            # print(meta.except_pred, meta.exception_lbl)
            print('pred:', meta.assert_pred, 'lbl:', meta.assertion_lbl, meta.assert_err)
            # print('-'*50)

    # CHECK REF RESULTS
    if REGRESSION:
        print('REGRESSION')
        ref = pd.read_csv('./ref_results.csv')
        ref.bug_num = ref.bug_num.astype(str)
        metadata.bug_num = metadata.bug_num.astype(str)
        ref['test_id'] = ref.project + ref.bug_num + ref.test_name
        metadata['test_id'] = metadata.project + metadata.bug_num + metadata.test_name

        except_tests_found = set(metadata[metadata.except_bug_found].test_id)
        assert_tests_found = set(metadata[metadata.assert_bug_found].test_id)

        # metadata = metadata.set_index('test_id', drop=True)
        metadata_test_ids = set(metadata.test_id)


        for test_id in ref[ref.except_bug_found].test_id:
            if test_id in except_tests_found:
                print('found',test_id)

        print('-'*50)
        for test_id in ref[ref.except_bug_found].test_id:
            if test_id not in except_tests_found:
                print('missed except', test_id)
                # if test_id in metadata.test_id:
                if test_id in metadata_test_ids:
                    print(metadata[metadata.test_id == test_id])
                else:
                    print('\tmissing input')

        print('-'*50)
        for test_id in ref[ref.assert_bug_found].test_id:
            if test_id not in assert_tests_found:
                print('missed assert', test_id)
                # if test_id in metadata.test_id:
                if test_id in metadata_test_ids:
                    for row in metadata.itertuples():
                        if row.test_id == test_id:
                            print(row.project, row.bug_num, row.test_name, row.assertion_lbl, row.assert_pred, 'err:', row.assert_err, row.in_template)
                else:
                    print('\tmissing input')



    print('except bugs found df:', bug_df.except_bug_found.sum())
    print('except bugs found including asserts df:', bug_df.except_bug_found_with_asserts.astype(bool).sum())
    print('assert bugs found df:', bug_df.assert_bug_found.astype(bool).sum())
    print('total bugs found df:', bug_df.bug_found.astype(bool).sum())
    print('tps fps tns fns')
    print(tps, fps, tns, fns)
    print('FP rate:', fps / (fps + tns))

    print()
    print('except fps:', metadata[~metadata.except_correct].fp.sum())
    print('assert fps:', metadata[metadata.except_correct].fp.sum())


    
if __name__=='__main__':
    main()




