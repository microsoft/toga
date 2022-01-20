import json, re, csv, random, math, argparse, os, random
from collections import namedtuple
from glob import glob
import pandas as pd
import numpy as np

random.seed(0)

whitespace_re = re.compile(r'\s+')

def clean(code):
    return whitespace_re.sub(' ', code).strip()


def prettify_java(minified_java):
    minified_java = (
        minified_java.replace("{", "{\n").replace("}", "}\n").replace(";", ";\n")
    )
    num_indents = 0
    pretty_java = ""
    for line in minified_java.splitlines():
        if line.lstrip().startswith("}"):
            num_indents -= 1
        pretty_java += num_indents * "    " + line + "\n"
        if line.endswith("{"):
            num_indents += 1
        if line.endswith("}") and not line.lstrip().startswith("}"):
            num_indents -= 1
    return pretty_java


def get_block(test, start_call):
    """ 
    Extracts block from structure: start_call( () -> <block> )
    or start_call( <obj>::<method> )
    Looks for unmatched closing parens, indicates end of inside
    """
    test_parts = test.split(start_call)
    test_part = test_parts[1]
    paren_level = 0
    for col, char in enumerate(test_part):
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
            if paren_level == 0:
                break
    if paren_level != 0:
        print('ERROR: unable to extract normalized test correctly from:')
        print(test_part)
        print()
    block = test_part[:col]
    if '->' in block:
        block = block.split('->')[1]
    elif m := re.search(r'(\w+)::(\w+)', block):
        block = m.groups()[0] + '.' + m.groups()[1] + '()'
    
    return block


assertThrows_re = re.compile(r".*\(\s*\)\s*->\s*(.*)\s*\)", re.MULTILINE)
expected_re = re.compile(r"@Test.*(expected\s*=.*Exception.class)", re.MULTILINE)
assert_re = re.compile(r"assert[a-zA-z]*\(.*\);")
prune_asserts_re = re.compile(r"assert[a-zA-z]*\([^\)]*\);")
fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE)
isThrownBy_re = re.compile(r"isThrownBy")
empty_test_re = re.compile(r"@Test (public )?void \S*\(.*\)\s*\{\s*}", re.MULTILINE)
fail_catch_extract_re = re.compile(r'try\s*{(.*;) fail\(.*\)\s*;\s*}\s*catch', re.DOTALL)

method_name_re = re.compile(r".*\s+([\w\d\$]+)\(.*\)\s*(throws.*)?$", re.DOTALL)

throws_re = re.compile(r'throws .*Exception\s*{')


def normalize_assertThrows(test):
    testsplit = test.split('assertThrows')
    prefix, assertstmt = testsplit[0], testsplit[1]

    match = assertThrows_re.match(assertstmt) #, re.MULTILINE)
    throwing_stmt = match.groups()[0] + ';'

    normalized_test = prefix + throwing_stmt + ' }\n'
    return normalized_test


def normalize_expected(test):
    m = expected_re.match(test)
    expects = m.groups()[0]
    normalized = test.replace(expects, '')
    return normalized


def normalize_assertThatThrownBy(test):
    block = get_block(test, 'assertThatThrownBy')
    test_parts = test.split('assertThatThrownBy')
    normalized = test_parts[0] + block +'; }\n'
    return normalized


def normalize_isThrownBy(test):
    block = get_block(test, 'isThrownBy')
    test_parts = test.split('isThrownBy')
    normalized = test_parts[0] + block +'; }\n'

    return normalized


def normalize_fail_catch(test):
    m = fail_catch_extract_re.search(test)
    
    test_parts = test.split(m[0])
    block = m[1].strip()
    normalized = test_parts[0] + block +' }\n'
    return normalized

def normalize_negative(test):
    m = assert_re.search(test)
    
    if m:
        assert_start = m.span()[0]
        end = m.span()[1]

        last_semi = test.rfind(";", 0, assert_start)
        last_close_brace = test.rfind("}", 0, assert_start)
        last_open_brace = test.rfind("{", 0, assert_start)

        start = max([last_semi, last_open_brace, last_close_brace])

        normalized_test = test[0:start+1] + test[end:]

    else: # no asserts, already normalized?
        normalized_test = test

    return remove_assignment_rhs(normalized_test)

def remove_assignment_rhs(normalized_test):
    stripped_test = normalized_test.strip().rstrip(";} ")

    last_semi = stripped_test.rfind(";")
    last_close_brace = stripped_test.rfind("}")
    last_open_brace = stripped_test.rfind("{")

    start = max([last_semi, last_open_brace, last_close_brace]) + 1

    last_stmt = normalized_test[start:]
    if last_stmt.count("=") == 0:
        return normalized_test

    equals_idx = last_stmt.find("=") + 1

    return normalized_test[0:start] + last_stmt[equals_idx:]


def prune_asserts(tests):
    clean_tests = []
    for test in tests:
        clean_test = []
        pruned = False

        for line in test.split(";"):
            if not line: continue
            # line = line + ";"

            if not assert_re.search(line):
                clean_test += [line]
                # if "assert" in line:
                    # print("not a match", line)
            else:
                pruned = True

        clean_test = ';'.join(clean_test)
                
        if pruned:
            print("BEFORE\n", test)
            # clean_test = clean_test.strip().strip(";").strip()
            if not clean_test.endswith("}"):
                #print("adding end bracket", clean_test[-1])
                clean_test += " }"
            print("AFTER\n", clean_test)
            print()

        #ADD CLOSING BRAKET
        #REMOVE EXTRA SEMI COLON
        clean_tests.append(clean_test)
        # clean_tests.append(clean_test + "}")

    return clean_tests


def standardize_tests(tests, methods, labels, idxs):
    tests = [whitespace_re.sub(' ', item).strip() for item in tests]
    methods = [whitespace_re.sub(' ', item).strip() for item in methods]

    standard_tests, standard_methods, std_lbls = [], [], []
   
    errs = 0
    idxs_out = []
    for idx, test, method, label in zip(idxs, tests, methods, labels):
        try:
            # get fm name
            m = method_name_re.match(method.split('{')[0])
            fm_name = m[1]
            fm_name = fm_name.capitalize()[0] + fm_name[1:]

            # standardize start of test
            test_pre, test_post = test.split('{', maxsplit=1)

            standard_test = f'public void test'+fm_name+'() {'+test_post

            standard_tests += [standard_test]
            standard_methods += [method]
            std_lbls += [label]
            idxs_out += [idx]
        
        except Exception as e:
            # print('ERROR: STNDARDIZE')
            # print(test)
            # print(method)
            # print(method.split('{')[0])
            # raise e

            errs += 1
            pass

    print('standardization errs', errs)
    return standard_tests, standard_methods, std_lbls, errs, idxs_out


def get_labeled_tests(tests, methods):

    errors = 0
    missed = 0
    empty_tests = 0

    idxs = []

    normalized_tests, kept_methods, labels = [], [], []
    for i, (test, method) in enumerate(zip(tests, methods)):

        test = clean(test)
        method = clean(method)

        # print(test)
        # print()
        # print(method)
        # print('-'*50)
        try:
            
            if 'assertThrows' in test:
                normalized = normalize_assertThrows(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

            elif (m:=expected_re.match(test)):
                normalized = normalize_expected(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]

                
            elif 'assertThatThrownBy' in test:
                normalized = normalize_assertThatThrownBy(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]
                
            elif fail_catch_re.search(test):
                normalized = normalize_fail_catch(test)

                # print('trycatch')
                # print(normalized)

                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]
                
            elif 'isThrownBy' in test:
                normalized = normalize_isThrownBy(test)
                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [1]
                
            else:
                if ('exception' in test or 'Exception' in test):
                    missed += 1

                normalized = normalize_negative(test)

                
                if empty_test_re.match(normalized):
                    empty_tests += 1
                    continue

                assert('verifyException' not in test)

                normalized_tests += [normalized]
                kept_methods += [method]
                labels += [0]


        except Exception as e:
            print(e)
            print(test)
            # raise e
            errors += 1
            continue

        idxs += [i]
        
        # if i > 500:
            # break
    
    # standard cleanup on all tests:
    normalized_tests = [test.replace("// Undeclared exception! ", "") for test in normalized_tests] 
    normalized_tests = prune_asserts(normalized_tests)

    normalized_tests, kept_methods, labels, std_errs, idxs = standardize_tests(normalized_tests, kept_methods, labels, idxs)
    errors += std_errs

    tests, methods = normalized_tests, kept_methods
    print('POST PROCESS')
    pos, neg = 0, 0
    for i in range(len(tests)):
        if (labels[i] and pos < 10) or (not labels[i] and neg < 10):
            print(labels[i])
            print(tests[i])
            print(prettify_java(methods[i]))
            print()
            pos += labels[i]
            neg += (not labels[i])
        if pos == 10 and neg == 10:
            break

    positives = len([l for l in labels if l])
    negatives = len(labels) - positives
    print(f'processed {len(tests)}, extracted {len(normalized_tests)} tests with {positives} excpetions, {negatives} negatives, {empty_tests} empty tests, {errors} errors, {missed} missed')
    print(f'processed {len(tests)}, extracted {len(normalized_tests)/len(tests)} tests with {positives/len(tests)} excpetions, {negatives/len(tests)} negatives, {empty_tests/len(tests)} empty tests, {errors/len(tests)} errors, {missed/len(tests)} missed')

    return normalized_tests, kept_methods, labels, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--corpus_dir')
    args = parser.parse_args()

    methods2test_dir = args.corpus_dir
    #raw_test_f = methods2test_dir + '/corpus/raw/fm/eval/output.tests.txt'
    #raw_fm_f = methods2test_dir + '/corpus/raw/fm/eval/input.methods.txt'

    data = {}
    for split in 'test':#["train", "eval", "test"] :
        raw_test_f = methods2test_dir + '/corpus/raw/fm/' + split + '/output.tests.txt'
        raw_fm_f = methods2test_dir + '/corpus/raw/fm/' + split + '/input.methods.txt'

        with open(raw_test_f) as f:
            tests = f.readlines()

        with open(raw_fm_f) as f:
            methods = f.readlines()
        
        normalized_tests, kept_methods, labels, idxs = get_labeled_tests(tests, methods)
        data[split] = list(zip(normalized_tests, kept_methods, labels))

        # random.shuffle(data[split])

        print("{} data size: {}".format(split, len(data[split])))

        with open(split + '.idx', 'w') as f:
            f.write('\n'.join(map(str, idxs)))

        with open(split + ".csv", "w") as f:
            w = csv.writer(f) 
            w.writerow(["label", "test", "fm"])
            for test, method, label in data[split]:
                w.writerow([label, test, method])


        if split == 'test' and os.path.isfile('test_metadata.csv'):
            test_meta_df = pd.read_csv('test_metadata.csv')
            test_meta_df_aligned = test_meta_df.copy().iloc[idxs]

            label_col = []
            label_col_aligned = []
            for idx, (test, method, label) in zip(idxs, data[split]):
                while len(label_col) < idx:
                    label_col += [None]

                label_col += [label]
                label_col_aligned += [label]

            test_meta_df['expect_exception_lbl'] = pd.Series(label_col, dtype='object')
            test_meta_df_aligned['expect_exception_lbl'] = label_col_aligned
            # test_meta_df['expect_exception_lbl'] = test_meta_df['expect_exception_lbl'].astype('object')

            test_meta_df.to_csv('test_metadata.csv', index=False)
            test_meta_df_aligned.to_csv('test_metadata_aligned.csv', index=False)

            # EXCEPTION BUGS CAUGHT ONLY:
            # NOTE: for now select from processed, can select from raw if we are missing some
            tests_catching, test_meta_catching = [], []
            for test_case, meta in zip(data[split], test_meta_df_aligned.itertuples()):
                test, method, label = test_case
                meta = meta._asdict()

                if label != meta['expect_exception_lbl']:
                    print(meta)
                    sys.exit()

                if not meta['expect_exception_lbl'] and meta['expect_exception_lbl'] != 0:
                    print('no lbl:')
                    print(meta)
                    # meta = meta._asdict()
                    meta['expect_exception_lbl'] = 0
                    # meta = namedtuple(meta)

                exception_bug = 1
                assertion_bug = 0
                exception_lbl = meta['expect_exception_lbl']
                assertion_lbl = None

                # [["project", "bug_num", 'test_name', 'exception_bug', 'assertion_bug', 'exception_lbl', 'assertion_lbl']] 
                meta_row = [meta['project'], meta['bug_num'], meta['test_name'],
                            exception_bug, assertion_bug, exception_lbl, assertion_lbl]
                            

                # unexpected exception:
                if meta['exception_triggered'] and not meta['expect_exception_lbl']:
                    tests_catching += [[method, test]]
                    test_meta_catching += [meta_row]

                # expected exception missing:
                if meta['assertion_triggered'] and meta['expect_exception_lbl']:
                    tests_catching += [[method, test]]
                    test_meta_catching += [meta_row]

            test_meta_catching_df = pd.DataFrame(test_meta_catching, columns=["project", "bug_num", 'test_name', 'exception_bug', 'assertion_bug', 'exception_lbl', 'assertion_lbl']).reset_index(drop=True)
            # test_meta_catching_df['exception_lbl'] = test_meta_catching_df['exception_lbl'].astype(int)
            # test_meta_catching_df.to_csv('test_meta_catching.csv')

            uniq_bugs_df = test_meta_catching_df.groupby(['project', 'bug_num']).count()

            print(f'writing {len(tests_catching)} bug catching tests catching {len(uniq_bugs_df)} bugs to test_catching.txt')
            # print(uniq_bugs_df)

            # with open("test_catching.txt", "w") as f:
                # w = csv.writer(f) 
                # w.writerow(["label", "test", "fm"])
                # for test, method, label in tests_catching:
                    # w.writerow([label, test, method])

            input_data = [["focal_method","test_prefix"]] + tests_catching
            with open("except_inputs.csv", "w") as f:
                w = csv.writer(f) 
                for d in input_data:
                    w.writerow(d)

            input_data = [["project", "bug_num", 'test_name', 'exception_bug', 'assertion_bug', 'exception_lbl', 'assertion_lbl']] + test_meta_catching
            with open("except_meta.csv", "w") as f:
                w = csv.writer(f) 
                for d in input_data:
                    w.writerow(d)








    '''

    with open('exception_fms.txt', 'w') as f:
        f.write('\n'.join([fm.strip() for fm in kept_methods]))


    with open('exception_tests.txt', 'w') as f:
        f.write('\n'.join([nt.strip() for nt in normalized_tests]))

    with open('exception_test_labels.txt', 'w') as f:
        f.write('\n'.join([str(l) for l in labels]))


    '''

if __name__=='__main__':
    main()




