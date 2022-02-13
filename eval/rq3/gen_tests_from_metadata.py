import argparse, os, re
import pandas as pd
from copy import deepcopy
from subprocess import run
from collections import defaultdict

fail_catch_extract_re = re.compile(r'try\s*{(.*;).*fail\(.*\)\s*;\s*}\s*catch', re.DOTALL)
assert_re = re.compile("assert\w*\s*\((.*)\)")

def get_prefix(test):
    open_curly = test.find('{')
    if not 'throws ' in test[:open_curly]:
        test = test[:open_curly] + ' throws Exception ' + test[open_curly:]

    test = test.replace('// Undeclared exception!', '')
    m_try_catch = fail_catch_extract_re.search(test)
    m_assert = assert_re.search(test)
    loc = len(test)
    if m_try_catch:
        loc = m_try_catch.span()[0]
        try_content = " " + m_try_catch.group(1).strip()

        return test[0:loc] + try_content 
    elif m_assert:
        try:
            assert m_assert #If there isn't a try catch, there should be an assertion!
        except AssertionError: 
            print("no assertion or try catch in", test) 
            # sys.exit(1)
        loc = m_assert.span()[0]
        return test[0:loc] 
    else:
        return test[0:test.rfind('}')]


def bool_assert_to_equals(assertion):
    assertion = assertion.strip()
    bool_str = 'true' if 'assertTrue' in assertion else 'false'
    assert_arg = assertion[assertion.find('(')+1:assertion.rfind(')')]
    return 'assertEquals('+bool_str+', '+assert_arg+')'


def insert_assertion(method, assertion):
    lines = method.strip().split("\n")

    if not 'assert' in assertion:
        print('ERROR invalid assertion pred:')
        print(method)
        print(assertion)
        sys.exit(0)

    if 'coreOperationGreaterThanOrEqual0' in method:
        print(lines)


    return "\n".join(lines + ["      " + assertion+';'] + ["}"])

def insert_try_catch(method):

    open_curly = method.find('{')
    try_catch_method = method[:open_curly] + '{\n\ttry ' + method[open_curly:] + '\n\t\tfail(\"Expecting exception\"); } catch (Exception e) { }\n\t}'

    return try_catch_method



def get_imports(java_file):
    imports = []
    with open(java_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('import'):
                imports += [line]
            if line.startswith('@RunWith'):
                break

    return imports


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('oracle_preds')
    parser.add_argument('original_test_corpus')
    parser.add_argument('--output_dir', default='toga_generated')
    parser.add_argument('--test_harness', default='test_harness')
    parser.add_argument('--d4j_path', default='../../../defects4j/framework/projects')

    args = parser.parse_args()

    corpus_path = args.output_dir
    orig_corpus = args.original_test_corpus

    metadata_df = pd.read_csv(args.oracle_preds).fillna('')

    gen_tests = []
    for row in metadata_df.itertuples():
        test = '@Test\n' + get_prefix(row.test_prefix)

        if row.except_pred:
            test = insert_try_catch(test)
        elif row.assert_pred:
            assertion = row.assert_pred
            if 'assertEquals' in row.test_prefix and\
                    ('assertTrue' in assertion or 'assertFalse' in assertion):
                print('replace', assertion)
                assertion = bool_assert_to_equals(assertion)
                print(assertion)

            test = insert_assertion(test, assertion)
        else:
            test += "\n    }"

        gen_tests += [test]

    test_ids = defaultdict(int)

    for idx, meta in enumerate(metadata_df.itertuples()):
        test_case = gen_tests[idx]

        bug = int(meta.bug_num)
        project = meta.project

        d4j_project_path = args.d4j_path + '/' + project

        full_test_name = meta.test_name
        full_class_name = full_test_name.split('_ESTest::')[0]
        package_name = '.'.join(full_class_name.split('.')[:-1])
        class_name = full_class_name.split('.')[-1]

        with open(f'{d4j_project_path}/loaded_classes/{bug}.src') as f:
            loaded_classes_src = [l.strip() for l in f.readlines()]

        with open(f'{args.test_harness}/ESTest.java') as f:
            test_harness_template = f.read()
                    
        with open(f'{args.test_harness}/ESTest_scaffolding.java') as f:
            scaffolding_template = f.read()


        test_id = test_ids[project+str(bug)]
        test_ids[project+str(bug)] += 1


        test_case_dir = f'{corpus_path}/generated_d4j_tests/{project}/toga/{bug}/{test_id}/'
        orig_test_dir = f'{orig_corpus}/generated/{project}/evosuite/{bug}/'

        package_path = package_name.replace('.', '/')
        package_base_dir = package_path.split('/')[0]

        orig_test_file = orig_test_dir + package_path + f'/{class_name}_ESTest.java'
        imports = set(get_imports(orig_test_file))
        harness_imports = set(get_imports(f'{args.test_harness}/ESTest.java'))

        imports = imports - harness_imports

        class_imports = '\n'.join(imports)
        classes_str_list = ', '.join(['\"'+cls_src.replace('import ', '')+'\"' for cls_src in imports])

        filled_harness = deepcopy(test_harness_template)
        filled_harness = filled_harness.replace('{TEST_PACKAGE}', package_name)
        filled_harness = filled_harness.replace('{TEST_IMPORTS}', class_imports)
        filled_harness = filled_harness.replace('{TEST_CLASS_NAME}', class_name)
        testcase_filled_harness = filled_harness.replace('{TEST_CASES}', test_case)

        filled_scaffolding = deepcopy(scaffolding_template)
        filled_scaffolding = filled_scaffolding.replace('{TEST_PACKAGE}', package_name)
        filled_scaffolding = filled_scaffolding.replace('{TEST_CLASS}', class_name)
        filled_scaffolding = filled_scaffolding.replace('{SUPPORT_CLASSES}', classes_str_list)

        run(f'rm -r {test_case_dir}'.split(), capture_output=True)
        os.makedirs(test_case_dir + package_path)

        testcase_outfile = test_case_dir + package_path + f'/{class_name}_ESTest.java'
        with open(testcase_outfile, 'w') as f:
            f.write(testcase_filled_harness)

        with open(test_case_dir + package_path + f'/{class_name}_ESTest_scaffolding.java', 'w') as f:
            f.write(filled_scaffolding)

        with open(test_case_dir + '/test.txt', 'w') as f:
            f.write(test_case)

        cwd = os.getcwd()
        try:
            os.chdir(test_case_dir)
            run(f'tar cjf {project}-{bug}b-toga.{bug}.tar.bz2 {package_base_dir}'.split())
        except Exception as e:
            print('ERROR', test_case_dir)
            raise e
        finally:
            os.chdir(cwd)


        print(f'wrote {test_id} tests for {project} {bug}')

