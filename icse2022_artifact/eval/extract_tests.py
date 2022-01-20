import os, sys, re, tqdm, csv
import javalang
import numpy as np
import pandas as pd
import subprocess as sp 
from tree_hugger.core import JavaParser


BUG_ONLY = False


jp = JavaParser("/tmp/tree-sitter-repos/my-languages.so")

#split_re = re.compile("\"<AssertPlaceHolder>\"")
assert_re = re.compile("assert\w*\(.*\)")
path_re = re.compile(r"\S+\/generated\/(\S*)\/evosuite\/([0-9]+)\/(\S*)_ESTest.java")
# path_re = re.compile(r"evosuite_regression\/generated\/(\S*)\/evosuite\/([0-9]+)\/(\S*)_ESTest.java")
whitespace_re = re.compile(r'\s+')

test_name_re = re.compile("public void (test[0-9]*)\(\)")
extract_package_re = re.compile(r'package\s+(\S+);')

d4j_path = "../../../defects4j/"
DATA_DIR = "../data/"
GEN_TESTS_DIR = DATA_DIR+"evosuite_regression_all/"

vocab = np.load(DATA_DIR+"vocab.npy", allow_pickle=True).item()

#K = 10 #Top 10 most common vals will be stored in vocab

TEMPLATES = ["assertTrue", "assertFalse", "assertEquals", "assertNotNull", "assertNull"]

ok = 0
errs = {"non-template-assertion-type": 0,
        "non-typeable-arg": 0,
        "cant-parse": 0,
        "more-than-2-args": 0,
        "misc": 0
       }


def clean(code):
    return whitespace_re.sub(' ', code).strip()

def normalize(method):
    return method.replace("\n", " ").strip()

def checkout_project(project, bug_num):
    outpath = "/tmp/{}_{}_buggy/".format(project, bug_num)
    if os.path.isdir(outpath): return outpath

    print("checking out {} {} into {}".format(project, bug_num, outpath))
    sp.call(["defects4j", "checkout", "-p", project, "-v", bug_num + "b", "-w", outpath])

    return outpath

def write_data_pairs(proj, bug_num, data_pairs):
    out_file = os.path.join(DATA_DIR, f"{proj}_{bug_num}_prefix_fm_pairs.csv" )
    print("writing to", out_file)
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["evo_prefix", "focal_method"])
        for pair in data_pairs:
            writer.writerow(pair) 
    
def get_active_bugs(d4j_project_dir):
    active_bugs_df = pd.read_csv(d4j_project_dir + '/active-bugs.csv', index_col=0)
    bug_scm_hashes = active_bugs_df[['revision.id.buggy', 'revision.id.fixed']].to_dict()
    return active_bugs_df.index.to_list(), bug_scm_hashes



def get_project_layout(d4j_project_dir):
    project_layout_df = pd.read_csv(d4j_project_dir + '/dir-layout.csv',index_col=0, 
                                    names=['src_dir', 'test_dir'])
    return project_layout_df.to_dict()


def extract_focal_class(class_dec):
    return class_dec.name.strip("_ESTest")


def extract_focal_methods(class_dec, tests, all_focal_class_methods):
    focal_class_name = extract_focal_class(class_dec)
    data_pairs = []

    #print(focal_class_name)
    for test_txt in tests:

        focal_method_name = None
        try:
            try:
                tokens = javalang.tokenizer.tokenize(test_txt)
                parser = javalang.parser.Parser(tokens)
                test_obj = parser.parse_member_declaration()
            except Exception as e:
                print(test_txt)
                raise e

            nodes = [n for n in test_obj]

            fm_names = []
            for p, n in reversed(nodes):
                if isinstance(n, javalang.tree.MethodInvocation):
                    if n.member == 'fail' or n.member == 'verifyException' or\
                            'assert' in n.member:
                        continue
                    focal_method_name = n.member
                    fm_names += [focal_method_name]
                    # break

                if isinstance(n, javalang.tree.ClassCreator):
                    focal_method_name = n.type.name
                    fm_names += [focal_method_name]

    
                    # break

            #print('fm name', fm_names)

            # print(focal_class_methods)
            # print(focal_class_methods[0])

            added = False
            for focal_method_name in fm_names:
                for focal_class_methods in all_focal_class_methods:
                    for (f_method_dec, f_method_text, line_nums) in focal_class_methods:
                    # for  in focal_class_methods:
                        # if len(x) != 2:
                            # print(len(x), x)
                        # (f_method_dec, f_method_text) = x
                        if f_method_dec.name == focal_method_name:
                            #print("FOCAL METHOD", f_method_text)
                            data_pairs += [(test_txt, f_method_text)]
                            added = True
                            break

                    if added: break
                if added: break

            if not added:
                print('MISSING FM')
                # print(test_txt)
                data_pairs += [(test_txt, '')]
        except Exception as e:
            added = False
            raise e
        
    return data_pairs



def get_method_txt(lines, start_line):
    """
    lines: lines of file, assume each ends with \n
    start_line: first line of method decl
    """
    method_def = ''
    method_sig = ''
    depth = 0
    method_collected = False
    in_string = False

    PRINTING=False

    line_nums = []
    for i in range(start_line, len(lines)):
        prev_char = ''
        line = lines[i]

        # if line.strip() ==   "public void test15()  throws Throwable  {":
            # PRINTING = True

        for col, char in enumerate(line):
            if char == '/' and prev_char == '/' and not in_string:
                prev_char = ''
                break
            if char == "\""  and not prev_char == "\\": 
                in_string = not in_string
                if PRINTING: print(line.strip(),':', char, in_string)
            elif char == '{' and not in_string:
                depth += 1
                if PRINTING: print(line.strip(), ':',char, in_string, '{', depth)
                #print('+', col, depth)

                if depth == 1: # open def, grab signature
                    method_sig = method_def + line[:col].strip() + ';'
            elif char == '}' and not in_string:
                depth -= 1
                if PRINTING: print(line.strip(), ':',char, in_string, '}', depth)
                #print('-', col, depth)
                if depth == 0: # end of method def
                    method_def += line[:col+1]
                    line_nums += [i+1]
                    method_collected = True
                    #print("Method collected")
                    break
        
            prev_char = char
        if method_collected:
            break
            
        method_def += line
        line_nums += [i+1]
    
    
    # method_sig = whitespace_re.sub(' ', method_sig).strip()
    # method_def = whitespace_re.sub(' ', method_def).strip()
    
    return method_sig, method_def, line_nums




def get_class_dec(test_file):
    try:
        with open(test_file) as f:
            class_txt = f.read()
            
        with open(test_file) as f:
            class_lines = f.readlines()
    
    except Exception as e:
        print('ERROR READING:', test_file)
        raise e

    try:
        tree = javalang.parse.parse(class_txt)
    except Exception as e:
        print("error parsing", test_file)
        raise e

    class_dec = tree.types[0]

    return class_dec, class_lines


def get_classes_with_inherited(full_class_path, src_path):
    
    ret_list = []

    while full_class_path:

        try:
            class_dec, class_lines = get_class_dec(full_class_path)
        except Exception as e:
            print('ERROR parsing', full_class_path)
            if ret_list: 
                return ret_list
            else:
                raise e

        ret_list += [(class_dec, class_lines)]
    
        full_class_path = None

        # get import list
        imports = {}
        for line in class_lines:
            if line.strip().startswith('import'):
                imported = line.strip().strip(';').split()[-1]
                import_cls = imported.split('.')[-1]
                imports[import_cls] = imported

        if hasattr(class_dec, 'extends') and class_dec.extends and class_dec.extends.name:
            extend_cls = class_dec.extends.name
            if extend_cls in imports:
                extend_full_cls = imports[extend_cls]
                full_class_path = src_path +'/'+ extend_full_cls.replace('.', '/') + '.java'

                print(full_class_path)
                print(class_dec.extends.name)
                print()


    return ret_list


        


def extract_all_methods(class_dec, class_lines):
    methods = []

    for method in class_dec.constructors:
        method_sig, method_def, line_nums = get_method_txt(class_lines, method.position.line-1)
        if method_def.count("@Test") > 1:
            continue

        methods.append((method, method_def, line_nums))

    for method in class_dec.methods:
        #get_focal_method_name(method)
        method_sig, method_def, line_nums = get_method_txt(class_lines, method.position.line-1)
        if method_def.count("@Test") > 1:
            continue

        methods.append((method, method_def, line_nums))

        #print(method_def)
    # print('collecting methods from ', class_dec.name)
    # for m in methods:
        # print(m[1])
    return methods


def split_test(test, line_nums, assert_line=None):
    # split by asserts
    split_tests = []
    relevant_lines = []
    for line, line_no in zip(test_method.split('\n'), line_nums):
        if not line.strip():
            continue

        if 'assert' in line:
            if assert_line is not None:
                if line_no == assert_line_no:
                    relevant_lines += [line]
                    relevant_lines += ['}']
                    split_tests += ['\n'.join(split_tests)]
                    break
            else: # no assert_line specified
                next_test = '\n'.join(relevant_lines + [line, '}'])
                split_tests += [next_test]

        else: # non assert line
            relevant_lines += [line]

    # if test did not have any asserts, just keep full test
    if not split_tests:
        split_tests += ['\n'.join(relevant_lines)]

    return split_tests



def gen_notnull_assert(assertion):
    # strip outer parens
    args_str = assertion[assertion.find('(')+1:assertion.rfind(')')]

    # split on nonstring ,
    in_str = False
    args = [], prev = 0
    for i, c in enumerate(args_str):
        if c == '"':
            in_str = not in_str
        if c == ',':
            args += [ args_str[prev:i] ]
            prev = i+1
    args += [ args_str[prev:] ]

    assert(len(args) == 2 or len(args) == 1)

    # ASSUME rightmost arg is variable (always case for evosuite):
    return 'assertNotNull('+args[-1]+')'




if __name__ == "__main__": 
    input_data = []
    metadata = []

    bug_assert_tests_df = pd.read_csv(DATA_DIR+'bug_catching_assert_tests.csv')
    bug_assert_tests_ids = set(bug_assert_tests_df.test_id)
    bug_assert_tests_ids_list = list(bug_assert_tests_df.test_id)
    # bug_assert_tests = {}

    bug_assert_tests_df['bug_id'] = bug_assert_tests_df['project'] + bug_assert_tests_df['bug'].astype(str)
    caught_assert_bug_ids = set(bug_assert_tests_df.bug_id)

    bug_assert_tests_df = bug_assert_tests_df.set_index('test_id', drop=True)


    for root, dirs, files in os.walk(GEN_TESTS_DIR):
        for f in files:
            if not f.endswith("_ESTest.java"): continue            
            full_fname = os.path.join(root, f)

            match = path_re.search(full_fname)
            if not match:
                continue
            print(full_fname)
            project = match.group(1) 
            bug_num = match.group(2) 
            class_path = match.group(3)

            # if not bug_num.endswith('0'):
                # continue

            # bug_num = str(int(bug_num)//100)

            #if not project + bug_num in caught_assert_bug_ids:
            #    continue

            d4j_project_dir = d4j_path+'/framework/projects/' + project

            project_layout = get_project_layout(d4j_project_dir)
            _, bug_scm_hashes = get_active_bugs(d4j_project_dir)

            try:
                bug_hash = bug_scm_hashes['revision.id.buggy'][int(bug_num)]
                src_dir = project_layout['src_dir'][bug_hash]
            except:
                print('ERROR: no bug hash/dir for', bug_num, project_dir)
                #sys.exit(1)
                continue

            project_dir = checkout_project(project, bug_num)
            full_class_path = os.path.join(project_dir, src_dir, class_path) + ".java"
            if not os.path.exists(full_class_path):
                print('ERROR: cannot get file:')
                print(full_class_path)
                continue


            #open(full_fname).read()
            try:
                class_dec, class_text = get_class_dec(full_fname)
            except Exception as e:
                print("ERROR:couldn't parse test_class", full_fname)
                continue


            try:
                src_path = os.path.join(project_dir, src_dir)
                focal_dec_text_pairs = get_classes_with_inherited(full_class_path, src_path)
                # focal_class_dec, focal_class_text = get_class_dec(full_class_path)
            except Exception as e:
                print("ERROR:couldn't parse focal class", project, bug_num, full_class_path)
                raise e
                continue


            package = ''
            for line in class_text:
                if m := extract_package_re.match(line.strip()):
                    package = m[1]
                    break

            jp.parse_file(full_fname)
            class_test_methods = jp.get_all_method_bodies()

            assert len(class_test_methods) == 1


            class_name, _ = list(class_test_methods.items())[0]



            test_methods = extract_all_methods(class_dec, class_text)
            clean_test_methods = []
            for obj, test_method, line_nums in test_methods:
                m2 = test_name_re.search(test_method)
                if not m2:
                    continue
                test_name = m2.group(1)
                full_test_name = package +'.' + class_name + '::' + test_name
                full_test_id = project+str(bug_num)+full_test_name


                split_tests = []
                if BUG_ONLY and full_test_id in bug_assert_tests_ids:
                    assert_line_no = bug_assert_tests_df.line_no[full_test_id]
                    split_tests = split_test(test_method, line_nums, assert_line_no)
                elif not BUG_ONLY:
                    split_tests = split_test(test_method, line_nums)

                clean_test_methods += [test_method]
                

            test_methods = clean_test_methods

            focal_class_methods = [extract_all_methods(fdec, ftxt) for fdec, ftxt in focal_dec_text_pairs]

            data_pairs = extract_focal_methods(class_dec, test_methods, focal_class_methods)
            write_data_pairs(project, bug_num, data_pairs)
            #WRITE THESE PAIRS SOMEWHERE! THEY ARE THE PREFIX, FOCAL METHOD PAIRS!
            #different file for each bug num and project probably...

            for test_method, focal_method in data_pairs:
                assertion = ''
                m = assert_re.search(test_method)
                if m:
                    assertion = m[0]

                m2 = test_name_re.search(test_method)
                test_name = m2.group(1)

                full_test_name = package +'.' + class_name + '::' + test_name
                test_id = project+str(bug_num)+full_test_name


                # TODO add bug metadata
                # TODO add exception lbl
                exception_bug = 0
                assertion_bug = 0
                exception_lbl = 0
                assert_err = ''
                assertion_lbl = assertion


                metadata += [(project, bug_num, full_test_name, exception_bug, assertion_bug, exception_lbl, assertion_lbl, assert_err, assertion_lbl)]
                input_data += [(test_method, focal_method)]

                print(test_id)
                print(test_method)
                # print()
                # print(clean_test)
                print()
                print(focal_method)
                print('-'*50)


    print(len(input_data), len(metadata))
    with open('assert_inputs.csv', 'w') as f1, open('assert_meta.csv', 'w') as f2:
        input_w = csv.writer(f1)
        meta_w = csv.writer(f2)

        input_w.writerow(['focal_method', 'test_prefix'])
        meta_w.writerow('project,bug_num,test_name,exception_bug,assertion_bug,exception_lbl,assert_err,assertion_lbl'.split(','))

        for input_pair, meta in zip(input_data, metadata):
            input_w.writerow(input_pair)
            meta_w.writerow(meta)

        if 0:
            for meta in bug_assert_tests_df.itertuples():
                # print(meta)
                test_id = meta.Index
                assertion, test_method, clean_test, focal_method = clean_data_dict[test_id]

                input_w.writerow([focal_method, test_method])

                exception_bug = 0
                assertion_bug = 1
                exception_lbl = 0
                assert_err = meta.assertion_error
                assertion_lbl = assertion
                # notnull_lbl = 0

                # if 'but was:<null>' in meta.assertion_error:
                    # notnull_lbl = 1
                    # notnull_assert = gen_notnull_assert(assertion)
                    # assertion_lbl += '<SEP>'

                # TODO any assertion processing? -1, etc?
                assertion_lbl = assertion_lbl.replace('assertEquals((-1)', 'assertEquals(-1')

                meta_w.writerow([meta.project, meta.bug, meta.failed_test, exception_bug, assertion_bug, exception_lbl, assert_err, assertion_lbl])

