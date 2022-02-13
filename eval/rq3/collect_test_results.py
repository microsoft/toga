import json, re, csv, random, math, javalang, os, random, argparse, sys
import subprocess as sp
import pandas as pd
from glob import glob
from tree_hugger.core import JavaParser

random.seed(0)


jp = JavaParser("../../tree-sitter-repos/my-languages.so")

whitespace_re = re.compile(r'\s+')
path_re = re.compile(r"\S+\/\S+\/(\S*)\/\S+\/([0-9]+)\/(\S*).java")

empty_test_re = re.compile(r'public void \S+\(\s*\)[^{]*{\s*}')

extract_trace_exception_re = re.compile(r'\.([\w\d\$]+Exception)', re.DOTALL)
extract_package_re = re.compile(r'package\s+(\S+);')

d4j_path = "../../../defects4j/"


def get_active_bugs(d4j_project_dir):
    active_bugs_df = pd.read_csv(d4j_project_dir + '/active-bugs.csv', index_col=0)
    bug_scm_hashes = active_bugs_df[['revision.id.buggy', 'revision.id.fixed']].to_dict()
    return active_bugs_df.index.to_list(), bug_scm_hashes


def get_project_layout(d4j_project_dir):
    project_layout_df = pd.read_csv(d4j_project_dir + '/dir-layout.csv',index_col=0, 
                                    names=['src_dir', 'test_dir'])
    return project_layout_df.to_dict()


def checkout_project(project, bug_num):
    outpath = "/tmp/{}_{}_buggy/".format(project, bug_num)
    if os.path.isdir(outpath): return outpath

    print("checking out {} {} into {}".format(project, bug_num, outpath))
    sp.call(["defects4j", "checkout", "-p", project, "-v", bug_num + "b", "-w", outpath])

    return outpath


def get_system_name(input_dir):
    log_dirs = glob(f'{input_dir}/results/bug_detection_log/*/*')
    for log_dir in log_dirs:
        if os.path.isdir(log_dir):
            system_name = log_dir.split('/')[-1]
            break

    return system_name


def extract_focal_class(class_dec):
    return class_dec.name.strip("_ESTest")
    # return class_dec.name.strip("_ESTest")


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
        jp.parse_file(full_fname)
    except Exception as e:
        print("error parsing", test_file)
        raise e

    return class_lines

def get_log_triggering_tests(log_f):
    trigger_log = ''
    try:
        with open(log_f) as f:
            trigger_log = f.read()
    except:
        pass

    exception_triggers = set()
    assertion_triggers = set()
    triggered_tests = set()

    # get list of failing tests
    trigger_traces = trigger_log.lstrip('---').split('\n---')

    total = 0

    for trace in trigger_traces:
        if not trace:
            continue

        total += 1
        
        lines = trace.split('\n')
        if len(lines) < 2:
            print(lines)
            sys.exit(1)
        test_name = lines[0].strip()

        assertion_bug = 'AssertionFailedError' in trace
        exception_bug = ('Exception' in trace or 'StackOverflowError' in trace)\
                            and not assertion_bug

        m = extract_trace_exception_re.search(lines[1])

        if 'Exception' in trace and not assertion_bug and not m:
            print('ERROR: missed exception')
            print(trace)
            print('-'*50)
        
        if exception_bug:
            exception_triggers.add(test_name)
        elif assertion_bug:
            assertion_triggers.add(test_name)

        triggered_tests.add(test_name)

        if not exception_bug and not assertion_bug:
            print('ERROR: trace not categorized')
            print(trace)
            print()

    unique = len(triggered_tests)

    return exception_triggers, assertion_triggers, triggered_tests, total, unique



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_dir')
    args = parser.parse_args()

    input_dir = args.input_dir
    
    all_test_methods = []
    buggy_exception_lbls, buggy_assertion_lbls, projects, bug_nums, test_names = [], [], [], [], []
    fixed_exception_lbls, fixed_assertion_lbls = [], []
    failed_buggy, failed_fixed = [], []
    evo_trace_exceptions, evo_expected_exceptions = [], []
    # trace_exceptions_dict = {}
    test_name_re = re.compile(r'public void (\S+)\(\s*\)')

    total_triggers = 0
    unique_triggers = 0
    n_processed_tests = 0

    system_name = get_system_name(input_dir)
    print(f'collecting results for {system_name}')

    bug_triggers_d = {}
    fix_triggers_d = {}
    matched_d = {}

    for root, dirs, files in os.walk(input_dir+"/aggregated_d4j_tests"):
        for f in files:
            if not f.endswith('.java'): continue
            if f.endswith('scaffolding.java'): continue

            full_fname = os.path.join(root, f)
            print(full_fname)

            match = path_re.search(full_fname)
            if not match:
                continue

            project = match.group(1) 
            bug_num = match.group(2) 
            class_path = match.group(3)

            bug_id = project + str(bug_num)


            project_dir = checkout_project(project, bug_num)

            d4j_project_dir = d4j_path+'/framework/projects/' + project


            project_layout = get_project_layout(d4j_project_dir)
            _, bug_scm_hashes = get_active_bugs(d4j_project_dir)

            # PROCESS TRIGGER LOG
            if bug_id in bug_triggers_d:
                bug_except_trigs, bug_assert_trigs, bug_triggers =\
                    bug_triggers_d[bug_id]
            else:
                buggy_log_f = f'{input_dir}/results/bug_detection_log/'+\
                            f'{project}/{system_name}/{bug_num}b.{bug_num}.trigger.log'

                bug_except_trigs, bug_assert_trigs, bug_triggers, total, unique = \
                    get_log_triggering_tests(buggy_log_f)

                bug_triggers_d[bug_id] = (bug_except_trigs, bug_assert_trigs, bug_triggers)

                total_triggers += total
                unique_triggers += unique


            if bug_id in fix_triggers_d:
                fix_except_trigs, fix_assert_trigs, fix_triggers =\
                    fix_triggers_d[bug_id]
            else:

                fixed_log_f = f'{input_dir}/results/bug_detection_log/'+\
                            f'{project}/{system_name}/{bug_num}f.{bug_num}.trigger.log'
                fix_except_trigs, fix_assert_trigs, fix_triggers, _, _ = \
                    get_log_triggering_tests(fixed_log_f)

                fix_triggers_d[bug_id] = (fix_except_trigs, fix_assert_trigs, fix_triggers)

            try:
                class_text = get_class_dec(full_fname)
            except Exception as e:
                print("ERROR:couldn't parse test_class", project, bug_num, full_fname)
                # continue
                raise e

            package = ''
            for line in class_text:
                if m := extract_package_re.match(line.strip()):
                    package = m[1]
                    break

            # get each test
            class_test_methods = jp.get_all_method_bodies()

            if not len(class_test_methods) == 1:
                print('ERROR PARSING', full_fname)
                continue


            class_name, test_methods = list(class_test_methods.items())[0]
            
            # prune empty tests
            test_methods = {name:md for (name, md) in test_methods.items() if not empty_test_re.search(md)}

            n_processed_tests += len(test_methods)


            matched = set()

            for test_name, test in test_methods.items():

                projects += [project]
                bug_nums += [bug_num]
                full_test_name = package +'.' + class_name + '::' + test_name
                test_names += [full_test_name]

                test_id = package +'.' + class_name + '::' + test_name

                buggy_exception_lbls += [bool(test_id in bug_except_trigs)]
                buggy_assertion_lbls += [bool(test_id in bug_assert_trigs)]
                failed_buggy += [bool(test_id in bug_triggers)]

                if test_id in bug_triggers:
                    matched.add(test_id)

                fixed_exception_lbls += [bool(test_id in fix_except_trigs)]
                fixed_assertion_lbls += [bool(test_id in fix_assert_trigs)]
                failed_fixed += [bool(test_id in fix_triggers)]

            not_matched = bug_triggers - matched

            all_test_methods += test_methods

    test_methods = [whitespace_re.sub(' ',test).strip() for test in all_test_methods]

    result_df = pd.DataFrame({
        'project': projects,
        'bug_num': bug_nums,
        'test_name': test_names,
        'bug_exception_triggered': buggy_exception_lbls,
        'bug_assertion_triggered': buggy_assertion_lbls,
        'failed_buggy': failed_buggy,
        'fixed_exception_triggered': fixed_exception_lbls,
        'fixed_assertion_triggered': fixed_assertion_lbls,
        'failed_fixed': failed_fixed,
        })
    buggy_sanity_check = (result_df['failed_buggy'] == (result_df['bug_exception_triggered'] | result_df['bug_assertion_triggered']))
    if not buggy_sanity_check.all():
        print('buggy_sanity_check')
        print(result_df[~buggy_sanity_check])
        print()

    fixed_sanity_check = (result_df['failed_fixed'] == (result_df['fixed_exception_triggered'] | result_df['fixed_assertion_triggered']))
    if not fixed_sanity_check.all():
        print('fixed_sanity_check')
        print(result_df[~fixed_sanity_check])
        print()


    result_df['TP'] = result_df['failed_buggy'] & (~result_df['failed_fixed'])
    result_df['FP'] = result_df['failed_buggy'] & (result_df['failed_fixed'])
    result_df['TN'] = (~result_df['failed_buggy']) & (~result_df['failed_fixed'])
    result_df['FN'] = (~result_df['failed_buggy']) & (result_df['failed_fixed'])

    TPs = result_df['TP'].sum()
    FPs = result_df['FP'].sum()
    TNs = result_df['TN'].sum()
    FNs = result_df['FN'].sum()

    test_P = TPs / (TPs + FPs)
    test_R = TPs / (TPs + FNs)
    test_F1 = 2*test_P*test_R / (test_P + test_R)

    result_df.to_csv(input_dir + '/test_data.csv', index=False)
    print('saved', input_dir + '/test_data.csv')

    grouped_bug = result_df.groupby(['project', 'bug_num']).sum()

    bugs_found = (grouped_bug['TP'] > 0).sum()
    FP_rate = FPs/(FPs + TNs)

    summary = pd.DataFrame([[f'{system_name}', bugs_found, FP_rate]],\
            columns=['approach', 'bugs_found', 'FP_rate'])

    print(summary)

    print(f'processed {n_processed_tests} extracted {len(test_methods)} missed {n_processed_tests - len(test_methods)}')
    print('total triggers', total_triggers, 'unique_triggers', unique_triggers)

