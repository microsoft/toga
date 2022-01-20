import pandas as pd
from collections import defaultdict
import os

df = pd.read_csv('./data/evosuite_regression_all/test_data.csv')
datapath = './data/evosuite_regression_all/results/bug_detection_log/'
illegal_arg_cnt = 0
total_assertion_fails = 0
assert_fails = 0

assert_value, expected_exception, wrong_exception, assert_bool, assert_null = 0, 0, 0, 0, 0
assert_unexpected_null = 0
broken_tests = 0
raised_exception = 0

bugs_covered = defaultdict(set)
bugs_covered_any = defaultdict(set)
# {
#     'assert_value' : set(),
#     'expected_exception': set(), 
#     'wrong_exception': set(), 
#     'assert_bool': set(),
# }
# assert_value_bugs, expected_exception_bugs, wrong_exception_bugs, assert_bool_bugs = set(), set(), set(), set()
df2 = df.groupby(['project', 'bug_num']).sum()
caught_bugs_df = df2[df2.TP > 0]

df['test_id'] = df.project + df.bug_num.astype(str) + df.test_name

TP_tests = set(df[df.TP].test_id)


bug_catching_assert_tests = []
bug_catching_tests = []

# for idx, row in df[df['found_bug'] & df['assertion_test']].iterrows():
for idx, row in caught_bugs_df.iterrows():
    
#     print(idx)
    project, bug = idx #(row['project'], row['bug_num'])
    
    run_num = 0
    for run_num in [0]:
        test_id = str(bug*100 + run_num)
        if not os.path.isfile(datapath + f'{project}/evosuite/{bug}b.{test_id}.trigger.log'):
            continue

        with open(datapath + f'{project}/evosuite/{bug}b.{test_id}.trigger.log') as f:
            traces = f.read().strip('---').split('\n---')
            
        for trace in traces:
            
            if 'StackOverflowError' in trace:
                continue
            
            assertion_error = ''
            exception_lbl = 0
            
            trace_lines = trace.split('\n')
            failed_test = trace_lines[0].strip()

            if project+str(bug)+failed_test not in TP_tests:
                print('skipping evosuite false positive test', project+str(bug)+failed_test)
                continue

            
            # get line_no
            failed_test_func = failed_test.replace('::', '.')
            line_no = -1
            for line in trace_lines:
                if failed_test_func in line:
                    line_no = int(line.split(':')[-1].strip(')'))
                    
            if line_no == -1:
                print('ERROR: line no')
                print(trace)
                sys.exit()
        
            
            if 'AssertionFailedError' in trace:
                total_assertion_fails += 1
                assertion_error = trace_lines[1]

                if 'junit.framework.AssertionFailedError:' in assertion_error:
                    assertion_error = assertion_error.split(':', 1)[1]
                elif 'junit.framework.AssertionFailedError' == assertion_error.strip():
                    for line in trace_lines[3:]:
                        if line.strip().startswith('at org.junit.Assert'):
                            assertion_error = line.strip().split('at org.junit.Assert.')[1]
                        else:
                            break
                else:
                    print('PARSE ERROR')
                    print(trace)
                    sys.exit()
                    
                # get assert line no
#                 print(trace)
                

                if 'Cannot load/analyze class' in trace:
    #                 print(trace)
                    # NOTE: this appears to be broken test?
                    broken_tests += 1
                    continue

                print(project, bug, assertion_error)

                bug_type = 'assertion'
                if 'expected:' in assertion_error:
                    assert_value += 1
                    bugs_covered['assert_value'].add(project+str(bug))
                    bugs_covered_any['any_assert'].add(project+str(bug))
                    
                    bug_catching_assert_tests += [(project, bug, failed_test, assertion_error, line_no)]
                    
                    if 'null' in assertion_error:
                        assert_unexpected_null += 1
                        bugs_covered['assert_null'].add(project+str(bug))

    #                 bugs_covered['total_asserts'].add(project+str(bug))

                if 'Expecting exception:' in assertion_error:
                    expected_exception += 1
                    bugs_covered['expected_exception'].add(project+str(bug))
                    bugs_covered_any['any_exception'].add(project+str(bug))
                    bug_type = 'exception'
                    
                    exception_lbl = 1

                if 'Exception was not thrown in' in assertion_error:
                    wrong_exception += 1
                    bugs_covered['wrong_exception'].add(project+str(bug))
                    
                    exception_lbl = 1
                    bug_type = 'exception'
                    
                    # we can't get these, skip (exception_lbl will be 1, but not considered bug finding)
#                     continue

                if 'assertTrue' in assertion_error or 'assertFalse' in assertion_error:
                    assert_bool += 1
                    bugs_covered['assert_bool'].add(project+str(bug))
                    bugs_covered_any['any_assert'].add(project+str(bug))
                    
                    bug_catching_assert_tests += [(project, bug, failed_test, assertion_error, line_no)]

                if 'assertNull' in assertion_error or 'assertNotNull' in assertion_error:
                    assert_null += 1
                    bugs_covered['assert_null'].add(project+str(bug))
                    bugs_covered_any['any_assert'].add(project+str(bug))
                    
                    bug_catching_assert_tests += [(project, bug, failed_test, assertion_error, line_no)]
                    
                    

            elif 'Exception' in trace:
    #             print(trace)
                raised_exception += 1
                bugs_covered['raised_exceptions'].add(project+str(bug))
                bugs_covered_any['any_exception'].add(project+str(bug))
                bug_type = 'exception'
                
                assertion_error = ''.join(trace_lines[1].strip().split('java.lang.'))
                
#                 print(trace)
#                 break
                
            bug_catching_tests += [(project, bug, failed_test, bug_type, exception_lbl, assertion_error, line_no)]


bug_catching_tests = pd.DataFrame(bug_catching_tests, columns='project, bug_num, test_name, bug_type, exception_lbl, error, line_no'.split(', '))
bug_catching_tests.to_csv('data/bug_catching_tests.csv')
